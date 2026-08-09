#!/usr/bin/env python3
"""
Shared Ollama client for the harness agentic benchmark.

Drives POST /api/chat with native tool calling, compiler seeding, warm-up,
done_reason capture, and bounded retries. Used by:
  - benchmark_agent.py  (Layer A native loop; Layer B drivers)
  - benchmark_quality.py (correctness screen, via call_with_retry)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

REQUEST_TIMEOUT = 180
DEFAULT_KEEP_ALIVE = "10m"

# Errors worth retrying: transient server/transport conditions. A real
# capability failure (http_400) is NEVER retried.
_RETRYABLE_ERRORS = {
    "timeout",
    "http_500",
    "http_503",
    "connection",
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    capabilities: frozenset[str]
    context_length: int        # 0 if the server did not report it
    parameter_size: str        # "" if absent
    is_cloud: bool             # ":cloud" in name

    @property
    def has_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def has_thinking(self) -> bool:
        return "thinking" in self.capabilities


@dataclass
class ChatResult:
    """One POST /api/chat round-trip, normalised across streaming chunks."""

    content: str = ""
    thinking: str = ""
    tool_calls: list[dict] = None      # [{"name","arguments","id"}]
    done_reason: str = ""             # "stop" | "length" | "load" | ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    ttft_seconds: Optional[float] = None
    load_seconds: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


def probe_model(host: str, model: str, timeout: int = 30) -> ModelProfile:
    """Read /api/show and build a ModelProfile. Raises on unreachable host."""
    try:
        r = requests.post(f"{host}/api/show", json={"model": model}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"cannot reach {host}: {e}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"{host}/api/show timed out after {timeout}s") from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"/api/show returned {r.status_code} for {model!r}: {e}") from e

    caps = frozenset(data.get("capabilities", []))
    model_info = data.get("model_info", {}) or {}
    context_length = 0
    for key, value in model_info.items():
        if re.search(r"\.context_length$", key):
            context_length = int(value)
            break
    details = data.get("details", {}) or {}
    parameter_size = details.get("parameter_size", "") or ""
    is_cloud = ":cloud" in model
    return ModelProfile(
        name=model,
        capabilities=caps,
        context_length=context_length,
        parameter_size=parameter_size,
        is_cloud=is_cloud,
    )


def _normalise_tool_call(tc: dict) -> Optional[dict]:
    fn = tc.get("function") or {}
    name = fn.get("name")
    if not name:
        return None
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args, "id": tc.get("id")}


def chat(
    host: str,
    model: str,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    num_ctx: int = 4096,
    num_predict: int = 1024,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    think: bool = False,
    timeout: int = REQUEST_TIMEOUT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
) -> ChatResult:
    """POST /api/chat with streaming; accumulate content/thinking/tool_calls."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "think": think,
        },
    }
    if tools is not None:
        payload["tools"] = tools
    if seed is not None:
        payload["options"]["seed"] = seed
    if think:
        # Top-level is the documented location for Ollama >= 0.7 / Qwen3.
        payload["think"] = True

    result = ChatResult()
    start = time.perf_counter()
    first_tok: Optional[float] = None

    try:
        with requests.post(
            f"{host}/api/chat", json=payload, stream=True, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                chunk = json.loads(raw)
                msg = chunk.get("message") or {}
                tok = msg.get("content") or ""
                think_tok = msg.get("thinking") or ""
                if tok:
                    result.content += tok
                if think_tok:
                    result.thinking += think_tok
                for tc in msg.get("tool_calls") or []:
                    norm = _normalise_tool_call(tc)
                    if norm and norm not in result.tool_calls:
                        result.tool_calls.append(norm)
                if (tok or think_tok or result.tool_calls) and first_tok is None:
                    first_tok = time.perf_counter()
                    result.ttft_seconds = first_tok - start
                if chunk.get("done"):
                    result.done_reason = chunk.get("done_reason", "") or ""
                    result.prompt_tokens = int(chunk.get("prompt_eval_count", 0) or 0)
                    result.completion_tokens = int(chunk.get("eval_count", 0) or 0)
                    result.load_seconds = ((
                        int(chunk.get("load_duration", 0) or 0)
                    ) / 1_000_000_000)
                    break
    except requests.exceptions.Timeout:
        result.error = "timeout"
    except requests.exceptions.ConnectionError as e:
        result.error = "connection"
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        result.error = f"http_{status}"
    except Exception as e:  # JSON decode error or anything else
        result.error = repr(e)
    finally:
        result.wall_seconds = time.perf_counter() - start

    return result


def warmup(host: str, model: str, timeout: int = 300) -> bool:
    """One tiny call to force model load so later timings are comparable."""
    r = chat(
        host,
        model,
        [{"role": "user", "content": "Say hi."}],
        num_predict=1,
        temperature=0.0,
        timeout=timeout,
    )
    return r.error is None


def call_with_retry(
    host: str,
    model: str,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    num_ctx: int = 4096,
    num_predict: int = 1024,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    think: bool = False,
    timeout: int = REQUEST_TIMEOUT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
) -> ChatResult:
    """chat() with bounded retries. Retries only transient errors, 2s then 6s."""
    delays = [2.0, 6.0]
    attempt = 0
    while True:
        r = chat(
            host,
            model,
            messages,
            tools=tools,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            seed=seed,
            think=think,
            timeout=timeout,
            keep_alive=keep_alive,
        )
        if r.error is None:
            return r
        if r.error in _RETRYABLE_ERRORS:
            pass  # timeout / http_500 / http_503 / connection -> retry
        elif r.error.startswith("http_"):
            # Any other HTTP error (notably http_400) is a real capability
            # failure and must never be retried.
            return r
        else:
            pass  # repr'd exception -> treat as transient, retry
        if attempt >= 2:
            return r  # gave up; report the last error
        time.sleep(delays[attempt])
        attempt += 1


def effective_num_ctx(profile: ModelProfile, requested: int) -> int:
    """Clamp to the model's trained context; Ollama truncates silently past it."""
    if profile.context_length > 0:
        return min(requested, profile.context_length)
    return requested