#!/usr/bin/env python3
"""
Layer A: the hermetic agent loop.

Drives POST /api/chat with native tools against a Workspace. This is the
"own loop" layer: it isolates model capability from harness noise. Layer B
(real harnesses) reuses the same Workspace + verifiers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import ollama_client
from agent_workspace import ToolRegistry, Workspace

DEFAULT_MAX_STEPS = 12
DEFAULT_WALL_BUDGET_S = 300


@dataclass
class Episode:
    task_id: str
    model: str
    seed: int
    backend: str                      # "native" | "opencode" | "omp" | "cline"
    steps: int = 0
    terminated: bool = False          # ended with a no-tool-calls assistant turn
    hit_step_budget: bool = False
    hit_wall_budget: bool = False
    tool_calls: list[dict] = field(default_factory=list)
    # {name, arguments, ok, violation, text}
    messages: list[dict] = field(default_factory=list)
    final_text: str = ""
    schema_violations: int = 0
    unknown_tool_calls: int = 0
    path_escapes: int = 0
    repeated_call_max: int = 0
    wall_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False
    error: Optional[str] = None
    passed: Optional[bool] = None     # set by the benchmark after verify()
    verify_reason: str = ""


def run_episode(
    host: str,
    model: str,
    task: "object",
    workspace: Workspace,
    registry: ToolRegistry,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    wall_budget_s: float = DEFAULT_WALL_BUDGET_S,
    seed: int = 1,
    num_ctx: int = 32768,
    system_prompt: str = "",
    temperature: float = 0.2,
    chat_fn: Callable = ollama_client.chat,
) -> Episode:
    """Run one task to completion (or budget) and return the full Episode."""
    ep = Episode(task_id=task.id, model=model, seed=seed, backend="native")
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": task.user_prompt})

    tools = registry.schemas(task.tools)
    start = time.perf_counter()
    call_counts: dict[tuple, int] = {}

    try:
        for step in range(max_steps):
            if time.perf_counter() - start >= wall_budget_s:
                ep.hit_wall_budget = True
                break
            r = chat_fn(
                host,
                model,
                messages,
                tools=tools,
                num_ctx=num_ctx,
                num_predict=2048,     # deliberately generous: observe truncation
                temperature=temperature,
                seed=seed,
            )
            if r.error is not None:
                ep.error = r.error
                break
            ep.prompt_tokens += r.prompt_tokens
            ep.completion_tokens += r.completion_tokens
            if r.done_reason == "length":
                ep.truncated = True

            if not r.tool_calls:
                # No tool calls -> assistant answer -> terminal.
                messages.append({"role": "assistant", "content": r.content})
                ep.final_text = r.content
                ep.terminated = True
                break

            # Assistant message carrying tool calls is appended verbatim.
            assistant_msg: dict = {"role": "assistant", "content": r.content}
            if r.thinking:
                assistant_msg["thinking"] = r.thinking
            assistant_msg["tool_calls"] = [
                {"id": tc.get("id"), "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in r.tool_calls
            ]
            messages.append(assistant_msg)

            # Execute each call in order; parallel calls run, not collapsed.
            for tc in r.tool_calls:
                outcome = registry.execute(tc["name"], tc["arguments"])
                key = (tc["name"], json.dumps(tc["arguments"], sort_keys=True))
                call_counts[key] = call_counts.get(key, 0) + 1
                if outcome.violation == "schema":
                    ep.schema_violations += 1
                elif outcome.violation == "unknown_tool":
                    ep.unknown_tool_calls += 1
                elif outcome.violation == "path_escape":
                    ep.path_escapes += 1
                ep.tool_calls.append({
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "ok": outcome.ok,
                    "violation": outcome.violation,
                    "text": outcome.text,
                })
                messages.append({
                    "role": "tool",
                    "content": outcome.text,
                    "tool_name": tc["name"],
                })
                if tc["name"] == "finish":
                    ep.terminated = True
                    ep.final_text = outcome.text
                    break
            if ep.terminated:
                break
            ep.steps = step + 1
    finally:
        ep.wall_seconds = time.perf_counter() - start

    if ep.error is None and not ep.terminated and not ep.hit_wall_budget:
        ep.hit_step_budget = True
    ep.repeated_call_max = max(call_counts.values(), default=0)
    ep.messages = messages
    return ep