#!/usr/bin/env python3
"""
Layer B harness drivers: real binaries behind one interface.

Isolation is mandatory. Every driver writes its config into the temp
workspace or a per-run data dir — never into the user's ~/.config/opencode
or ~/.omp — and pins the Ollama host explicitly because both 127.0.0.1:11434
and 192.168.0.149:11434 are reachable with different model sets.

Layer B never reads harness stdout for pass/fail; it re-runs the task's
verify() on the workspace afterwards. stdout events are diagnostics only.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama_client
from agent_workspace import Workspace

# The host every driver pins. Never rely on a harness default.
BENCH_HOST = "http://192.168.0.149:11434"

# The real tool names the workspace exposes (diagnostics only).
_KNOWN_TOOLS = {
    "read_file", "list_dir", "grep", "write_file",
    "edit_file", "run_tests", "finish",
}

# One budget for every harness. Asymmetric budgets made transfer_delta
# meaningless: omp was self-capping at 90 s via --max-time while opencode's
# worker subtree kept running past its own nominal 120 s (see _run_proc).
E2E_BUDGET_S = 300
HARNESS_TIMEOUTS = {"opencode": E2E_BUDGET_S, "omp": E2E_BUDGET_S,
                    "cline": E2E_BUDGET_S}

# omp resolves thinking level from "auto", which picked "high" for a 27b
# reasoning model and burned the whole budget before any tool call. Native
# Layer A sends no thinking directive, so pin omp to match it.
OMP_THINKING = "off"

# omp's per-model output reserve (maxTokens) as discovered from the registry.
# It is NOT overridable from omp's config surface (modelOverrides did not
# apply in testing), so a context window of exactly this size leaves zero
# input budget and drives omp into a compaction loop that discards tool
# output. Any window handed to omp must clear it by a usable margin.
OMP_OUTPUT_RESERVE = 32768
OMP_MIN_INPUT_BUDGET = 8192

# After a budget kill, how long to keep draining the pipes before giving up.
_KILL_DRAIN_S = 15

# Canonical workspace-tool vocabulary. Harness-native tool names map onto
# these so the SAME end-state verifiers grade every backend.
_TOOL_ALIASES = {
    "read": "read_file", "cat": "read_file", "view": "read_file",
    "write": "write_file", "create": "write_file", "overwrite": "write_file",
    "edit": "edit_file", "find_replace": "edit_file",
    "glob": "list_dir", "ls": "list_dir", "list": "list_dir",
    "bash": "run_tests", "powershell": "run_tests", "run_pytest": "run_tests",
    "complete": "finish", "finish": "finish",
}


@dataclass
class HarnessRun:
    exit_code: int
    wall_seconds: float = 0.0
    stdout_events: list = field(default_factory=list)
    tool_names: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)   # Episode-compatible view
    final_text: str = ""
    prompt_tokens: int = -1
    completion_tokens: int = -1
    raw_stdout_path: str = ""
    raw_stderr_path: str = ""
    timed_out: bool = False
    # Provenance for transfer_delta: without these a delta cannot be read as
    # a harness effect rather than a budget/context/thinking difference.
    budget_s: int = 0
    context_budget: int = -1      # what we told the harness; -1 = left alone
    thinking_level: str = ""      # as the harness resolved it
    compactions: int = 0          # auto-compaction rounds (drops tool output)
    thinking_parts: int = 0       # reasoning the model emitted anyway
    server_context: int = -1      # context the model was actually loaded with


class HarnessDriver:
    name: str = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def prepare(self, ws: Workspace, model: str,
                profile: ollama_client.ModelProfile) -> None:
        raise NotImplementedError

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int, run_tag: str = "") -> HarnessRun:
        raise NotImplementedError


def _which(names: list[str]) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _kill_tree(proc) -> None:
    """Kill the harness AND its descendants.

    subprocess's own timeout kills only the direct child. opencode/omp spawn
    worker subtrees that inherit the pipes and keep working — that is how a
    120 s budget produced 398 s of wall clock and end-state passes, while omp
    (which self-terminated on --max-time) got no such grace. Killing the tree
    makes the budget mean the same thing for every harness.
    """
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _run_proc(cmd: list, cwd: str, env: dict, timeout_s: int,
              data_dir: Path, tag: str) -> HarnessRun:
    """Run a harness binary, capturing raw bytes.

    Deliberately NOT `text=True`: that makes Python decode the child pipe
    with the locale encoding (cp1252 on Windows), which throws from inside
    subprocess's reader thread on a byte like 0x9d and leaves stdout=None.
    We capture bytes and decode with utf-8/errors=replace ourselves, so
    arbitrary harness output can never crash the runner.

    On budget expiry the whole process tree is killed, then the pipes are
    drained for a bounded extra window so partial output still lands on disk.
    """
    hi = HarnessRun(exit_code=-1, raw_stdout_path="", raw_stderr_path="")
    out = data_dir / f"{tag}.stdout.ndjson"
    err = data_dir / f"{tag}.stderr.txt"
    hi.raw_stdout_path = str(out)
    hi.raw_stderr_path = str(err)
    hi.budget_s = int(timeout_s)
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        hi.exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        hi.timed_out = True
        hi.exit_code = -1
        _kill_tree(proc)
        try:
            # communicate() keeps what it already buffered, so this returns
            # the partial output rather than discarding it.
            stdout, stderr = proc.communicate(timeout=_KILL_DRAIN_S)
        except subprocess.TimeoutExpired:
            stdout, stderr = b"", b""
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8")
    if isinstance(stderr, str):
        stderr = stderr.encode("utf-8")
    hi.wall_seconds = time.perf_counter() - start
    out.write_bytes(stdout or b"")
    err.write_bytes(stderr or b"")
    return hi


def _parse_ndjson(raw: str) -> list:
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _norm_args(arguments) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) or {}
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        return {}
    # opencode passes filePath/file_path, not path; normalize so the
    # workspace-vocabulary path checks in verifiers match.
    for aliassrc in ("filePath", "file_path", "pathname"):
        if aliassrc in arguments and "path" not in arguments:
            arguments["path"] = arguments[aliassrc]
    return arguments


def _result_text(result) -> str:
    """omp tool result -> text. {"content":[{"type":"text","text":...}]}."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content
                     if isinstance(c, dict) and c.get("type") == "text"]
            if parts:
                return "\n".join(p for p in parts if p)
        for key in ("output", "text", "result"):
            if isinstance(result.get(key), str):
                return result[key]
    return ""


def _iter_tool_events(events: list):
    """Yield (name, arguments, ok, text) from either harness's real schema.

    opencode: {"part": {"tool": ..., "state": {"input":..., "output":...}}}
    omp:      {"type":"tool_execution_end","toolName":...,"result":{...},
               "isError":bool}   (verified against a live omp --mode json run)
    """
    omp_started = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        # ── omp ────────────────────────────────────────────────
        if etype == "tool_execution_start" and ev.get("toolName"):
            omp_started[ev.get("toolCallId")] = ev.get("args")
            continue
        if etype == "tool_execution_end" and ev.get("toolName"):
            args = ev.get("args")
            if args is None:
                args = omp_started.get(ev.get("toolCallId"))
            yield (ev["toolName"], _norm_args(args),
                   not bool(ev.get("isError")), _result_text(ev.get("result")))
            continue
        # ── opencode ───────────────────────────────────────────
        part = ev.get("part")
        if isinstance(part, dict):
            tool = part.get("tool") or part.get("name")
            if tool:
                state = part.get("state") or {}
                args = state.get("input")
                if args is None:
                    args = state.get("arguments") or {}
                text = state.get("output") or state.get("result") or ""
                status = state.get("status", "completed")
                yield (tool, _norm_args(args),
                       status in ("completed", "success", "ok"),
                       _result_text(text) or str(text))


def _scan_tool_names(events: list) -> list:
    """Ordered unique canonical tool names actually invoked.

    Was a substring scan over json.dumps(event), which matched any mention of
    a tool name in prose — 'finish' showed up on runs that called nothing.
    """
    found = []
    for name, _args, _ok, _text in _iter_tool_events(events):
        canon = _TOOL_ALIASES.get(name, name)
        if canon not in found:
            found.append(canon)
    return found


def _tool_use_parts(events: list) -> list:
    """Harness-native tool events -> [{"name","arguments","ok","text"}]."""
    return [{"name": _TOOL_ALIASES.get(name, name), "arguments": args,
             "ok": ok, "violation": None, "text": str(text)}
            for name, args, ok, text in _iter_tool_events(events)]


def _final_text_from_events(events: list) -> str:
    """Assistant text, so fabrication/grounding verifiers see an answer.

    opencode: {"part": {"type":"text","text":...}}
    omp:      {"type":"message_end","message":{"role":"assistant",
               "content":[{"type":"text","text":...}]}}
    """
    texts = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        part = ev.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
        if ev.get("type") == "message_end":
            msg = ev.get("message")
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                for c in (msg.get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        texts.append(c.get("text", ""))
    return " ".join(t for t in texts if t).strip()


def _count_compactions(events: list) -> int:
    """Auto-compaction rounds. Compaction can drop the tool output the task
    depends on (observed live: model answered 'unable to extract any text'
    after its read result was compacted away), so it is a first-class
    diagnostic, not noise.
    """
    return sum(1 for ev in events if isinstance(ev, dict)
               and ev.get("type") == "auto_compaction_start")


def _thinking_level(events: list) -> str:
    for ev in events:
        if isinstance(ev, dict) and ev.get("type") == "thinking_level_changed":
            return str(ev.get("resolved") or ev.get("thinkingLevel") or "")
    return ""


def _count_thinking(events: list) -> int:
    """Assistant content parts that are model reasoning.

    Distinct from _thinking_level, which only reports the level the harness
    *asked* for: a model can keep emitting thinking blocks even with
    --thinking off (observed on qwen3.6-unsloth-vl-agent:27b-112k, where a
    trivial one-file task still took 598 s). Without this, a report shows
    "thinking off" next to a run that spent its whole budget reasoning.
    """
    n = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message")
        if ev.get("type") == "message_end" and isinstance(msg, dict) \
                and msg.get("role") == "assistant":
            for c in (msg.get("content") or []):
                if isinstance(c, dict) and c.get("type") == "thinking":
                    n += 1
        # opencode reports reasoning tokens instead of parts
        for usage in _usage_blocks(ev):
            if isinstance(usage.get("reasoning"), int) and usage["reasoning"] > 0:
                n += 1
    return n


def _deep_int(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], int):
            return obj[key]
        for v in obj.values():
            r = _deep_int(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _deep_int(v, key)
            if r is not None:
                return r
    return None


def _usage_blocks(obj):
    """Yield dicts that are semantically token-usage blocks.

    Scoped by KEY name so a tool argument that happens to be called "input"
    can never be reported as a token count:
      omp      -> message_end.message.usage {input, output, totalTokens}
      opencode -> step_finish.part.tokens  {total, input, output, ...}
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in ("usage", "tokens") and isinstance(val, dict):
                yield val
            yield from _usage_blocks(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from _usage_blocks(val)


def _scan_tokens(events: list) -> tuple[int, int]:
    """(prompt, completion) tokens, from the last usage block seen.

    Both harnesses report cumulative usage, so the final block is the run
    total. Falls back to explicitly-named token keys.
    """
    pin = pout = None
    for ev in events:
        for usage in _usage_blocks(ev):
            if isinstance(usage.get("input"), int):
                pin = usage["input"]
            if isinstance(usage.get("output"), int):
                pout = usage["output"]
    if pin is None or pout is None:
        for ev in events:
            for key in ("inputTokens", "input_tokens", "promptTokens"):
                if pin is None:
                    pin = _deep_int(ev, key)
            for key in ("outputTokens", "output_tokens", "completionTokens"):
                if pout is None:
                    pout = _deep_int(ev, key)
    return (pin if pin is not None else -1, pout if pout is not None else -1)


def _finish_run(hi: HarnessRun, stdout: str, data_dir: Path,
                tag: str) -> HarnessRun:
    hi.stdout_events = _parse_ndjson(stdout)
    hi.tool_names = _scan_tool_names(hi.stdout_events)
    if not (hi.raw_stdout_path and hi.raw_stderr_path):
        hi.raw_stdout_path = str(data_dir / f"{tag}.stdout.ndjson")
        hi.raw_stderr_path = str(data_dir / f"{tag}.stderr.txt")
    return hi


def _tagged(base: str, run_tag: str) -> str:
    """Per-run artifact tag.

    The tag used to be model-only, so every run of a model overwrote the same
    stdout file and all 30 samples in a report pointed at the last one.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_tag)
    return f"{base}_{safe}" if safe else base


def _absorb_events(hi: HarnessRun) -> HarnessRun:
    """Parse the captured stdout into the diagnostic fields (shared)."""
    stdout = ""
    if hi.raw_stdout_path and Path(hi.raw_stdout_path).exists():
        stdout = Path(hi.raw_stdout_path).read_text(encoding="utf-8",
                                                    errors="replace")
    events = _parse_ndjson(stdout)
    hi.stdout_events = events
    hi.tool_names = _scan_tool_names(events)
    hi.tool_calls = _tool_use_parts(events)
    hi.final_text = _final_text_from_events(events)
    hi.prompt_tokens, hi.completion_tokens = _scan_tokens(events)
    hi.compactions = _count_compactions(events)
    hi.thinking_parts = _count_thinking(events)
    hi.thinking_level = _thinking_level(events)
    return hi


class OpenCodeDriver(HarnessDriver):
    """Project-level opencode.json has the highest config precedence."""

    name = "opencode"

    def __init__(self, host: str = BENCH_HOST, num_ctx: Optional[int] = None):
        self.host = host
        self.num_ctx = num_ctx
        self.data_dir = Path(tempfile.mkdtemp(prefix="hb_oc_"))
        self.context_budget = -1

    def available(self) -> tuple[bool, str]:
        return (True, "") if _which(["opencode", "opencode.cmd"]) \
            else (False, "opencode binary not on PATH")

    def prepare(self, ws: Workspace, model: str,
                 profile: ollama_client.ModelProfile) -> None:
        base = self.host + "/v1"
        # Cap the client-side context budget at the benchmark's num_ctx so the
        # harness is not budgeting against the model's full window while Layer
        # A ran at num_ctx. Neither harness can set Ollama's runtime num_ctx
        # (both speak an OpenAI-compatible API), so this equalises budgeting
        # only; the loaded context is recorded per run instead of assumed.
        ctx = profile.context_length
        if self.num_ctx:
            ctx = min(self.num_ctx, profile.context_length)
        self.context_budget = ctx
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "ollama": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Ollama Bench",
                    "options": {"baseURL": base},
                    "models": {model: {"name": model, "limit": {
                        "context": ctx, "output": 8192}}},
                }
            },
            "permission": {"bash": "allow", "edit": "allow", "webfetch": "deny"},
        }
        (ws.root / "opencode.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8")

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int, run_tag: str = "") -> HarnessRun:
        binp = _which(["opencode", "opencode.cmd"])
        tag = _tagged(f"oc_{model.replace(':','_')}", run_tag)
        cmd = [binp, "run", "--auto", "--format", "json", "-m",
               f"ollama/{model}", "--dir", str(ws.root), prompt]
        hi = _run_proc(cmd, str(ws.root), dict(os.environ), timeout_s,
                       self.data_dir, tag)
        hi.context_budget = self.context_budget
        return _absorb_events(hi)


class OmpDriver(HarnessDriver):
    """Oh My Pi. Host pinned via subprocess env only; user config untouched."""

    name = "omp"

    def __init__(self, host: str = BENCH_HOST, num_ctx: Optional[int] = None,
                 thinking: str = OMP_THINKING,
                 omp_context: Optional[int] = None):
        self.host = host
        # num_ctx is accepted for a uniform driver signature but deliberately
        # NOT used as omp's window: omp couples the window to a 32768 output
        # reserve, so window == num_ctx == 32768 leaves zero input budget.
        # Only the explicit omp_context knob can move omp's budgeting.
        self.num_ctx = num_ctx
        self.thinking = thinking
        self.omp_context = omp_context
        self.data_dir = Path(tempfile.mkdtemp(prefix="hb_omp_"))
        self.context_budget = -1

    def available(self) -> tuple[bool, str]:
        return (True, "") if _which(["omp.exe", "omp"]) \
            else (False, "omp binary not on PATH")

    def prepare(self, ws: Workspace, model: str,
                 profile: ollama_client.ModelProfile) -> None:
        pass  # host and context pinned per-run via env

    def _safe_window(self) -> int:
        """Reject a window that cannot fit the non-overridable output reserve.

        OLLAMA_CONTEXT_LENGTH does move omp's budgeting (verified: the target
        model reports 112000 unset, the pinned value when set) but NOT Ollama's
        runtime num_ctx. A window at or near OMP_OUTPUT_RESERVE reproduces the
        compaction loop, so refuse it loudly instead of shipping a config that
        silently discards tool output.
        """
        if not self.omp_context:
            return 0
        window = int(self.omp_context)
        if window - OMP_OUTPUT_RESERVE < OMP_MIN_INPUT_BUDGET:
            print(f"    omp: refusing OLLAMA_CONTEXT_LENGTH={window} — output "
                  f"reserve {OMP_OUTPUT_RESERVE} leaves < "
                  f"{OMP_MIN_INPUT_BUDGET} input tokens (compaction loop); "
                  f"leaving omp's discovered window in place")
            return 0
        return window

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int, run_tag: str = "") -> HarnessRun:
        binp = _which(["omp.exe", "omp"])
        env = dict(os.environ)
        env["OLLAMA_HOST"] = self.host
        env.setdefault("OLLAMA_BASE_URL", self.host)
        window = self._safe_window()
        if window:
            env["OLLAMA_CONTEXT_LENGTH"] = str(window)
            self.context_budget = window
        tag = _tagged(f"omp_{model.replace(':','_')}", run_tag)
        # No --max-time: the outer budget plus tree-kill governs every harness
        # identically. --max-time made omp the only harness that stopped its
        # own agent loop at the budget.
        cmd = [binp, "-p", "--mode", "json", "--model", f"ollama/{model}",
               "--cwd", str(ws.root), "--auto-approve", "--no-session",
               "--thinking", self.thinking, prompt]
        hi = _run_proc(cmd, str(ws.root), env, timeout_s, self.data_dir, tag)
        hi.context_budget = self.context_budget
        return _absorb_events(hi)


class ClineDriver(HarnessDriver):
    """Requires `npm install -g cline` (Node >= 22)."""

    name = "cline"

    def __init__(self, host: str = BENCH_HOST, num_ctx: Optional[int] = None):
        self.host = host
        self.num_ctx = num_ctx
        self.data_dir = Path(tempfile.mkdtemp(prefix="hb_cline_"))

    def available(self) -> tuple[bool, str]:
        if _which(["cline"]):
            return True, ""
        return False, "cline not installed; run `npm install -g cline` (Node >= 22)"

    def prepare(self, ws: Workspace, model: str,
                 profile: ollama_client.ModelProfile) -> None:
        env = dict(os.environ)
        env["CLINE_DATA_DIR"] = str(self.data_dir)
        subprocess.run(
            ["cline", "auth", "--provider", "ollama", "--modelid", model,
             "--baseurl", self.host],
            cwd=str(ws.root), env=env, capture_output=True,
            encoding="utf-8", errors="replace", timeout=120)

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int, run_tag: str = "") -> HarnessRun:
        binp = _which(["cline"])
        env = dict(os.environ)
        env["CLINE_DATA_DIR"] = str(self.data_dir)
        tag = _tagged(f"cl_{model.replace(':','_')}", run_tag)
        cmd = [binp, "-P", "ollama", "-m", model, "-c", str(ws.root),
               "-t", str(int(timeout_s)), "--yolo", "--json", prompt]
        hi = _run_proc(cmd, str(ws.root), env, timeout_s, self.data_dir, tag)
        return _absorb_events(hi)


class NativeDriver(HarnessDriver):
    """Wrap the Layer A loop behind the same interface (used in benchmark_agent)."""

    name = "native"

    def __init__(self, host: str = BENCH_HOST, num_ctx: Optional[int] = None):
        self.host = host
        self.num_ctx = num_ctx

    def available(self) -> tuple[bool, str]:
        return True, ""

    def prepare(self, ws: Workspace, model: str,
                 profile: ollama_client.ModelProfile) -> None:
        pass

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int, run_tag: str = "") -> HarnessRun:
        return HarnessRun(exit_code=0)


def build_drivers(num_ctx: Optional[int] = None,
                  omp_context: Optional[int] = None) -> dict:
    """Drivers with their context knobs. omp_context is separate from num_ctx
    on purpose — see OmpDriver._safe_window."""
    drivers: dict[str, HarnessDriver] = {}
    for cls in (OpenCodeDriver, OmpDriver, ClineDriver):
        kwargs = {"num_ctx": num_ctx}
        if cls is OmpDriver:
            kwargs["omp_context"] = omp_context
        d = cls(**kwargs)
        if d.available()[0]:
            drivers[d.name] = d
    return drivers


def server_context_length(host: str, model: str) -> int:
    """Context the server actually loaded the model with, from /api/ps.

    Ground truth for the native-vs-harness context question: neither harness
    can set Ollama's runtime num_ctx, so the only honest way to report the
    effective window is to ask the server while the model is resident.
    """
    try:
        import requests
        r = requests.get(f"{host}/api/ps", timeout=10)
        for m in r.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return int(m.get("context_length") or -1)
    except Exception:
        return -1
    return -1


def harness_versions() -> dict[str, str]:
    out = {}
    probes = (("opencode", ["opencode", "opencode.cmd"]),
              ("omp", ["omp", "omp.exe"]),
              ("cline", ["cline"]))
    for name, candidates in probes:
        binp = _find_binary(candidates)
        if not binp:
            out[name] = "not-installed"
            continue
        try:
            r = subprocess.run([binp, "--version"], capture_output=True,
                               encoding="utf-8", errors="replace", timeout=30)
            ver = (r.stdout or r.stderr).strip().splitlines()
            out[name] = ver[0] if ver else "?"
        except Exception as e:
            out[name] = f"unavailable ({e.__class__.__name__})"
    return out


# small module-level alias used by drivers


def _find_binary(names) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None