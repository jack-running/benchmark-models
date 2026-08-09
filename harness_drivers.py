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

HARNESS_TIMEOUTS = {"opencode": 120, "omp": 90, "cline": 150}

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


class HarnessDriver:
    name: str = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def prepare(self, ws: Workspace, model: str,
                profile: ollama_client.ModelProfile) -> None:
        raise NotImplementedError

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int) -> HarnessRun:
        raise NotImplementedError


def _which(names: list[str]) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _run_proc(cmd: list, cwd: str, env: dict, timeout_s: int,
              data_dir: Path, tag: str) -> HarnessRun:
    """Run a harness binary, capturing raw bytes.

    Deliberately NOT `text=True`: that makes Python decode the child pipe
    with the locale encoding (cp1252 on Windows), which throws from inside
    subprocess's reader thread on a byte like 0x9d and leaves stdout=None.
    We capture bytes and decode with utf-8/errors=replace ourselves, so
    arbitrary harness output can never crash the runner.
    """
    hi = HarnessRun(exit_code=-1, raw_stdout_path="", raw_stderr_path="")
    out = data_dir / f"{tag}.stdout.ndjson"
    err = data_dir / f"{tag}.stderr.txt"
    hi.raw_stdout_path = str(out)
    hi.raw_stderr_path = str(err)
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                              timeout=timeout_s)
        stdout, stderr = proc.stdout or b"", proc.stderr or b""
        hi.exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        hi.timed_out = True
        hi.exit_code = -1
        stdout = e.stdout or b""
        stderr = e.stderr or b""
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8")
    if isinstance(stderr, str):
        stderr = stderr.encode("utf-8")
    hi.wall_seconds = time.perf_counter() - start
    out.write_bytes(stdout)
    err.write_bytes(stderr)
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


def _scan_tool_names(events: list) -> list:
    found = []
    for ev in events:
        s = json.dumps(ev)
        for name in sorted(_KNOWN_TOOLS):
            if name in s and name not in found:
                found.append(name)
    return found


def _tool_use_parts(events: list) -> list:
    """Harness-native tool events -> [{"name","arguments","ok","text"}].

    opencode emits {"type":"tool_use","part":{...,"tool":"glob","state":{...}}}.
    omp emits its own tool schema; we tolerate a few layouts here.
    """
    calls = []
    for ev in events:
        part = ev.get("part") if isinstance(ev, dict) else None
        if not isinstance(part, dict):
            continue
        tool = part.get("tool") or part.get("name")
        if not tool:
            continue
        state = part.get("state") or {}
        arguments = state.get("input")
        if arguments is None:
            arguments = state.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) or {}
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        # opencode passes filePath/file_path, not path; normalize so the
        # workspace-vocabulary path checks in verifiers match.
        for aliassrc in ("filePath", "file_path", "pathname"):
            if aliassrc in arguments and "path" not in arguments:
                arguments["path"] = arguments[aliassrc]
        text = state.get("output") or state.get("result") or ""
        status = state.get("status", "completed")
        calls.append({
            "name": _TOOL_ALIASES.get(tool, tool),
            "arguments": arguments,
            "ok": status in ("completed", "success", "ok"),
            "violation": None,
            "text": str(text),
        })
    return calls


def _final_text_from_events(events: list) -> str:
    """Grab the last assistant text so fabrication/grounding verifiers work."""
    texts = []
    for ev in events:
        part = ev.get("part") if isinstance(ev, dict) else None
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
    return " ".join(t for t in texts if t).strip()


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


def _scan_tokens(events: list) -> tuple[int, int]:
    pin = pout = None
    for ev in events:
        for key in ("input", "inputTokens", "input_tokens", "promptTokens"):
            if pin is None:
                pin = _deep_int(ev, key)
        for key in ("output", "outputTokens", "output_tokens",
                    "completionTokens"):
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


class OpenCodeDriver(HarnessDriver):
    """Project-level opencode.json has the highest config precedence."""

    name = "opencode"

    def __init__(self, host: str = BENCH_HOST):
        self.host = host
        self.data_dir = Path(tempfile.mkdtemp(prefix="hb_oc_"))

    def available(self) -> tuple[bool, str]:
        return (True, "") if _which(["opencode", "opencode.cmd"]) \
            else (False, "opencode binary not on PATH")

    def prepare(self, ws: Workspace, model: str,
                 profile: ollama_client.ModelProfile) -> None:
        base = self.host + "/v1"
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "ollama": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Ollama Bench",
                    "options": {"baseURL": base},
                    "models": {model: {"name": model, "limit": {
                        "context": profile.context_length, "output": 8192}}},
                }
            },
            "permission": {"bash": "allow", "edit": "allow", "webfetch": "deny"},
        }
        (ws.root / "opencode.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8")

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int) -> HarnessRun:
        binp = _which(["opencode", "opencode.cmd"])
        tag = f"oc_{model.replace(':','_')}"
        cmd = [binp, "run", "--auto", "--format", "json", "-m",
               f"ollama/{model}", "--dir", str(ws.root), prompt]
        hi = _run_proc(cmd, str(ws.root), dict(os.environ), timeout_s,
                       self.data_dir, tag)
        stdout = hi.raw_stdout_path and Path(hi.raw_stdout_path).read_text(
            encoding="utf-8", errors="replace") or ""
        events = _parse_ndjson(stdout)
        hi.stdout_events = events
        hi.tool_names = _scan_tool_names(events)
        hi.tool_calls = _tool_use_parts(events)
        hi.final_text = _final_text_from_events(events)
        hi.prompt_tokens, hi.completion_tokens = _scan_tokens(events)
        return hi


class OmpDriver(HarnessDriver):
    """Oh My Pi. Host pinned via subprocess env only; user config untouched."""

    name = "omp"

    def __init__(self, host: str = BENCH_HOST):
        self.host = host
        self.data_dir = Path(tempfile.mkdtemp(prefix="hb_omp_"))

    def available(self) -> tuple[bool, str]:
        return (True, "") if _which(["omp.exe", "omp"]) \
            else (False, "omp binary not on PATH")

    def prepare(self, ws: Workspace, model: str,
                 profile: ollama_client.ModelProfile) -> None:
        pass  # host pinned per-run via env

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int) -> HarnessRun:
        binp = _which(["omp.exe", "omp"])
        env = dict(os.environ)
        env["OLLAMA_HOST"] = self.host
        env.setdefault("OLLAMA_BASE_URL", self.host)
        tag = f"omp_{model.replace(':','_')}"
        cmd = [binp, "-p", "--mode", "json", "--model", f"ollama/{model}",
               "--cwd", str(ws.root), "--auto-approve", "--no-session",
               "--max-time", str(int(timeout_s)), prompt]
        hi = _run_proc(cmd, str(ws.root), env, timeout_s, self.data_dir, tag)
        stdout = Path(hi.raw_stdout_path).read_text(encoding="utf-8",
                                                    errors="replace")
        events = _parse_ndjson(stdout)
        hi.stdout_events = events
        hi.tool_names = _scan_tool_names(events)
        hi.tool_calls = _tool_use_parts(events)
        hi.final_text = _final_text_from_events(events)
        hi.prompt_tokens, hi.completion_tokens = _scan_tokens(events)
        return hi


class ClineDriver(HarnessDriver):
    """Requires `npm install -g cline` (Node >= 22)."""

    name = "cline"

    def __init__(self, host: str = BENCH_HOST):
        self.host = host
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
            timeout_s: int) -> HarnessRun:
        binp = _which(["cline"])
        env = dict(os.environ)
        env["CLINE_DATA_DIR"] = str(self.data_dir)
        tag = f"cl_{model.replace(':','_')}"
        cmd = [binp, "-P", "ollama", "-m", model, "-c", str(ws.root),
               "-t", str(int(timeout_s)), "--yolo", "--json", prompt]
        hi = _run_proc(cmd, str(ws.root), env, timeout_s, self.data_dir, tag)
        stdout = Path(hi.raw_stdout_path).read_text(encoding="utf-8",
                                                    errors="replace")
        events = _parse_ndjson(stdout)
        hi.stdout_events = events
        hi.tool_names = _scan_tool_names(events)
        hi.tool_calls = _tool_use_parts(events)
        hi.final_text = _final_text_from_events(events)
        hi.prompt_tokens, hi.completion_tokens = _scan_tokens(events)
        return hi


class NativeDriver(HarnessDriver):
    """Wrap the Layer A loop behind the same interface (used in benchmark_agent)."""

    name = "native"

    def __init__(self, host: str = BENCH_HOST):
        self.host = host

    def available(self) -> tuple[bool, str]:
        return True, ""

    def prepare(self, ws: Workspace, model: str,
                 profile: ollama_client.ModelProfile) -> None:
        pass

    def run(self, ws: Workspace, model: str, prompt: str,
            timeout_s: int) -> HarnessRun:
        return HarnessRun(exit_code=0)


def build_drivers() -> dict[str, HarnessDriver]:
    drivers: dict[str, HarnessDriver] = {}
    for cls in (OpenCodeDriver, OmpDriver, ClineDriver):
        d = cls()
        if d.available()[0]:
            drivers[d.name] = d
    return drivers


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