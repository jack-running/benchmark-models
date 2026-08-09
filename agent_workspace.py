#!/usr/bin/env python3
"""
Real agent workspace: files on disk and a small real tool registry.

Nothing is copied from MockToolSandbox — in particular an unknown tool here
returns an error string, never a plausible success. The tool schemas are the
literal `parameters` payloads sent to the model, because harness realism
depends on them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable, Optional

# The only environment given to model-run subprocesses. Deliberately NOT
# os.environ: model-generated code must not see user credentials.
SAFE_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    "PYTHONDONTWRITEBYTECODE": "1",
}


class PathEscape(Exception):
    pass


class UnknownTool(Exception):
    pass


class SchemaViolation(Exception):
    """Raised when tool arguments do not match the tool's JSON schema."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


@dataclass
class ToolOutcome:
    text: str
    ok: bool = True
    violation: Optional[str] = None  # None | "unknown_tool" | "schema" | "path_escape"


@dataclass
class Workspace:
    root: Path

    @classmethod
    def create(cls, fixture: dict[str, str]) -> "Workspace":
        """Make a temp dir and write each relpath -> content."""
        root = Path(tempfile.mkdtemp(prefix="hb_"))
        ws = cls(root)
        for relpath, content in fixture.items():
            ws._write_raw(relpath, content)
        return ws

    def _write_raw(self, relpath: str, content: str):
        p = self._under(relpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def _under(self, relpath: str) -> Path:
        """Path inside the workspace; already safe for fixture writes."""
        root = self.root.resolve()
        p = (root / relpath).resolve()
        if not p.is_relative_to(root):
            raise PathEscape(relpath)
        return p

    def resolve(self, p: str) -> Path:
        """Every model-facing tool goes through this; escapes are rejected."""
        root = self.root.resolve()
        target = (root / p).resolve()
        if not target.is_relative_to(root):
            raise PathEscape(p)
        return target

    def cleanup(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def snapshot(self) -> dict[str, str]:
        """Map relative paths to current contents (skips __pycache__)."""
        out: dict[str, str] = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(self.root)
                out[str(rel).replace(os.sep, "/")] = p.read_text(encoding="utf-8")
        return out


# ─────────────────────────────────────────────────────────────
# Args validation
# ─────────────────────────────────────────────────────────────

_TYPES = {"string", "integer", "boolean", "object", "array"}


def validate_arguments(schema: dict, arguments: dict) -> list[str]:
    """Hand-rolled schema check: required present, types match, no unknown keys."""
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]
    issues: list[str] = []
    props = schema.get("properties", {})
    required = schema.get("required", []) or []

    if "type" in schema and schema["type"] not in (None, "object"):
        issues.append(f"top-level type must be object, got {schema['type']}")

    for key in required:
        if key not in arguments:
            issues.append(f"missing required argument: '{key}'")

    for key, value in arguments.items():
        if key not in props:
            issues.append(f"unknown argument: '{key}'")
            continue
        want = props[key].get("type", "string")
        got = _typeof(value)
        if got != want:
            issues.append(f"argument '{key}' must be {want}, got {got}")

    return issues


def _typeof(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


class ToolRegistry:
    """The seven tools the native loop and the harnesses share."""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self._fns: dict[str, Callable] = {
            "read_file": self._read,
            "list_dir": self._list_dir,
            "grep": self._grep,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "run_tests": self._run_tests,
            "finish": self._finish,
        }

    # ── tool definitions (the literal `parameters` payloads) ──

    TOOL_DEFS: dict[str, dict] = {
        "read_file": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        "list_dir": {
            "name": "list_dir",
            "description": "List the entries of a directory. Directories are suffixed with '/'.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        "grep": {
            "name": "grep",
            "description": "Search for regex matches in a file. Returns relpath:lineno:line, max 50.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
        "write_file": {
            "name": "write_file",
            "description": "Create or overwrite a file with text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        "edit_file": {
            "name": "edit_file",
            "description": (
                "Replace an exact substring in an existing file. "
                "old_string must match exactly once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
        "run_tests": {
            "name": "run_tests",
            "description": "Run the project test suite (pytest) and return the tail of its output.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "finish": {
            "name": "finish",
            "description": "Signal that the task is complete, with a one-sentence summary.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    }

    def schemas(self, names: list[str]) -> list[dict]:
        """The `tools` array for /api/chat."""
        out = []
        for name in names:
            d = self.TOOL_DEFS.get(name)
            if d is None:
                raise UnknownTool(name)
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": d["name"],
                        "description": d["description"],
                        "parameters": d["parameters"],
                    },
                }
            )
        return out

    def execute(self, name: str, arguments: dict) -> ToolOutcome:
        fn = self._fns.get(name)
        if fn is None:
            return ToolOutcome(
                f"Error: unknown tool '{name}'. Available: "
                + ", ".join(sorted(self._fns)),
                ok=False,
                violation="unknown_tool",
            )
        schema = self.TOOL_DEFS.get(name)
        issues = validate_arguments(schema.get("parameters") or {}, arguments)
        if issues:
            return ToolOutcome(
                "Error: invalid arguments: " + "; ".join(issues),
                ok=False,
                violation="schema",
            )
        try:
            text = fn(**arguments)
        except PathEscape:
            return ToolOutcome(
                "Error: path escapes the workspace",
                ok=False,
                violation="path_escape",
            )
        except SchemaViolation as e:
            return ToolOutcome(
                "Error: " + "; ".join(e.issues), ok=False, violation="schema"
            )
        except Exception as e:
            return ToolOutcome(f"Error: {e}", ok=False)
        return ToolOutcome(text, ok=True)

    # ── tool bodies ───────────────────────────────────────────────

    def _read(self, path: str) -> str:
        p = self.ws.resolve(str(path))
        if not p.is_file():
            return f"Error: no such file: {path}"
        return p.read_text(encoding="utf-8")

    def _list_dir(self, path: str) -> str:
        p = self.ws.resolve(str(path))
        if not p.is_dir():
            return f"Error: no such directory: {path}"
        entries = []
        for child in sorted(p.iterdir()):
            entries.append(child.name + ("/" if child.is_dir() else ""))
        return "\n".join(entries) if entries else "(empty)"

    def _grep(self, pattern: str, path: str) -> str:
        p = self.ws.resolve(str(path))
        if not p.is_file():
            return f"Error: no such file: {path}"
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        lines = []
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                lines.append(f"{path}:{lineno}:{line}")
                if len(lines) >= 50:
                    lines.append("... (truncated at 50 matches)")
                    break
        return "\n".join(lines) if lines else f"No matches for {pattern!r} in {path}"

    def _write_file(self, path: str, content: str) -> str:
        p = self.ws.resolve(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        data = content if isinstance(content, str) else str(content)
        p.write_text(data, encoding="utf-8")
        return f"Wrote {len(data)} bytes to {path}"

    def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        p = self.ws.resolve(str(path))
        if not p.is_file():
            return f"Error: no such file: {path}"
        text = p.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found"
        if count > 1:
            return f"Error: old_string is not unique ({count} matches)"
        p.write_text(text.replace(old_string, new_string), encoding="utf-8")
        return f"Replaced 1 occurrence in {path}"

    def _run_tests(self) -> str:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.ws.root),
                capture_output=True,
                text=True,
                timeout=60,
                env=dict(SAFE_ENV),
            )
        except FileNotFoundError:
            return "Error: pytest unavailable"
        except subprocess.TimeoutExpired:
            return "Tests timed out after 60s (exit_code=none)"
        out = (proc.stdout + "\n" + proc.stderr).strip()
        tail = "\n".join(out.splitlines()[-40:]) if out else "(no output)"
        return f"{tail}\n---\nexit_code={proc.returncode}"

    def _finish(self, summary: str) -> str:
        return f"Finished: {summary}"