"""Offline tests for the hermetic workspace + tools (no server)."""
import os

import pytest

from agent_workspace import (PathEscape, SAFE_ENV, ToolRegistry,
                             Workspace, validate_arguments)


FIXTURE = {"app/a.py": "x = 1\n", "app/b.py": "hello world\n"}


def test_workspace_create_and_snapshot():
    ws = Workspace.create(FIXTURE)
    try:
        snap = ws.snapshot()
        assert snap == FIXTURE
    finally:
        ws.cleanup()


def test_resolve_rejects_escape():
    ws = Workspace.create(FIXTURE)
    try:
        with pytest.raises(PathEscape):
            ws.resolve("../outside.txt")
        with pytest.raises(PathEscape):
            ws.resolve("app/../../etc/passwd")
    finally:
        ws.cleanup()


def test_registry_schemas_match_task_tools():
    ws = Workspace.create(FIXTURE)
    try:
        reg = ToolRegistry(ws)
        schemas = reg.schemas(["read_file", "finish"])
        names = {s["function"]["name"] for s in schemas}
        assert names == {"read_file", "finish"}
        with pytest.raises(Exception):
            reg.schemas(["does_not_exist"])
    finally:
        ws.cleanup()


def test_validate_arguments_required_and_types():
    schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern", "path"],
    }
    assert validate_arguments(schema, {"pattern": "x", "path": "a"}) == []
    issues = validate_arguments(schema, {"pattern": "x"})
    assert any("missing required argument" in i for i in issues)
    issues = validate_arguments(schema, {"pattern": "x", "path": "a", "extra": 1})
    assert any("unknown argument" in i for i in issues)
    issues = validate_arguments(schema, {"pattern": 5, "path": "a"})
    assert any("must be string" in i for i in issues)


def test_tool_read_and_grep():
    ws = Workspace.create(FIXTURE)
    try:
        reg = ToolRegistry(ws)
        out = reg.execute("read_file", {"path": "app/b.py"})
        assert out.ok and "hello world" in out.text
        out = reg.execute("read_file", {"path": "missing.py"})
        assert out.ok and "no such file" in out.text
        g = reg.execute("grep", {"pattern": "hello", "path": "app/b.py"})
        assert g.ok and "hello" in g.text
    finally:
        ws.cleanup()


def test_tool_escape_is_caught():
    ws = Workspace.create(FIXTURE)
    try:
        reg = ToolRegistry(ws)
        out = reg.execute("read_file", {"path": "..%2Fsecret.txt"})
        out = reg.execute("read_file", {"path": "../secret.txt"})
        assert out.ok is False
        assert out.violation == "path_escape"
    finally:
        ws.cleanup()


def test_safe_env_never_contains_user_secrets():
    os.environ["SECRET_TEST"] = "SECRET_VALUE"
    assert "SECRET_TEST" not in SAFE_ENV
    assert "SECRET_VALUE" not in " ".join(SAFE_ENV.values())
    assert "PATH" in SAFE_ENV


def test_subprocess_env_does_not_leak_secrets(tmp_path):
    """V7: run_tests subprocess env must not expose host credentials."""
    import subprocess
    import sys
    os.environ["SECRET_TEST"] = "SECRET_VALUE"
    probe = "import os; print('SECRET_TEST' in os.environ)"
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True, timeout=30,
                          env=dict(SAFE_ENV))
    assert proc.stdout.strip() == "False"