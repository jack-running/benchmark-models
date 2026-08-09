"""Offline tests for the native loop via a fake chat_fn (no server)."""
from agent_loop import run_episode
from agent_tasks import TASKS_BY_ID
from agent_workspace import ToolRegistry, Workspace
from ollama_client import ChatResult


def _result(tool_calls=None, content="", done="stop", error=None):
    return ChatResult(content=content, tool_calls=tool_calls or [],
                      done_reason=done, error=error, wall_seconds=0.1,
                      prompt_tokens=10, completion_tokens=5)


def test_plain_answer_terminates():
    """No tool calls on the first turn -> terminal assistant answer."""
    def chat_fn(*a, **k):
        return _result(tool_calls=[], content="I'm done")
    ws = Workspace.create(TASKS_BY_ID["g1_probe_read"].fixture)
    try:
        ep = run_episode("http://h", "m", TASKS_BY_ID["g1_probe_read"], ws,
                         ToolRegistry(ws), chat_fn=chat_fn)
        assert ep.terminated is True
        assert ep.final_text == "I'm done"
        assert ep.hit_step_budget is False
        assert ep.error is None
    finally:
        ws.cleanup()


def test_step_budget_hit_when_always_tool_calls():
    """Never answer => burn the step budget, terminate via finish => stop."""
    def chat(*a, **k):
        return _result(tool_calls=[{"name": "finish", "arguments": {"summary": "s"}}],
                       content="")
    # probe task has finish in its tool set; always finish -> 1 step
    ws = Workspace.create(TASKS_BY_ID["g1_probe_read"].fixture)
    try:
        ep = run_episode("http://h", "m", TASKS_BY_ID["g1_probe_read"], ws,
                         ToolRegistry(ws), chat_fn=chat,
                         max_steps=10)
        assert ep.terminated is True       # finished
        assert ep.steps == 0
        assert ep.tool_calls[-1]["name"] == "finish"
    finally:
        ws.cleanup()


def test_schema_violation_counts():
    bad = {"name": "read_file", "arguments": {"path": 123}}  # int path
    state = {"turns": 0}

    def chat(*a, **k):
        state["turns"] += 1
        if state["turns"] == 1:
            return _result(tool_calls=[bad], content="")
        return _result(tool_calls=[], content="never mind")

    ws = Workspace.create(TASKS_BY_ID["g1_probe_read"].fixture)
    try:
        ep = run_episode("http://h", "m", TASKS_BY_ID["g1_probe_read"], ws,
                         ToolRegistry(ws), chat_fn=chat,
                         max_steps=5)
        assert ep.schema_violations == 1
        assert ep.tool_calls[0]["ok"] is False
        assert ep.tool_calls[0]["violation"] == "schema"
    finally:
        ws.cleanup()


def test_error_short_circuits_loop():
    def chat(*a, **k):
        return _result(error="http_400")
    ws = Workspace.create(TASKS_BY_ID["g1_probe_read"].fixture)
    try:
        ep = run_episode("http://h", "m", TASKS_BY_ID["g1_probe_read"], ws,
                         ToolRegistry(ws), chat_fn=chat,
                         max_steps=5)
        assert ep.error == "http_400"
        assert ep.terminated is False
    finally:
        ws.cleanup()


def test_wall_budget_flagged():
    import time

    def chat(*a, **k):
        time.sleep(0.05)
        return _result(tool_calls=[{"name": "read_file", "arguments": {"path": "notes.txt"}}],
                       content="")
    ws = Workspace.create(TASKS_BY_ID["g1_probe_read"].fixture)
    try:
        ep = run_episode("http://h", "m", TASKS_BY_ID["g1_probe_read"], ws,
                         ToolRegistry(ws), chat_fn=chat, max_steps=100,
                         wall_budget_s=0.1)
        assert ep.hit_wall_budget is True
    finally:
        ws.cleanup()