"""Offline tests for ollama_client pure helpers (no server)."""
from ollama_client import (ModelProfile, _normalise_tool_call,
                           effective_num_ctx)


def test_model_profile_tools_and_thinking():
    p = ModelProfile(name="m", capabilities=frozenset(["tools", "thinking"]),
                     context_length=8192, parameter_size="12.0B", is_cloud=False)
    assert p.has_tools is True
    assert p.has_thinking is True
    q = ModelProfile(name="m2", capabilities=frozenset(["completion"]),
                     context_length=0, parameter_size="", is_cloud=True)
    assert q.has_tools is False
    assert q.has_thinking is False
    assert q.is_cloud is True


def test_effective_num_ctx_clamps_to_model():
    small = ModelProfile(name="m", capabilities=frozenset(["tools"]),
                         context_length=8192, parameter_size="", is_cloud=False)
    assert effective_num_ctx(small, 65536) == 8192
    assert effective_num_ctx(small, 4096) == 4096
    unbounded = ModelProfile(name="m", capabilities=frozenset(["tools"]),
                             context_length=0, parameter_size="", is_cloud=False)
    assert effective_num_ctx(unbounded, 65536) == 65536


def test_normalise_tool_call_handles_string_arguments():
    tc = {
        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        "id": "call_1",
    }
    norm = _normalise_tool_call(tc)
    assert norm["name"] == "read_file"
    assert norm["arguments"] == {"path": "a.py"}
    assert norm["id"] == "call_1"


def test_normalise_tool_call_invalid_schema():
    tc = {"function": {"name": "grep", "arguments": "not-json"}, "id": "c"}
    norm = _normalise_tool_call(tc)
    assert norm["arguments"] == {}