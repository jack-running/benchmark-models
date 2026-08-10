"""Layer B driver plumbing — subprocess capture must survive hostile bytes."""

import os
import subprocess
import sys
import time

import harness_drivers as hd


def test_run_proc_survives_non_cp1252_bytes(tmp_path):
    """Regression: `text=True` decoded the child pipe with the locale codec
    (cp1252 on Windows); a byte like 0x9d then threw UnicodeDecodeError in
    subprocess's reader thread and left stdout=None, crashing _run_proc.
    Bytes are captured raw and decoded utf-8/errors=replace instead.
    """
    child = (
        "import os,sys\n"
        "os.write(1, b'{\"type\":\"session\",\"name\":\"\\x9d\"}\\n')\n"
        "os.write(2, b'stderr \\x9d\\n')\n"
    )
    cmd = [sys.executable, "-c", child]
    hi = hd._run_proc(cmd, str(tmp_path), dict(), 30, tmp_path, "reg")
    assert hi.exit_code == 0
    assert hi.timed_out is False
    raw = tmp_path.joinpath("reg.stdout.ndjson").read_bytes()
    assert b"\x9d" in raw  # byte preserved verbatim in the artifact
    # driver-side consumers re-read with utf-8 + errors=replace
    text = tmp_path.joinpath("reg.stdout.ndjson").read_text(
        encoding="utf-8", errors="replace")
    assert "session" in text


def test_run_proc_timeout_writes_partial_output(tmp_path):
    child = (
        "import os,time\n"
        "os.write(1, b'partial line\\n')\n"
        "time.sleep(30)\n"
    )
    cmd = [sys.executable, "-c", child]
    hi = hd._run_proc(cmd, str(tmp_path), dict(), 2, tmp_path, "reg2")
    assert hi.timed_out is True
    assert hi.exit_code == -1
    raw = tmp_path.joinpath("reg2.stdout.ndjson").read_bytes()
    assert b"partial line" in raw


def test_parse_ndjson_skips_garbage_lines():
    raw = '{"type":"session"}\nnot-json\n{"type":"message"}\n'
    events = hd._parse_ndjson(raw)
    assert len(events) == 2
    assert events[0]["type"] == "session"


def test_kill_tree_stops_grandchildren(tmp_path):
    """A budget kill must stop the whole subtree.

    subprocess's own timeout kills only the direct child; opencode/omp spawn
    workers that inherit the pipes and keep working, which is how a 120 s
    budget produced 398 s of wall clock. Compile checks cannot catch this, so
    exercise it for real on the host platform.
    """
    marker = tmp_path / "grandchild.log"
    grandchild = (
        "import time\n"
        f"f=open(r'{marker}','a')\n"
        "\nwhile True:\n    f.write('x'); f.flush(); time.sleep(0.1)\n"
    )
    child = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}])\n"
        "time.sleep(60)\n"
    )
    hi = hd._run_proc([sys.executable, "-c", child], str(tmp_path), dict(os.environ),
                      3, tmp_path, "tree")
    assert hi.timed_out is True
    time.sleep(1.0)
    size_after_kill = marker.stat().st_size if marker.exists() else 0
    time.sleep(1.5)
    size_later = marker.stat().st_size if marker.exists() else 0
    assert size_later == size_after_kill, "grandchild survived the budget kill"


def test_run_proc_records_budget(tmp_path):
    hi = hd._run_proc([sys.executable, "-c", "pass"], str(tmp_path), dict(os.environ),
                      42, tmp_path, "b")
    assert hi.budget_s == 42


def test_tagged_makes_per_run_artifacts_unique():
    a = hd._tagged("omp_m", "c1_add_function_s1")
    b = hd._tagged("omp_m", "c1_add_function_s2")
    assert a != b and "c1_add_function_s1" in a
    assert hd._tagged("omp_m", "") == "omp_m"


def test_omp_never_takes_window_from_num_ctx():
    """num_ctx must not reach OLLAMA_CONTEXT_LENGTH: omp's window is coupled
    to a 32768 output reserve, so window == num_ctx == 32768 leaves no input
    budget and drives a compaction loop."""
    assert hd.OmpDriver(num_ctx=32768)._safe_window() == 0
    assert hd.OmpDriver(omp_context=32768)._safe_window() == 0   # refused
    assert hd.OmpDriver(omp_context=65536)._safe_window() == 65536


def test_omp_cmd_has_no_self_cap_and_pins_thinking():
    d = hd.OmpDriver()
    assert d.thinking == "off"
    assert hd.HARNESS_TIMEOUTS["omp"] == hd.HARNESS_TIMEOUTS["opencode"]


# ── event parsing: real schemas from live runs ────────────────

_OMP_EVENTS = [
    {"type": "thinking_level_changed", "thinkingLevel": "high",
     "configured": "auto", "resolved": "high"},
    {"type": "auto_compaction_start", "reason": "threshold"},
    {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read",
     "args": {"path": "notes.txt"}},
    {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "read",
     "result": {"content": [{"type": "text", "text": "TOKEN-9"}]},
     "isError": False},
    {"type": "message_end", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "the token is TOKEN-9"}],
        "usage": {"input": 10059, "output": 90}}},
]

_OC_EVENTS = [
    {"part": {"tool": "glob", "state": {"input": {"filePath": "a.py"},
                                        "output": "a.py", "status": "completed"}}},
    {"part": {"type": "text", "text": "done looking"}},
]


def test_omp_tool_and_text_parsing():
    assert hd._scan_tool_names(_OMP_EVENTS) == ["read_file"]   # alias applied
    calls = hd._tool_use_parts(_OMP_EVENTS)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["path"] == "notes.txt"
    assert calls[0]["text"] == "TOKEN-9"        # nested content[].text
    assert calls[0]["ok"] is True
    assert hd._final_text_from_events(_OMP_EVENTS) == "the token is TOKEN-9"
    assert hd._scan_tokens(_OMP_EVENTS) == (10059, 90)
    assert hd._count_compactions(_OMP_EVENTS) == 1
    assert hd._thinking_level(_OMP_EVENTS) == "high"


def test_opencode_parsing_still_works():
    assert hd._scan_tool_names(_OC_EVENTS) == ["list_dir"]     # glob -> list_dir
    calls = hd._tool_use_parts(_OC_EVENTS)
    assert calls[0]["arguments"]["path"] == "a.py"             # filePath alias
    assert hd._final_text_from_events(_OC_EVENTS) == "done looking"


def test_tool_names_ignore_prose_mentions():
    """Was a substring scan over json.dumps(event): any text mentioning a tool
    name counted as a call, so 'finish' appeared on runs that called nothing."""
    events = [{"type": "message_end", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "I will call finish and run_tests soon"}]}}]
    assert hd._scan_tool_names(events) == []


def test_scan_tokens_ignores_tool_argument_named_input():
    events = [{"type": "tool_execution_end", "toolName": "bash",
               "result": {"content": [{"type": "text", "text": "ok"}]},
               "args": {"input": 999}}]
    assert hd._scan_tokens(events) == (-1, -1)


def test_scan_tokens_reads_opencode_step_finish_shape():
    """opencode nests usage at step_finish.part.tokens; the omp-focused
    rewrite regressed this to (-1,-1) until the scan became key-scoped."""
    events = [{"type": "step_finish", "part": {"tokens": {
        "total": 6716, "input": 6640, "output": 76, "reasoning": 0,
        "cache": {"write": 0, "read": 0}}}}]
    assert hd._scan_tokens(events) == (6640, 76)


def test_scan_tokens_prefers_last_cumulative_block():
    events = [
        {"type": "step_finish", "part": {"tokens": {"input": 100, "output": 5}}},
        {"type": "step_finish", "part": {"tokens": {"input": 900, "output": 40}}},
    ]
    assert hd._scan_tokens(events) == (900, 40)
