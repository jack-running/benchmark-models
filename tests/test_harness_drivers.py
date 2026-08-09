"""Layer B driver plumbing — subprocess capture must survive hostile bytes."""

import subprocess
import sys

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
