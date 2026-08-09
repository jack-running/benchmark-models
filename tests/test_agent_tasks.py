"""Offline tests for agent_tasks: task registry + end-state verifiers."""
import agent_tasks as at
from agent_workspace import Workspace


def test_task_count_and_axes():
    assert len(at.ALL_TASKS) == 19
    ids = [t.id for t in at.ALL_TASKS]
    assert len(set(ids)) == 19
    assert len(at.E2E_TASKS) == 6
    # the binding acceptance: V3 needs 19 tasks x k episodes
    assert {t.id for t in at.TASKS_BY_ID.values()} == set(ids)


def test_axes_present():
    axes = {t.axis for t in at.ALL_TASKS}
    assert {"probe", "completion", "edit", "instruction", "grounding",
            "fabrication", "recovery"} <= axes


def test_grounding_marker_in_signal():
    assert at.GROUNDING_NONCE == "ANSWER-K9F2M"
    assert at.GROUNDING_NONCE in at.GROUNDING_CORPUS["pkg/signal.py"]
    corpus_chars = sum(len(c) for c in at.GROUNDING_CORPUS.values())
    assert len(at.GROUNDING_CORPUS) == 40
    assert 200_000 < corpus_chars < 400_000


def test_probe_verify_end_state():
    task = at.TASKS_BY_ID["g1_probe_read"]
    ws = at.Workspace.create(task.fixture)
    try:
        class Ep:
            tool_calls = []
            final_text = ""
        ep = Ep()
        ep.tool_calls = [{"name": "read_file", "arguments": {"path": "notes.txt"}}]
        ep.final_text = f"the token is {at.PROBE_NONCE}"
        ok, reason = task.verify(ws, ep)
        assert ok is True
        # missing the read call -> fails
        ep2 = Ep()
        ep2.tool_calls = []
        ep2.final_text = at.PROBE_NONCE
        ok2, _ = task.verify(ws, ep2)
        assert ok2 is False
    finally:
        ws.cleanup()


def test_fabrication_verifier_allows_only_read_ints():
    task = at.TASKS_BY_ID["g4_fabrication"]
    ws = at.Workspace.create(task.fixture)
    try:
        class Ep:
            tool_calls = []
            final_text = ""
        # fabricates an integer it never read -> fail
        ep = Ep()
        ep.tool_calls = [{"name": "list_dir", "arguments": {"path": "."},
                          "text": "config.yaml"}]
        ep.final_text = "timeout=4771"
        ok, reason = task.verify(ws, ep)
        assert ok is False
        assert "fabricated" in reason
    finally:
        ws.cleanup()


def test_fabrication_verifier_allows_number_seen_in_read():
    """A number grounded in a read_file output is legit (harness-agnostic)."""
    task = at.TASKS_BY_ID["g4_fabrication"]
    ws = at.Workspace.create(task.fixture)
    try:
        class Ep:
            tool_calls = []
            final_text = ""
        ep = Ep()
        ep.tool_calls = [{"name": "list_dir", "arguments": {"path": "."},
                          "text": "config.yaml\n"},
                         {"name": "read_file", "arguments": {"path": "config.yaml"},
                          "text": "timeout: 4771\n"}]
        ep.final_text = "The timeout is 4771"
        ok, reason = task.verify(ws, ep)
        assert ok is True
    finally:
        ws.cleanup()


def test_harness_rules_flag_violations():
    from agent_tasks import check_harness_rules
    task = at.TASKS_BY_ID["c1_add_function"]

    class Ep:
        def __init__(self, calls, terminated):
            self.tool_calls = calls
            self.terminated = terminated

    # no write on fixture, ended with finish, no repeats -> all ok
    ok_ep = Ep([
        {"name": "finish", "arguments": {"summary": "s"}},
    ], True)
    rules = check_harness_rules(task, ok_ep)
    assert all(r[0] for r in rules.values())

    # writing to a pre-existing fixture file violates R1
    bad_ep = Ep([
        {"name": "write_file", "arguments": {"path": "utils.py", "content": "x"}},
        {"name": "finish", "arguments": {"summary": "s"}},
    ], True)
    rules = check_harness_rules(task, bad_ep)
    assert rules["r1"][0] is False