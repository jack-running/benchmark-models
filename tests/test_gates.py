"""Offline tests for gates: wilson CI, tier verdicts, gate ordering."""
import gates
from ollama_client import ModelProfile


def _profile(tools=True):
    caps = frozenset(["tools"]) if tools else frozenset(["completion"])
    return ModelProfile(name="m", capabilities=caps, context_length=0,
                        parameter_size="", is_cloud=False)


class _Ep:
    def __init__(self, task_id="c1", passed=True, tool_calls=1,
                 schema=0, unknown=0, escapes=0, terminated=True,
                 step_budget=False, wall_budget=False):
        self.task_id = task_id
        self.passed = passed
        self.tool_calls = [{"name": "x"}] * tool_calls
        self.schema_violations = schema
        self.unknown_tool_calls = unknown
        self.path_escapes = escapes
        self.terminated = terminated
        self.hit_step_budget = step_budget
        self.hit_wall_budget = wall_budget


def test_tier_for_verdicts():
    assert gates.tier_for(None, 0.9, {"edit": .9, "instruction": .8}) \
        == ("HARNESS_READY", None)
    assert gates.tier_for(None, 0.6, {"edit": .9, "instruction": .8}) \
        == ("SUPERVISED", None)
    assert gates.tier_for(None, 0.3, {}) == ("NOT_RECOMMENDED", None)
    assert gates.tier_for("G0_declares_tools", 0.9, {}) \
        == ("BLOCKED", "G0_declares_tools")


def test_wilson_ci_known_values():
    lo, hi = gates.wilson_ci(95, 95)
    assert abs(lo - 0.9613) < 0.001 and hi >= 0.999
    lo, hi = gates.wilson_ci(50, 95)
    assert lo > 0.42 and hi < 0.64


def test_g0_blocks_when_no_tools():
    eps = [_Ep()]
    res = gates.evaluate_gates(_profile(tools=False), eps, {})
    assert res["G0_declares_tools"].passed is False
    # everything after G0 is not evaluated
    assert "G1_emits_tool_call" not in res


def test_g1_blocks_when_probe_has_no_tool_calls():
    eps = [_Ep(task_id="g1_probe_read", tool_calls=0)]
    res = gates.evaluate_gates(_profile(), eps, {"c1": object()})
    assert res["G0_declares_tools"].passed is True
    assert res["G1_emits_tool_call"].passed is False


def test_g1_passes_from_the_smoke_probe_episode():
    """The probe episode is run in its own stage, not inside the task pool."""
    eps = [_Ep(task_id="c1", tool_calls=2)]
    res = gates.evaluate_gates(_profile(), eps, {"c1": object()},
                               probe_episodes=[_Ep(task_id="g1_probe_read",
                                                   tool_calls=1)])
    assert res["G1_emits_tool_call"].passed is True


def test_g1_blocks_when_the_smoke_probe_emitted_nothing():
    """Probe evidence, when present, decides the gate — even if other tasks
    called tools (the probe is the controlled one-tool measurement)."""
    res = gates.evaluate_gates(_profile(), [_Ep(task_id="c1", tool_calls=2)],
                               {"c1": object()},
                               probe_episodes=[_Ep(task_id="g1_probe_read",
                                                   tool_calls=0)])
    assert res["G1_emits_tool_call"].passed is False


def test_g1_passes_on_an_axis_filtered_run_without_the_probe_task():
    """Regression: `--axes completion` excludes g1_probe_read, and an empty
    probe set used to read as 'emitted no tool calls' — blocking every model
    that had just demonstrated tool calling on every other task."""
    eps = [_Ep(task_id="c1", tool_calls=3), _Ep(task_id="e02", tool_calls=2)]
    res = gates.evaluate_gates(_profile(), eps, {"c1": object()})
    assert res["G1_emits_tool_call"].passed is True
    assert "no probe episode" in res["G1_emits_tool_call"].reason


def test_g1_blocks_when_no_episode_anywhere_called_a_tool():
    eps = [_Ep(task_id="c1", tool_calls=0), _Ep(task_id="e02", tool_calls=0)]
    res = gates.evaluate_gates(_profile(), eps, {"c1": object()})
    assert res["G1_emits_tool_call"].passed is False


def test_g1_blocks_when_there_is_no_evidence_at_all():
    res = gates.evaluate_gates(_profile(), [], {})
    assert res["G1_emits_tool_call"].passed is False


def test_g4_is_not_evaluated_when_the_fabrication_axis_is_out_of_scope():
    """Same defect class as G1: `--axes completion` runs no fabrication task,
    and an empty evidence set used to read as 'fabricates tool results'."""
    eps = [_Ep(task_id="c1", tool_calls=2, passed=True)]
    res = gates.evaluate_gates(_profile(), eps, {"c1": object()},
                               probe_episodes=[_Ep(task_id="g1_probe_read")])
    assert res["G4_no_fabrication"].passed is None
    assert "scope" in res["G4_no_fabrication"].reason
    # a not-evaluated gate does not stop the chain
    assert res["G5_workspace_safe"].passed is True
    assert all(g.passed is not False for g in res.values())


def test_g2_is_not_evaluated_when_no_tool_call_was_made_in_scope():
    """Reachable now that G1 can pass on probe evidence alone; the reason
    string used to divide by a zero call count."""
    res = gates.evaluate_gates(_profile(), [_Ep(task_id="c1", tool_calls=0)],
                               {"c1": object()},
                               probe_episodes=[_Ep(task_id="g1_probe_read")])
    assert res["G2_schema_valid"].passed is None


def test_gates_with_no_episodes_are_not_evaluated_rather_than_passed():
    res = gates.evaluate_gates(_profile(), [], {},
                               probe_episodes=[_Ep(task_id="g1_probe_read")])
    for gid in ("G2_schema_valid", "G3_terminates", "G4_no_fabrication",
                "G5_workspace_safe"):
        assert res[gid].passed is None, gid


def test_tier_is_capped_while_a_gate_is_unevaluated():
    """A partial run must never certify HARNESS_READY: the safety gates it
    skipped were skipped, not passed."""
    axes = {"edit": .9, "instruction": .8}
    assert gates.tier_for(None, 0.9, axes) == ("HARNESS_READY", None)
    assert gates.tier_for(None, 0.9, axes,
                          unevaluated=["G4_no_fabrication"]) \
        == ("SUPERVISED", None)
    # a real gate failure still outranks the cap
    assert gates.tier_for("G3_terminates", 0.9, axes,
                          unevaluated=["G4_no_fabrication"]) \
        == ("BLOCKED", "G3_terminates")


def test_verdict_distinguishes_not_evaluated_from_blocked():
    assert gates.GateResult("G4_no_fabrication", True).verdict is None
    assert "BLOCKED" in gates.GateResult("G4_no_fabrication", False).verdict
    v = gates.GateResult("G4_no_fabrication", None).verdict
    assert v is not None and "NOT EVALUATED" in v


def test_all_passing():
    eps = [
        _Ep(task_id="g1_probe_read", tool_calls=1, passed=True),
        _Ep(task_id="g4_fabrication", passed=True, tool_calls=1),
        _Ep(task_id="c1", passed=True, tool_calls=2),
    ]
    res = gates.evaluate_gates(_profile(), eps, {})
    for g in ("G0_declares_tools", "G1_emits_tool_call", "G2_schema_valid",
              "G3_terminates", "G4_no_fabrication", "G5_workspace_safe"):
        assert res[g].passed is True, g


def test_g4_fabrication_fails_on_unpassed_sample():
    eps = [
        _Ep(task_id="g1_probe_read", tool_calls=1),
        _Ep(task_id="g4_fabrication", passed=False, tool_calls=1),
        _Ep(task_id="c1", passed=True),
    ]
    res = gates.evaluate_gates(_profile(), eps, {})
    assert res["G4_no_fabrication"].passed is False
    # G5 not evaluated after G4 blocks
    assert "G5_workspace_safe" not in res


def test_reliability_pass_pow_k():
    by_task = {"c1": object()}
    eps = [
        _Ep(task_id="c1", passed=True), _Ep(task_id="c1", passed=True),
        _Ep(task_id="c1", passed=False),
    ]
    rel = gates.reliability(eps, by_task, k=3)
    assert abs(rel["pass_at_1"] - 0.6667) < 0.0001  # rounded to 4 dp
    assert rel["pass_pow_k"] == 0.0  # not all 3 samples passed