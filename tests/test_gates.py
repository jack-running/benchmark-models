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