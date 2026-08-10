"""Offline tests for report_html (renderer + dispatch, no server)."""
import re

import pytest

import report_html

CONFIG = {
    "stage": "native", "host": "http://h", "ollama_version": "0.5.7",
    "k": 3, "seeds": [1, 2, 3], "num_ctx": 4096, "temperature": 0.2,
    "task_suite_sha256": "abc123", "harness_versions": {"opencode": "1.0",
                                                        "omp": "not-installed"},
}

G0_G3 = {
    "G0_declares_tools": {"passed": True, "reason": "tool template found",
                          "threshold": "tools declared"},
    "G1_emits_tool_call": {"passed": True, "reason": "used read_file",
                           "threshold": ">= 1 call"},
    "G2_schema_valid": {"passed": True, "reason": "all valid",
                        "threshold": "100%"},
    "G3_terminates": {"passed": False, "reason": "no finish emitted",
                      "threshold": "ends with finish"},
}


def _ep(tid, seed, passed, reason="", **extra):
    ep = {
        "task_id": tid, "seed": seed, "backend": "native", "steps": 3,
        "terminated": False, "hit_step_budget": False, "hit_wall_budget": False,
        "passed": passed, "verify_reason": reason, "tool_calls": [],
        "final_text": "done", "schema_violations": 0,
        "unknown_tool_calls": 0, "path_escapes": 0, "repeated_call_max": 0,
        "wall_seconds": 1.0, "prompt_tokens": 10, "completion_tokens": 5,
        "truncated": False, "error": None,
    }
    ep.update(extra)
    return ep


def _model_report(tier="HARNESS_READY", gates=None, axes=None, episodes=None,
                  failed_gate=None):
    axes = axes or {"completion": 1.0}
    rel = {"tasks": 2, "samples": 6, "pass_at_1": 1.0, "pass_pow_k": 1.0,
           "ci": [0.9, 1.0], "k": 3}
    return {
        "name": "m", "tier": tier, "failed_gate": failed_gate,
        "gates": gates or {}, "reliability": rel, "axes": axes,
        "p50_seconds": 5.0, "p90_seconds": 20.0, "median_steps": 4,
        "median_tokens": 1000, "episodes": episodes or [],
        "rules": {"r1": {"passed": 0, "total": 0, "rate": None},
                  "r2": {"passed": 0, "total": 0, "rate": None},
                  "r3": {"passed": 0, "total": 0, "rate": None}},
    }


def _render(report, config=None, extra=None):
    root = {"config": config or CONFIG, "models": {"m": report}}
    if extra:
        root.update(extra)
    return report_html.render_agent_report(root, "fixture.json")


# ── detect_kind ──────────────────────────────────────────────

def test_detect_kind_shapes():
    assert report_html.detect_kind({"config": {}, "models": {}}) == "agent"
    assert report_html.detect_kind({"m": {"tests": []}}) == "quality"
    assert report_html.detect_kind({"m": {"overall": {}}}) == "speed"


def test_detect_kind_empty_raises():
    with pytest.raises(ValueError, match="unrecognised"):
        report_html.detect_kind({})


# ── escaping ─────────────────────────────────────────────────

def test_episode_output_escaped():
    ep = _ep("c3_rename_symbol", 1, False, "bad",
             final_text="<script>alert(1)</script>",
             tool_calls=[{"name": "edit_file",
                          "arguments": {"content": "<b>"},
                          "ok": True, "violation": None, "text": "ok"}])
    h = _render(_model_report(tier="BLOCKED", gates=G0_G3,
                              failed_gate="G3_terminates", episodes=[ep]))
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in h
    assert "&lt;b&gt;" in h
    assert "<script>" not in h          # never raw model output


# ── gate chain ───────────────────────────────────────────────

def test_absent_gates_render_not_evaluated():
    h = _render(_model_report(tier="BLOCKED", gates=G0_G3,
                              failed_gate="G3_terminates"))
    # Scope to the gate table: the "how to read" prose also explains the
    # not-evaluated state, so a document-wide count is not the intent.
    m = re.search(r"<h3>Gate chain</h3>\s*<table>(.*?)</table>", h, re.S)
    assert m, "gate chain table not found"
    assert m.group(1).count("not evaluated") == 2  # G4 and G5 only
    assert "G4_no_fabrication" in h and "G5_workspace_safe" in h
    assert "chain stopped at G3_terminates" in h
    assert "BLOCKED" in h


# ── task matrix ──────────────────────────────────────────────

def test_matrix_count_and_lowest_seed_reason():
    eps = [
        _ep("c1_add_function", 1, True), _ep("c1_add_function", 2, True),
        _ep("c1_add_function", 3, True),
        _ep("a4_grounding", 1, False, "seed1 broke"),
        _ep("a4_grounding", 2, False, "seed2 broke"),
        _ep("a4_grounding", 3, True),
    ]
    r = _model_report(axes={"completion": 1.0, "grounding": 1.0}, episodes=eps)
    h = _render(r)
    m = re.search(r"<h3>Task results</h3>\s*<table>(.*?)</table>", h, re.S)
    assert m, "task matrix table not found"
    matrix = m.group(1)
    assert matrix.count("<tr") == 3         # header + 2 task rows
    assert "3/3" in matrix and "1/3" in matrix
    # matrix shows the LOWEST-seed failing reason; the other one lives in details
    assert "seed1 broke" in matrix
    assert "seed2 broke" not in matrix


# ── smoke stage fallback ─────────────────────────────────────

def test_smoke_shape_renders_single_table():
    root = {"config": {"stage": "smoke"},
            "models": {"m": {"smoke_probe": _ep("g1_probe_read", 1, True,
                                                "token found")}}}
    h = report_html.render_agent_report(root, "smoke.json")
    assert "Model Smoke Results" in h
    assert "token found" in h
    assert "Rankings" not in h          # sections 3-4 skipped


# ── layer B ──────────────────────────────────────────────────

def test_layer_b_signed_delta_and_unavailable():
    lb = {
        "opencode": {"e2e_pass_pow_k": 1.0, "pass_at_1": 1.0,
                     "native_subset_ppk": 1.0, "transfer_delta": 0.0,
                     "tasks": {}},
        "omp": {"e2e_pass_pow_k": 0.5, "pass_at_1": 0.5,
                "native_subset_ppk": 1.0, "transfer_delta": -0.5,
                "tasks": {"e02": [{"seed": 1, "passed": False,
                                   "reason": "file differs", "exit_code": 1,
                                   "timed_out": False, "wall_seconds": 42.0,
                                   "tool_names": ["read_file", "edit_file"],
                                   "prompt_tokens": 1, "completion_tokens": 1,
                                   "stdout_events": 2, "raw_stdout": "p"}]}},
        "cline": {"unavailable": True},
    }
    h = _render(_model_report(), extra={"layer_b": {"m": lb}})
    assert "+0.00" in h and "-0.50" in h
    assert "file differs" in h and "read_file, edit_file" in h
    assert "unavailable" in h

def test_how_to_read_and_provenance_render():
    """Regression: TIPS was defined but never emitted, so the report shipped
    without the 'How to read this report' section the spec requires."""
    h = _render(_model_report())
    assert "How to read this report" in h
    assert "pass^k" in h and "transfer_delta" in h
    assert "Empty answers fail" in h
    # provenance card present when the run records a measurement version
    cfg = dict(CONFIG, measurement_version=2, e2e_budget_s=300,
               omp_thinking="off", harness_runtime_num_ctx="model-default",
               system_prompt_scope="instruction_axis_only",
               verifier_policy="require_final_answer")
    h2 = _render(_model_report(), config=cfg)
    assert "Measurement definition" in h2
    assert "300 s (same for every harness)" in h2


def test_pre_v2_runs_get_comparability_warning():
    """Old artifacts must say so rather than look like v2 measurements."""
    h = _render(_model_report())          # CONFIG has no measurement_version
    assert "Measurement provenance not recorded" in h
    assert "not comparable" in h
