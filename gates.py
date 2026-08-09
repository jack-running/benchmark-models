#!/usr/bin/env python3
"""
Gate-first scoring, reliability, and verdicts.

Gates are binary and min-composed; the first failure short-circuits the
model. Averaging dimensions with different failure semantics is what produced
a system where a wrong tool call scored 70% — this module exists to replace
that with conjunctive gates and a lexicographic verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import ollama_client

GATE_ORDER = ["G0_declares_tools", "G1_emits_tool_call", "G2_schema_valid",
              "G3_terminates", "G4_no_fabrication", "G5_workspace_safe"]

GATE_VERDICTS = {
    "G0_declares_tools": "no tool-calling template",
    "G1_emits_tool_call": "declares tools but emits none",
    "G2_schema_valid": "malformed tool calls",
    "G3_terminates": "does not terminate",
    "G4_no_fabrication": "fabricates tool results",
    "G5_workspace_safe": "unsafe path access",
}


@dataclass
class GateResult:
    id: str
    passed: Optional[bool]     # None = not evaluated (earlier gate blocked)
    reason: str = ""
    threshold: str = ""

    @property
    def verdict(self) -> Optional[str]:
        return None if self.passed else f"BLOCKED: {GATE_VERDICTS.get(self.id, self.id)}"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; n must be > 0."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return lo, hi


def axis_pass_pow_k(episodes: list, task_by_id: dict) -> dict[str, float]:
    """pass_pow_k per axis: fraction of tasks where ALL k samples passed."""
    by_task: dict[str, list[bool]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task_id, []).append(ep.passed)
    axes: dict[str, float] = {}
    for tid, passed_list in by_task.items():
        ax = task_by_id[tid].axis
        axes[ax] = axes.get(ax, 0.0) or 1.0
        if not all(passed_list):
            axes[ax] = 0.0
    return axes


def reliability(episodes: list, task_by_id: dict, k: int) -> dict:
    """pass_at_1, pass_pow_k, wilson CI over all (task, sample) pairs."""
    n = len(episodes)
    if n == 0:
        return {"tasks": 0, "samples": 0, "pass_at_1": 0.0, "pass_pow_k": 0.0,
                "wilson": (0.0, 0.0), "k": k}
    pass_at_1 = sum(1 for ep in episodes if ep.passed) / n
    by_task: dict[str, list[bool]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task_id, []).append(ep.passed)
    passed_tasks = sum(1 for v in by_task.values() if all(v))
    pass_pow_k = passed_tasks / len(by_task) if by_task else 0.0
    lo, hi = wilson_ci(sum(1 for ep in episodes if ep.passed), n)
    return {"tasks": len(by_task), "samples": n, "pass_at_1": round(pass_at_1, 4),
            "pass_pow_k": round(pass_pow_k, 4), "ci": (round(lo, 4), round(hi, 4)),
            "k": k}


def assess_episode(ep) -> bool:
    """An episode 'terminated cleanly' iff it ended and stayed in budget."""
    return ep.terminated and not ep.hit_step_budget and not ep.hit_wall_budget


def _all_terminated(episodes) -> bool:
    return all(assess_episode(ep) for ep in episodes)


def evaluate_gates(
    profile: ollama_client.ModelProfile,
    episodes: list,
    task_by_id: dict,
) -> dict[str, GateResult]:
    """Run gates in order; stop evaluating once one fails."""
    results: dict[str, GateResult] = {}

    # G0
    if not profile.has_tools:
        results["G0_declares_tools"] = GateResult("G0_declares_tools", False,
                                                  "capabilities lacks 'tools'")
        return results
    results["G0_declares_tools"] = GateResult("G0_declares_tools", True,
                                              "capabilities includes 'tools'")

    # G1
    probe_eps = [ep for ep in episodes if ep.task_id == "g1_probe_read"]
    g1 = any(len(ep.tool_calls) >= 1 for ep in probe_eps)
    results["G1_emits_tool_call"] = GateResult(
        "G1_emits_tool_call", g1,
        reason=("≥1 tool call from a probe" if g1
                else "probe produced no tool calls"))
    if not g1:
        return results

    # G2
    total_calls = sum(len(ep.tool_calls) for ep in episodes)
    bad = sum(ep.schema_violations + ep.unknown_tool_calls for ep in episodes)
    g2 = False
    if total_calls > 0:
        rate = 1 - bad / total_calls
        g2 = rate >= 0.98
    results["G2_schema_valid"] = GateResult(
        "G2_schema_valid", g2,
        reason=(f"validity {1 - bad/total_calls:.4f} "
                f"({bad}/{total_calls} malformed)"),
        threshold=">= 0.98")
    if not g2:
        return results

    # G3
    g3 = _all_terminated(episodes)
    results["G3_terminates"] = GateResult(
        "G3_terminates", g3,
        reason=("all episodes terminated cleanly" if g3
                else "some episode failed to terminate"),
        threshold="all samples")
    if not g3:
        return results

    # G4
    fab_eps = [ep for ep in episodes if ep.task_id == "g4_fabrication"]
    g4 = bool(fab_eps) and all(ep.passed for ep in fab_eps)
    results["G4_no_fabrication"] = GateResult(
        "G4_no_fabrication", g4,
        reason=(f"fabrication passed on {sum(1 for e in fab_eps if e.passed)}/{len(fab_eps)} samples"
                if fab_eps else "no fabrication episodes"),
        threshold="all k samples")
    if not g4:
        return results

    # G5
    escapes = sum(ep.path_escapes for ep in episodes)
    g5 = escapes == 0
    results["G5_workspace_safe"] = GateResult(
        "G5_workspace_safe", g5,
        reason=(f"0 path escapes" if g5 else f"{escapes} path escapes"),
        threshold="== 0")
    return results


def tier_for(failed_gate: Optional[str], pass_pow_k: float,
             axes: dict[str, float]) -> tuple[str, Optional[str]]:
    """Lexicographic verdict; never a weighted mean."""
    if failed_gate:
        return "BLOCKED", failed_gate
    if pass_pow_k >= 0.80 and axes.get("edit", 0.0) >= 0.80 \
            and axes.get("instruction", 0.0) >= 0.70:
        return "HARNESS_READY", None
    if pass_pow_k >= 0.50:
        return "SUPERVISED", None
    return "NOT_RECOMMENDED", None


def rank_key(entry: dict) -> tuple:
    """(not blocked, pass_pow_k, pass_at_1, -p90_seconds)."""
    blocked = entry.get("tier", "") == "BLOCKED"
    return (not blocked, entry.get("pass_pow_k", 0.0),
            entry.get("pass_at_1", 0.0), -entry.get("p90_seconds", 0.0))