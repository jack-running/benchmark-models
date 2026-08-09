#!/usr/bin/env python3
"""
Harness-ready agentic benchmark (Layer A native loop + Layer B real harnesses).

Gate-first, two-layer. Layer A runs a hermetic tool loop against /api/chat
with native tools to isolate model capability. Layer B drives the real
opencode / omp / cline binaries over the same fixtures with the same verifiers.
Scoring is a reliability profile behind binary gates, never a weighted mean.

Stages (each runs up to, and including, itself):
  probe  — /api/show for every model; filters the fleet (cheap, no GPU)
  smoke  — probe_read + one completion task, k=1; drops G1 failures
  native — all tasks x k samples; computes G2-G5 and the axis profile
  e2e    — the 6 E2E tasks x k x selected harnesses (Layer B)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

import requests

import ollama_client
from agent_loop import run_episode
from agent_tasks import (ALL_TASKS, E2E_TASKS, TASKS_BY_ID,
                         HARNESS_SYSTEM_PROMPT, check_harness_rules)
from agent_workspace import ToolRegistry, Workspace
import gates
import harness_drivers

DEFAULT_HOST = harness_drivers.BENCH_HOST
SKIP_MODELS = {"qwen3-embedding:8b", "deepseek-ocr:latest"}
TEMPERATURE = 0.2
STAGES = ["probe", "smoke", "native", "e2e"]


# ─────────────────────────────────────────────────────────────
# small utilities
# ─────────────────────────────────────────────────────────────

def get_ollama_version(host: str) -> str:
    try:
        r = requests.get(f"{host}/api/version", timeout=10)
        return r.json().get("version", "?")
    except Exception:
        return "?"


def list_models(host: str) -> list[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=15)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"Warning: cannot list models on {host}: {e}")
        return []


def suite_sha256() -> str:
    h = hashlib.sha256()
    for t in ALL_TASKS:
        h.update(t.id.encode())
        h.update(t.user_prompt.encode())
        for key in sorted(t.fixture):
            h.update(key.encode())
            h.update(t.fixture[key].encode())
    return h.hexdigest()[:16]


def ep_to_json(ep) -> dict:
    return {
        "task_id": ep.task_id, "seed": ep.seed, "backend": ep.backend,
        "steps": ep.steps, "terminated": ep.terminated,
        "hit_step_budget": ep.hit_step_budget, "hit_wall_budget": ep.hit_wall_budget,
        "passed": ep.passed, "verify_reason": ep.verify_reason,
        "tool_calls": ep.tool_calls, "final_text": ep.final_text[:4000],
        "schema_violations": ep.schema_violations,
        "unknown_tool_calls": ep.unknown_tool_calls, "path_escapes": ep.path_escapes,
        "repeated_call_max": ep.repeated_call_max,
        "wall_seconds": round(ep.wall_seconds, 3),
        "prompt_tokens": ep.prompt_tokens, "completion_tokens": ep.completion_tokens,
        "truncated": ep.truncated, "error": ep.error,
    }


def _warmed_msg(prompt: str, num_ctx: int) -> list[dict]:
    return [{"role": "user", "content": prompt}]


# ─────────────────────────────────────────────────────────────
# Layer A
# ─────────────────────────────────────────────────────────────

def run_native(host, profile_map, task_pool, k, max_steps, wall_budget,
               num_ctx, temperature):
    """Episodes + snapshots per model over task_pool x seeds 1..k."""
    out = {}
    for model in profile_map:
        print(f"\n  Layer A native → {model}")
        ollama_client.warmup(host, model)
        episodes, snapshots = [], []
        for tsk in task_pool:
            for seed in range(1, k + 1):
                ws = Workspace.create(tsk.fixture)
                registry = ToolRegistry(ws)
                sysp = HARNESS_SYSTEM_PROMPT if tsk.axis == "instruction" else ""
                ep = run_episode(
                    host, model, tsk, ws, registry,
                    max_steps=tsk.max_steps or max_steps, wall_budget_s=wall_budget,
                    seed=seed, num_ctx=tsk.num_ctx or num_ctx,
                    system_prompt=sysp, temperature=temperature,
                )
                ok, reason = tsk.verify(ws, ep)
                ep.passed, ep.verify_reason, ep.backend = ok, reason, "native"
                episodes.append(ep)
                snapshots.append({"task_id": tsk.id, "seed": seed,
                                  "state": ws.snapshot()})
                ws.cleanup()
        out[model] = {"episodes": episodes, "snapshots": snapshots}
    return out


def build_model_report(model, profile, episodes, task_by_id, k, num_ctx) -> dict:
    rel = gates.reliability(episodes, task_by_id, k)
    axes = gates.axis_pass_pow_k(episodes, task_by_id)
    walls = sorted(e.wall_seconds for e in episodes if e.wall_seconds > 0)
    p50 = statistics.median(walls) if walls else 0.0
    p90 = walls[int(len(walls) * 0.90)] if walls else p50
    steps = [e.steps for e in episodes]
    toks = [e.prompt_tokens + e.completion_tokens for e in episodes]
    gr = gates.evaluate_gates(profile, episodes, task_by_id)
    failed = next((g.id for g in gr.values() if g.passed is False), None)
    tier, fg = gates.tier_for(failed, rel["pass_pow_k"], axes)
    gates_d = {g.id: {"passed": g.passed, "reason": g.reason,
                      "threshold": g.threshold} for g in gr.values()}
    return {
        "name": model, "tier": tier, "failed_gate": fg or failed,
        "gates": gates_d, "reliability": rel, "axes": axes,
        "p50_seconds": round(p50, 2), "p90_seconds": round(p90, 2),
        "median_steps": statistics.median(steps) if steps else 0,
        "median_tokens": statistics.median(toks) if toks else 0,
        "episodes": [ep_to_json(e) for e in episodes],
        "rules": _rules_report(episodes, task_by_id),
    }


def _rules_report(episodes, task_by_id):
    """R1/R2/R3 compliance aggregated over instruction-axis episodes."""
    agg = {"r1": [0, 0], "r2": [0, 0], "r3": [0, 0]}  # [pass, total]
    for ep in episodes:
        tsk = task_by_id.get(ep.task_id)
        if tsk is None or tsk.axis != "instruction":
            continue
        rules = check_harness_rules(tsk, ep)
        for key in ("r1", "r2", "r3"):
            agg[key][1] += 1
            if rules[key][0]:
                agg[key][0] += 1
    return {key: ({"passed": v[0], "total": v[1],
                   "rate": round(v[0]/v[1], 3) if v[1] else None})
            for key, v in agg.items()}


# ─────────────────────────────────────────────────────────────
# Layer B
# ─────────────────────────────────────────────────────────────

def make_driver(name: str):
    for cls in (harness_drivers.OpenCodeDriver, harness_drivers.OmpDriver,
                harness_drivers.ClineDriver):
        if cls.name == name:
            d = cls()
            return d if d.available()[0] else None
    return None


def run_e2e(host, profile_map, harnesses, k, num_ctx):
    """Layer B per model: {harness: {task_id: [samples]}}."""
    out = {}
    for model, prof in profile_map.items():
        print(f"\n  Layer B e2e → {model}")
        out[model] = {}
        for hname in harnesses:
            driver = make_driver(hname)
            if driver is None:
                print(f"    {hname}: unavailable, skipped")
                out[model][hname] = {"unavailable": True}
                continue
            print(f"    harness: {hname}")
            out[model][hname] = {"unavailable": False, "tasks": {}}
            for tsk in E2E_TASKS:
                samples = []
                for seed in range(1, k + 1):
                    ws = Workspace.create(tsk.fixture)
                    try:
                        driver.prepare(ws, model, prof)
                    except Exception as e:
                        samples.append({"seed": seed, "error": repr(e),
                                        "passed": False, "reason": "prepare failed"})
                        ws.cleanup()
                        continue
                    budget = harness_drivers.HARNESS_TIMEOUTS.get(hname, 120)
                    hr = driver.run(ws, model, tsk.user_prompt, budget)
                    ok, reason = tsk.verify(ws, hr)
                    samples.append({
                        "seed": seed, "passed": ok, "reason": reason,
                        "exit_code": hr.exit_code, "timed_out": hr.timed_out,
                        "wall_seconds": round(hr.wall_seconds, 2),
                        "tool_names": hr.tool_names,
                        "prompt_tokens": hr.prompt_tokens,
                        "completion_tokens": hr.completion_tokens,
                        "stdout_events": len(hr.stdout_events),
                        "raw_stdout": hr.raw_stdout_path,
                    })
                    ws.cleanup()
                out[model][hname]["tasks"][tsk.id] = samples
    return out


def e2e_pass_pow_k(runs_by_task) -> tuple[float, float]:
    per_task_ok = {}
    samples = passed = 0
    for tid, lst in runs_by_task.items():
        ok = [e["passed"] for e in lst if "passed" in e]
        per_task_ok[tid] = all(ok) if ok else False
        samples += len(ok)
        passed += sum(ok)
    n = len(per_task_ok)
    return round(sum(per_task_ok.values()) / n, 4) if n else 0.0, \
        (round(passed / samples, 4) if samples else 0.0)


def native_e2e_subset_ppk(episodes, task_by_id) -> float:
    """native pass_pow_k restricted to the 6 E2E task ids.

    episodes are the serialized JSON dicts (ep_to_json), so read keys, not
    attributes.
    """
    ids = {t.id for t in E2E_TASKS}
    by_task: dict[str, list[bool]] = {}
    for e in episodes:
        tid = e.get("task_id") if isinstance(e, dict) else getattr(e, "task_id", None)
        if tid in ids:
            by_task.setdefault(tid, []).append(e.get("passed") if isinstance(e, dict)
                                               else e.passed)
    if not by_task:
        return 0.0
    per_task = [all(ok) for ok in by_task.values()]
    return round(sum(per_task) / len(per_task), 4)


# ─────────────────────────────────────────────────────────────
# report / json
# ─────────────────────────────────────────────────────────────

def _write_json(path, root):
    Path(path).write_text(json.dumps(root, indent=2), encoding="utf-8")
    print(f"\nJSON → {path}")


def print_summary(reports):
    print("\n" + "=" * 72)
    print("  GATE-FIRST RANKINGS (pass^k)")
    print("=" * 72)

    def _key(kv):
        r = kv[1]
        rel = r["reliability"]
        return (r["tier"] != "BLOCKED", rel["pass_pow_k"], rel["pass_at_1"],
                -r["p90_seconds"])

    order = sorted(reports.items(), key=_key, reverse=True)
    print(f"{'#':<3} {'model':<38} {'tier':<16} {'pass^1':>7} {'pass^k':>7} {'G-fail':>8}")
    for i, (m, r) in enumerate(order, 1):
        rel = r["reliability"]; axe = r["axes"]
        ax = "".join(f"{k}:{v:.2f} " for k, v in axe.items())
        print(f"{i:<3}{m:<38}{r['tier']:<16}{rel['pass_at_1']:>7.2f} "
              f"{rel['pass_pow_k']:>7.2f}  {r['failed_gate'] or '-'}  {ax}")
    print("\nAxes shown as name:pass^k")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="native", choices=STAGES)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--models", nargs="+")
    p.add_argument("--axes", nargs="+")
    p.add_argument("--harness", action="append", choices=["opencode", "omp", "cline"])
    p.add_argument("-k", "--samples", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--wall-budget", type=int, default=300)
    p.add_argument("--num-ctx", type=int, default=32768)
    p.add_argument("--json", dest="out_json")
    p.add_argument("--include-cloud", action="store_true")
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = p.parse_args()

    host = args.host
    ollama_version = get_ollama_version(host)
    print(f"Ollama {ollama_version} @ {host}")

    # ── probe ────────────────────────────────────────────────
    probe_pool: dict[str, ollama_client.ModelProfile] = {}
    dropped = []
    pool = list(args.models) if args.models else list_models(host)
    for m in pool:
        if not args.include_cloud and ":cloud" in m:
            dropped.append((m, "cloud (excluded unless --include-cloud)"))
            continue
        if m in SKIP_MODELS:
            dropped.append((m, "skip-list (embedding/ocr)"))
            continue
        try:
            prof = ollama_client.probe_model(host, m)
        except Exception as e:
            dropped.append((m, f"probe failed: {e.__class__.__name__}"))
            continue
        if not prof.has_tools:
            dropped.append((m, "G0_declares_tools: no tool template"))
            continue
        probe_pool[m] = prof

    print("\nDropped at probe:")
    for m, r in dropped:
        print(f"  • {m} — {r}")
    print(f"\nProbe kept {len(probe_pool)} tool-capable models:")
    for m, prof in probe_pool.items():
        print(f"  • {m} (ctx={prof.context_length}, {prof.parameter_size})")
    if args.stage == "probe":
        return

    # ── smoke (G1) ───────────────────────────────────────────
    smoke_eps = {}
    for m, prof in probe_pool.items():
        tsk = TASKS_BY_ID["g1_probe_read"]
        ws = Workspace.create(tsk.fixture)
        ep = run_episode(host, m, tsk, ws, ToolRegistry(ws),
                         max_steps=args.max_steps, wall_budget_s=args.wall_budget,
                         seed=1, num_ctx=args.num_ctx, temperature=args.temperature)
        ok, reason = tsk.verify(ws, ep)
        ep.passed, ep.verify_reason, ep.backend = ok, reason, "native"
        smoke_eps[m] = ep
        ws.cleanup()

    native_pool = {m: prof for m in probe_pool
                   if len(smoke_eps[m].tool_calls) >= 1}
    print("\nSmoke dropped (G1: no tool call emitted):")
    for m in probe_pool:
        if m not in native_pool:
            print(f"  • {m}")

    if args.stage == "smoke":
        # one completion task for a quick signal beyond the probe
        tsk = TASKS_BY_ID["c1_add_function"]
        for m in list(native_pool):
            ws = Workspace.create(tsk.fixture)
            ep = run_episode(host, m, tsk, ws, ToolRegistry(ws),
                             max_steps=args.max_steps, wall_budget_s=args.wall_budget,
                             seed=1, num_ctx=args.num_ctx,
                             temperature=args.temperature)
            ok, reason = tsk.verify(ws, ep)
            ep.passed, ep.verify_reason, ep.backend = ok, reason, "native"
            print(f"  smoke {tsk.id}: {m} → {'PASS' if ok else 'FAIL'} ({reason[:80]})")
            ws.cleanup()
            if not ok:
                native_pool.pop(m, None)
        print("\nsmoke stage complete")
        if args.out_json:
            root = {"config": {"stage": "smoke", "ollama_version": ollama_version,
                               "host": host, "task_suite_sha256": suite_sha256()},
                    "models": {m: {"smoke_probe": ep_to_json(smoke_eps[m])}
                               for m in smoke_eps}}
            _write_json(args.out_json, root)
        return

    # ── native ───────────────────────────────────────────────
    tasks = [t for t in ALL_TASKS if not args.axes or t.axis in args.axes]
    native_out = run_native(host, native_pool, tasks, args.samples,
                            args.max_steps, args.wall_budget, args.num_ctx,
                            args.temperature)

    reports = {}
    for m, data in native_out.items():
        reports[m] = build_model_report(m, probe_pool[m], data["episodes"],
                                        TASKS_BY_ID, args.samples, args.num_ctx)

    print_summary(reports)

    root = {
        "config": {
            "ollama_version": ollama_version, "host": host,
            "temperature": args.temperature, "seeds": list(range(1, args.samples + 1)),
            "k": args.samples, "num_ctx": args.num_ctx,
            "task_suite_sha256": suite_sha256(),
            "harness_versions": harness_drivers.harness_versions(),
            "stage": "native",
        },
        "models": reports,
    }

    if args.stage == "e2e":
        # additional layer B runs on e2e-capable models
        harnesses = args.harness or None
        available = [n for n in ("opencode", "omp", "cline")
                     if harnesses is None or n in harnesses]
        # Layer B runs on every model that reached the native stage. Gate
        # verdicts describe native-loop trust; harness transfer is measured
        # separately on the 6 E2E tasks even when G3 blocks a model.
        e2e_pool = {m: reports[m] for m in native_pool}
        e2e_out = run_e2e(host, {m: prof for m, prof in native_pool.items()
                                 if m in e2e_pool}, available, args.samples,
                          args.num_ctx)
        root["layer_b"] = {}
        for m, hmap in e2e_out.items():
            root["layer_b"][m] = {}
            for hname, d in hmap.items():
                if d.get("unavailable"):
                    root["layer_b"][m][hname] = {"unavailable": True}
                    continue
                ppk, p1 = e2e_pass_pow_k(d["tasks"])
                native_ppk = native_e2e_subset_ppk(
                    reports[m]["episodes"], TASKS_BY_ID)
                root["layer_b"][m][hname] = {
                    "e2e_pass_pow_k": ppk, "pass_at_1": p1,
                    "transfer_delta": round(ppk - native_ppk, 4),
                    "native_subset_ppk": native_ppk,
                    "tasks": d["tasks"],
                }
        root["config"]["stage"] = "e2e"

    if args.out_json:
        _write_json(args.out_json, root)


if __name__ == "__main__":
    main()