#!/usr/bin/env python3
"""
Render benchmark result JSON files as self-contained, dark-themed HTML reports.

Three benchmark families are supported, dispatched on the top-level JSON shape:

  agent   — benchmark_agent.py output: per-model gates, axes, episode matrix,
            failure traces, harness rules and the optional Layer B (e2e) block.
  quality — benchmark_quality.py output: delegated to
            benchmark_quality.save_html_report (no reimplementation).
  speed   — benchmark_ollama.py output: delegated to
            benchmark_ollama.save_html_report.

Reports are a single self-contained file: inline <style>, no JavaScript, no
external assets. Collapsible regions use native <details>/<summary>.

Usage:
  python report_html.py results/v3_k3.json results/v4_e2e.json \
      results/qA/quality_ctx4096_nothink_*.json
  python report_html.py -o out/report.html results/v3_k3.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from agent_tasks import TASKS_BY_ID

# GitHub-dark palette shared with benchmark_ollama.py / benchmark_quality.py,
# plus the classes the agent report needs. The table/th/td/tr, p.meta and .tip
# rules are copied verbatim from benchmark_ollama.py:526-542.
BASE_CSS = """\
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #e6edf3; padding: 32px 24px; line-height: 1.5; }
  h1   { color: #58a6ff; font-size: 1.6em; margin-bottom: 4px; }
  h2   { color: #79c0ff; font-size: 1.1em; margin: 32px 0 12px;
         border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  p.meta { color: #8b949e; font-size: 0.85em; margin-bottom: 24px; }
  table  { width: 100%; border-collapse: collapse; font-size: 0.88em; }
  th { background: #161b22; color: #79c0ff; padding: 10px 12px;
       text-align: left; border: 1px solid #30363d; white-space: nowrap; }
  td { padding: 9px 12px; border: 1px solid #30363d; vertical-align: middle; }
  tr:nth-child(even) { background: #0d1117; }
  tr:nth-child(odd)  { background: #111318; }
  tr:hover           { background: #1c2128; }
  .tip { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px 20px; margin: 20px 0; font-size: 0.9em; color: #8b949e; }
  .tip strong { color: #e6edf3; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-weight: bold; font-size: 0.8em; color: #0d1117; }
  .ok  { color: #3fb950; }
  .bad { color: #f85149; }
  .muted { color: #8b949e; }
  .bar { background: #21262d; height: 10px; border-radius: 5px; width: 160px; }
  .bar > i { display: block; height: 100%; border-radius: 5px; background: #3fb950; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px 20px; margin: 12px 0; }
  pre { background: #161b22; padding: 8px; border-radius: 4px;
        overflow-x: auto; font-size: 0.82em; white-space: pre-wrap; }
  code { background: #161b22; padding: 1px 5px; border-radius: 3px; font-size: 0.88em; }
  details summary { cursor: pointer; color: #58a6ff; user-select: none; }"""

TIER_COLORS = {
    "HARNESS_READY": "#3fb950",
    "SUPERVISED": "#d29922",
    "NOT_RECOMMENDED": "#f85149",
    "BLOCKED": "#f85149",
}

# The canonical gate chain: evaluated in order, first failure stops the chain.
GATE_ORDER = [
    "G0_declares_tools",
    "G1_emits_tool_call",
    "G2_schema_valid",
    "G3_terminates",
    "G4_no_fabrication",
    "G5_workspace_safe",
]

RULES = [
    ("r1", "R1 never write_file on an existing file"),
    ("r2", "R2 end with finish"),
    ("r3", "R3 never read the same path twice"),
]


# ─────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────

def e(v) -> str:
    """Escape a dynamic value for HTML; None renders as empty string."""
    if v is None:
        return ""
    return html.escape(str(v))


def _trunc(s, n: int) -> str:
    s = str(s)
    return s[:n] + "…" if len(s) > n else s


def _pct_bar(v: float) -> str:
    return f'<div class="bar"><i style="width:{v * 100:.0f}%"></i></div>'


def _mark(ok: bool) -> str:
    return '<span class="ok">✓</span>' if ok else '<span class="bad">✗</span>'


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", model)


def _tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#8b949e")
    return f'<span class="badge" style="background:{color}">{e(tier)}</span>'


def _fact_rows(pairs) -> str:
    """Two-column facts table (th = label, td = escaped value)."""
    rows = "".join(
        f"<tr><th style='white-space:nowrap'>{e(k)}</th><td>{e(v)}</td></tr>\n"
        for k, v in pairs
    )
    return f"<table>{rows}</table>"


# ─────────────────────────────────────────────────────────────
# agent renderer
# ─────────────────────────────────────────────────────────────

def _rank_key(item):
    """Same ranking key as benchmark_agent.print_summary."""
    _m, r = item
    rel = r["reliability"]
    return (r["tier"] != "BLOCKED", rel["pass_pow_k"], rel["pass_at_1"],
            -r["p90_seconds"])


def _tax_axis(tid: str) -> tuple[str, str]:
    """(axis, name) for a task id; unknown ids degrade gracefully."""
    t = TASKS_BY_ID.get(tid)
    if t is None:
        return "unknown", tid
    return t.axis, t.name


def _tax_pos(tid: str, axes_order: list) -> tuple:
    ax, _n = _tax_axis(tid)
    if ax in axes_order:
        return (axes_order.index(ax), tid)
    return (len(axes_order), tid)  # unknown axes last


def _gate_chain_table(gates: dict, failed_gate) -> str:
    rows = ""
    for gid in GATE_ORDER:
        g = gates.get(gid)
        if g is None:
            rows += (
                f"<tr><td>{e(gid)}</td>"
                f"<td><span class='muted'>– not evaluated</span></td>"
                f"<td>chain stopped at {e(failed_gate)}</td><td>—</td></tr>\n"
            )
        elif g.get("passed") is None:
            # Evaluated-but-no-evidence: the run's scope contained no task for
            # this gate. Never render it as a failure.
            rows += (
                f"<tr><td>{e(gid)}</td>"
                f"<td><span class='muted'>– not evaluated</span></td>"
                f"<td>{e(g.get('reason'))}</td>"
                f"<td>{e(g.get('threshold'))}</td></tr>\n"
            )
        else:
            rows += (
                f"<tr><td>{e(gid)}</td>"
                f"<td>{_mark(bool(g.get('passed')))}</td>"
                f"<td>{e(g.get('reason'))}</td>"
                f"<td>{e(g.get('threshold'))}</td></tr>\n"
            )
    head = "<tr><th>Gate</th><th>Result</th><th>Reason</th><th>Threshold</th></tr>"
    return f"<table>{head}{rows}</table>"


def _axes_table(axes: dict) -> str:
    rows = ""
    for name, v in axes.items():
        rows += (f"<tr><td>{e(name)}</td><td>{_pct_bar(float(v))}</td>"
                 f"<td>{v:.2f}</td></tr>\n")
    head = "<tr><th>Axis</th><th>Pass^k</th><th>Rate</th></tr>"
    return f"<table>{head}{rows}</table>"


def _task_matrix(episodes: list, axes_order: list) -> str:
    by_task: dict[str, list] = {}
    for ep in episodes:
        by_task.setdefault(ep["task_id"], []).append(ep)
    order = sorted(by_task, key=lambda tid: _tax_pos(tid, axes_order))
    seeds = sorted({ep["seed"] for ep in episodes})

    head = "<tr><th>Axis</th><th>Task</th>"
    head += "".join(f"<th>seed {s}</th>" for s in seeds)
    head += "<th>Passed</th><th>Why it failed</th></tr>"

    rows = ""
    for tid in order:
        eps = sorted(by_task[tid], key=lambda x: x["seed"])
        ax, name = _tax_axis(tid)
        by_seed = {ep["seed"]: ep for ep in eps}
        seed_cells = "".join(
            _mark(by_seed[s]["passed"]) if s in by_seed
            else "<span class='muted'>–</span>"
            for s in seeds
        )
        n = sum(1 for ep in eps if ep["passed"])
        failing = [ep for ep in eps if not ep["passed"]]
        if failing:
            reason = e(_trunc(failing[0]["verify_reason"], 160))
            tint = " style='background:#161b22'"
        else:
            reason = ""
            tint = ""
        rows += (f"<tr{tint}><td>{e(ax)}</td><td>{e(name)}</td>"
                 f"{seed_cells}<td>{n}/{len(eps)}</td>"
                 f"<td>{reason}</td></tr>\n")
    return f"<table>{head}{rows}</table>"


def _failure_details(failing: list, axes_order: list) -> str:
    failing = sorted(failing, key=lambda ep: (_tax_pos(ep["task_id"], axes_order),
                                              ep["seed"]))
    out = ""
    for ep in failing:
        summary = e(_trunc(ep["verify_reason"], 80))
        facts = _fact_rows([
            ("steps", ep.get("steps")),
            ("terminated", ep.get("terminated")),
            ("hit_step_budget", ep.get("hit_step_budget")),
            ("hit_wall_budget", ep.get("hit_wall_budget")),
            ("schema_violations", ep.get("schema_violations")),
            ("unknown_tool_calls", ep.get("unknown_tool_calls")),
            ("path_escapes", ep.get("path_escapes")),
            ("repeated_call_max", ep.get("repeated_call_max")),
            ("truncated", ep.get("truncated")),
            ("error", ep.get("error")),
            ("wall_seconds", ep.get("wall_seconds")),
            ("prompt_tokens", ep.get("prompt_tokens")),
            ("completion_tokens", ep.get("completion_tokens")),
        ])

        trace_rows = ""
        for i, tc in enumerate(ep.get("tool_calls") or [], 1):
            args = json.dumps(tc.get("arguments"), ensure_ascii=False)
            viol = e(tc.get("violation")) or "—"
            trace_rows += (
                f"<tr><td>{i}</td><td>{e(tc.get('name'))}</td>"
                f"<td><code>{e(_trunc(args, 200))}</code></td>"
                f"<td>{_mark(bool(tc.get('ok')))}</td><td>{viol}</td>"
                f"<td><pre>{e(_trunc(tc.get('text'), 300))}</pre></td></tr>\n"
            )
        trace_head = "<tr><th>#</th><th>Tool</th><th>Arguments</th>" \
                     "<th>OK</th><th>Violation</th><th>Output</th></tr>"
        final_text = e(_trunc(ep.get("final_text"), 2000))

        out += (
            f"<details><summary>{e(ep['task_id'])} · seed {ep['seed']} · "
            f"{summary}</summary>"
            f"{facts}"
            f"<table>{trace_head}{trace_rows}</table>"
            f"<pre>{final_text}</pre>"
            f"</details>\n"
        )
    return out


def _all_episodes_table(episodes: list) -> str:
    rows = ""
    for ep in sorted(episodes, key=lambda x: (x["task_id"], x["seed"])):
        rows += (
            f"<tr><td>{e(ep['task_id'])}</td><td>{ep['seed']}</td>"
            f"<td>{_mark(bool(ep['passed']))}</td><td>{e(ep.get('steps'))}</td>"
            f"<td>{e(ep.get('wall_seconds'))}</td>"
            f"<td>{e(ep.get('prompt_tokens'))}+{e(ep.get('completion_tokens'))}</td></tr>\n"
        )
    head = "<tr><th>Task</th><th>Seed</th><th>Pass</th><th>Steps</th>" \
           "<th>Wall s</th><th>Tokens (p+c)</th></tr>"
    return f"<table>{head}{rows}</table>"


def _rules_table(rules: dict) -> str:
    rows = ""
    for key, label in RULES:
        r = rules.get(key)
        if r is None:
            continue
        rate = r.get("rate")
        rate_s = f"{rate * 100:.0f}%" if rate is not None else "—"
        rows += (f"<tr><td>{e(label)}</td>"
                 f"<td>{e(r.get('passed'))}/{e(r.get('total'))}</td>"
                 f"<td>{rate_s}</td></tr>\n")
    head = "<tr><th>Rule</th><th>Passed / total</th><th>Rate</th></tr>"
    return f"<table>{head}{rows}</table>"


def _thinking_cell(smp: dict) -> str:
    """Model reasoning actually emitted, regardless of the requested level.

    A run can show "thinking off" and still spend its whole budget reasoning
    (observed: a trivial one-file task took 598 s on a 27b thinking model), so
    flag it rather than trusting the requested level.
    """
    n = smp.get("thinking_parts")
    if n is None:
        return "—"
    return f'<span class="bad">{e(n)}</span>' if n else e(n)


def _layer_b_block(lb_entry: dict) -> str:
    """Layer B summary table + per-harness task tables."""
    head = "<tr><th>Harness</th><th>E2E pass^k</th><th>Pass@1</th>" \
           "<th>Native subset</th><th>Delta</th></tr>"
    rows = ""
    for hname, d in lb_entry.items():
        if isinstance(d, dict) and d.get("unavailable"):
            rows += (f"<tr><td>{e(hname)}</td>"
                     f"<td><span class='muted'>unavailable</span></td>"
                     f"<td>—</td><td>—</td><td>—</td></tr>\n")
            continue
        ppk = d.get("e2e_pass_pow_k")
        p1 = d.get("pass_at_1")
        native = d.get("native_subset_ppk")
        delta = d.get("transfer_delta")
        ppk_s = f"{ppk:.2f}" if ppk is not None else "—"
        p1_s = f"{p1:.2f}" if p1 is not None else "—"
        native_s = f"{native:.2f}" if native is not None else "—"
        if delta is not None:
            cls = "bad" if delta < 0 else "ok"
            delta_s = f"<span class='{cls}'>{delta:+.2f}</span>"
        else:
            delta_s = "—"
        rows += (f"<tr><td>{e(hname)}</td><td>{ppk_s}</td><td>{p1_s}</td>"
                 f"<td>{native_s}</td><td>{delta_s}</td></tr>\n")
    out = f"<table>{head}{rows}</table>"

    for hname, d in lb_entry.items():
        if not isinstance(d, dict) or d.get("unavailable"):
            continue
        tasks = d.get("tasks") or {}
        t_head = ("<tr><th>Task</th><th>Seed</th><th>Pass</th><th>Reason</th>"
                  "<th>Exit</th><th>Timed out</th><th>Wall s</th>"
                  "<th>Budget s</th><th>Compactions</th><th>Thinking</th>"
                  "<th>Reasoning parts</th>"
                  "<th>Ctx budget</th><th>Loaded ctx</th><th>Answer chars</th>"
                  "<th>Tools</th></tr>")
        t_rows = ""
        for tid in sorted(tasks):
            for smp in sorted(tasks[tid], key=lambda x: x.get("seed", 0)):
                tools = ", ".join(e(t) for t in (smp.get("tool_names") or []))
                # A run killed at its budget, or one that compacted, did not
                # get a fair shot — flag both so a delta is never read as
                # pure harness quality.
                to = smp.get("timed_out")
                to_s = (f"<span class='bad'>{e(to)}</span>" if to
                        else f"<span class='muted'>{e(to)}</span>")
                comp = smp.get("compactions")
                comp_s = (f"<span class='bad'>{e(comp)}</span>" if comp
                          else e(comp if comp is not None else "—"))
                t_rows += (
                    f"<tr><td>{e(tid)}</td><td>{e(smp.get('seed'))}</td>"
                    f"<td>{_mark(bool(smp.get('passed')))}</td>"
                    f"<td>{e(smp.get('reason'))}</td>"
                    f"<td>{e(smp.get('exit_code'))}</td>"
                    f"<td>{to_s}</td>"
                    f"<td>{e(smp.get('wall_seconds'))}</td>"
                    f"<td>{e(smp.get('budget_s') if smp.get('budget_s') is not None else '—')}</td>"
                    f"<td>{comp_s}</td>"
                    f"<td>{e(smp.get('thinking_level') or '—')}</td>"
                    f"<td>{_thinking_cell(smp)}</td>"
                    f"<td>{e(smp.get('context_budget') if smp.get('context_budget') is not None else '—')}</td>"
                    f"<td>{e(smp.get('server_context') if smp.get('server_context') is not None else '—')}</td>"
                    f"<td>{e(smp.get('final_text_chars') if smp.get('final_text_chars') is not None else '—')}</td>"
                    f"<td>{tools}</td></tr>\n"
                )
        out += (f"<h3>{e(hname)}</h3>"
                f"<table>{t_head}{t_rows}</table>")
    return out


def _meta_line(cfg: dict) -> str:
    def g(key):
        return cfg.get(key)

    hv = cfg.get("harness_versions")
    hv_s = " · ".join(f"{e(n)} {e(v)}" for n, v in hv.items()) if hv else "n/a"
    seeds = g("seeds")
    seeds_s = ", ".join(str(s) for s in seeds) if seeds else "n/a"

    def v(key):
        return g(key) if g(key) is not None else "n/a"

    return (
        f"stage <strong>{e(v('stage'))}</strong> · host {e(v('host'))} · "
        f"ollama {e(v('ollama_version'))} · k={e(v('k'))} · "
        f"seeds [{seeds_s}] · num_ctx {e(v('num_ctx'))} · "
        f"temperature {e(v('temperature'))} · "
        f"suite {e(v('task_suite_sha256'))} · harnesses {hv_s}"
    )


def _provenance_card(cfg: dict) -> str:
    """Measurement definition this report was produced under.

    Rendered only for runs that record it (measurement_version >= 2). Older
    artifacts predate these fields and are NOT comparable on the fabrication /
    grounding axes or on transfer_delta.
    """
    if not cfg.get("measurement_version"):
        return (
            '<div class="card"><span class="muted">Measurement provenance not '
            'recorded (pre-v2 run). Layer B budgets were asymmetric '
            '(opencode 120 s vs omp 90 s + a self-cap) and empty answers could '
            'pass the fabrication axis, so transfer_delta and the fabrication '
            'axis are not comparable with v2 runs.</span></div>'
        )
    rows = [
        ("measurement version", cfg.get("measurement_version")),
        ("Layer B budget", f"{cfg.get('e2e_budget_s')} s (same for every harness)"),
        ("omp thinking", cfg.get("omp_thinking")),
        ("omp context budget", cfg.get("omp_context") if cfg.get("omp_context")
         else "harness default (not pinned)"),
        ("harness runtime num_ctx", cfg.get("harness_runtime_num_ctx")),
        ("system prompt scope", cfg.get("system_prompt_scope")),
        ("verifier policy", cfg.get("verifier_policy")),
    ]
    cells = " &nbsp;·&nbsp; ".join(
        f"{e(k)} <strong>{e(val)}</strong>" for k, val in rows
        if val is not None)
    return (f'<div class="card"><strong>Measurement definition</strong><br>'
            f'<span class="muted">{cells}</span></div>')


TIPS = """\
<div class="tip">
  <strong>How to read this report</strong><br><br>
  <strong>Usability:</strong> a model is usable in a harness only when its tier is
  <strong>HARNESS_READY</strong> or <strong>SUPERVISED</strong>. A
  <strong>BLOCKED</strong> / <strong>NOT_RECOMMENDED</strong> tier means a gate
  failed, and the pass-rate numbers below are diagnostic only.<br><br>
  <strong>Gates:</strong> the gate chain is conjunctive and evaluated
  <strong>in order</strong>. The first failing gate stops the chain, so every
  later gate shows as <em>not evaluated</em> — that is not a pass.<br><br>
  <strong>Metrics:</strong> <strong>pass@1</strong> is the fraction of individual
  episodes that passed. <strong>pass^k</strong> is the fraction of tasks where
  <strong>all k seeds</strong> passed; a gap between the two is flakiness.<br><br>
  <strong>Layer B transfer:</strong> <strong>transfer_delta</strong> is
  <em>only</em> a harness effect if the run was fair. Check the per-harness
  columns before concluding anything: a <span class="bad">timed out</span> run
  was killed at its budget and never finished, and
  <span class="bad">compactions</span> &gt; 0 means the harness dropped earlier
  context — often the tool output the task depended on. Also note that no
  harness can set Ollama's runtime <code>num_ctx</code>, so the harness phase
  runs at the model's default window while Layer A runs at the configured
  <code>num_ctx</code>; the window actually loaded is reported per run as
  <em>Loaded ctx</em>.<br><br>
  <strong>Empty answers fail.</strong> The grounding and fabrication verifiers
  require a final answer: a harness that was killed or never spoke scores a
  fail, not a free pass.<br><br>
  <strong>Axes:</strong><br>
  &nbsp;· probe — read a file and report its contents<br>
  &nbsp;· completion — produce a correct code change end to end<br>
  &nbsp;· edit — surgical edits across a multi-file codebase<br>
  &nbsp;· instruction — follow the harness rules and end with finish<br>
  &nbsp;· grounding — stick to what is actually in the workspace<br>
  &nbsp;· fabrication — refuse to invent facts absent from the files<br>
  &nbsp;· recovery — diagnose and fix a failing test suite
</div>
"""


def render_agent_report(root: dict, source: str) -> str:
    cfg = root.get("config", {})
    models = root["models"]
    smoke = any("tier" not in r for r in models.values())

    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Agentic Benchmark Report</title>
<style>
{BASE_CSS}
</style>
</head>
<body>
<h1>Agentic Benchmark Report</h1>
<p class="meta">{_meta_line(cfg)} &nbsp;|&nbsp; Source: {e(source)}</p>
{_provenance_card(cfg)}
{TIPS}
"""
    if smoke:
        body = _smoke_section(models)
    else:
        body = _full_sections(root, models)
    return head + body + "</body>\n</html>\n"


def _smoke_section(models: dict) -> str:
    rows = ""
    for model, r in models.items():
        probe = r.get("smoke_probe") or {}
        tools = ", ".join(e(tc.get("name"))
                          for tc in (probe.get("tool_calls") or []))
        rows += (f"<tr><td><a href='#{_slug(model)}'>{e(model)}</a></td>"
                 f"<td>{_mark(bool(probe.get('passed')))}</td>"
                 f"<td>{e(probe.get('verify_reason'))}</td>"
                 f"<td>{e(probe.get('steps'))}</td>"
                 f"<td>{tools}</td></tr>\n")
    head = "<tr><th>Model</th><th>Pass</th><th>Reason</th>" \
           "<th>Steps</th><th>Tools used</th></tr>"
    return f"<h2>Model Smoke Results</h2><table>{head}{rows}</table>"


def _full_sections(root: dict, models: dict) -> str:
    cfg = root.get("config", {})
    ranked = sorted(models.items(), key=_rank_key, reverse=True)

    # ── rankings table ──
    rows = ""
    for i, (model, r) in enumerate(ranked, 1):
        rel = r["reliability"]
        tasks = rel.get("tasks", 0)
        full = (f"{round(rel['pass_pow_k'] * tasks)}/{tasks}"
                if tasks else "—")
        ci = rel.get("ci") or []
        ci_s = (f"{ci[0]:.2f}–{ci[1]:.2f}" if len(ci) == 2 else "—")
        rows += (
            f"<tr><td>{i}</td>"
            f"<td><a href='#{_slug(model)}'>{e(model)}</a></td>"
            f"<td>{_tier_badge(r['tier'])}</td>"
            f"<td>{rel['pass_at_1']:.2f}</td>"
            f"<td>{rel['pass_pow_k']:.2f}</td>"
            f"<td>{ci_s}</td><td>{full}</td>"
            f"<td>{e(r.get('failed_gate')) or '—'}</td>"
            f"<td>{r.get('p50_seconds', 0):.0f}</td>"
            f"<td>{r.get('p90_seconds', 0):.0f}</td>"
            f"<td>{e(r.get('median_steps'))}</td>"
            f"<td>{e(r.get('median_tokens'))}</td></tr>\n"
        )
    rank_head = ("<tr><th>#</th><th>Model</th><th>Tier</th><th>Pass@1</th>"
                 "<th>Pass^k</th><th>95% CI</th><th>Tasks ✓</th>"
                 "<th>Gate fail</th><th>p50 s</th><th>p90 s</th>"
                 "<th>Med steps</th><th>Med tokens</th></tr>")
    out = f"<h2>Rankings</h2><table>{rank_head}{rows}</table>"

    # ── per-model sections ──
    for model, r in ranked:
        out += _model_section(root, model, r)
    return out


def _model_section(root: dict, model: str, r: dict) -> str:
    cfg = root.get("config", {})
    rel = r["reliability"]
    gates = r.get("gates") or {}
    failed_gate = r.get("failed_gate")
    axes = r.get("axes") or {}
    axes_order = list(axes.keys())

    if r["tier"] == "BLOCKED":
        reason = e((gates.get(failed_gate) or {}).get("reason"))
        verdict = (f"BLOCKED at {e(failed_gate)}: {reason}. "
                   f"Not usable in a harness until this gate passes.")
    else:
        k = cfg.get("k") or rel.get("k") or "?"
        skipped = r.get("unevaluated_gates") or []
        scope = (f" Scope: {', '.join(skipped)} had no task in this run, so "
                 f"they are not evaluated and the verdict cannot exceed "
                 f"SUPERVISED." if skipped else "")
        verdict = (f"{e(r['tier'])} — {rel['pass_pow_k']:.0%} of tasks "
                   f"passed on all {k} seeds.{scope}")
    verdict_card = (f"<div class='card'>{_tier_badge(r['tier'])} "
                    f"&nbsp; {e(verdict)}</div>")

    episodes = r.get("episodes") or []
    failing = [ep for ep in episodes if not ep["passed"]]
    failed_rows = _failure_details(failing, axes_order)

    rules = r.get("rules")
    rules_html = ""
    if rules and any(x.get("total") for x in rules.values()):
        rules_html = (f"<h3>Harness rules</h3>{_rules_table(rules)}")

    lb_entry = (root.get("layer_b") or {}).get(model)
    lb_html = (f"<h3>Layer B transfer</h3>{_layer_b_block(lb_entry)}"
               if lb_entry else "")

    return f"""
<h2 id="{_slug(model)}">{e(model)}</h2>
{verdict_card}
<h3>Gate chain</h3>
{_gate_chain_table(gates, failed_gate)}
<h3>Axes (pass^k)</h3>
{_axes_table(axes)}
<h3>Task results</h3>
{_task_matrix(episodes, axes_order)}
<h3>Failure details ({len(failing)})</h3>
{failed_rows}
<details><summary>All episodes ({len(episodes)})</summary>
{_all_episodes_table(episodes)}
</details>
{rules_html}
{lb_html}
"""


# ─────────────────────────────────────────────────────────────
# dispatch + CLI
# ─────────────────────────────────────────────────────────────

def detect_kind(root) -> str:
    if isinstance(root, dict) and "models" in root and "config" in root:
        return "agent"
    if (isinstance(root, dict) and root and
            all(isinstance(v, dict) and "tests" in v for v in root.values())):
        return "quality"
    if (isinstance(root, dict) and root and
            all(isinstance(v, dict) and "overall" in v for v in root.values())):
        return "speed"
    raise ValueError(f"unrecognised benchmark JSON: {root!r:.200}")


def render_to(src_path: str, out_path: str) -> str:
    """Load a result JSON, dispatch to the right renderer, write the HTML."""
    with open(src_path, encoding="utf-8") as f:
        root = json.load(f)
    kind = detect_kind(root)
    base = Path(src_path).name
    if kind == "agent":
        html_out = render_agent_report(root, src_path)
        Path(out_path).write_text(html_out, encoding="utf-8")
        print(f"📊  HTML report → {out_path}")
    elif kind == "quality":
        # Delegate to the existing renderer — do not reimplement.
        import benchmark_quality
        categories = sorted({c for m in root.values()
                             for c in m.get("categories", {})})
        mctx = re.search(r"_ctx(\d+)_", base)
        num_ctx = int(mctx.group(1)) if mctx else 0
        think = "_think_" in base and "_nothink_" not in base
        benchmark_quality.save_html_report(root, categories, num_ctx,
                                           out_path, think)
    else:  # speed
        import benchmark_ollama
        benchmark_ollama.save_html_report(root, out_path)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="benchmark result JSON files")
    p.add_argument("-o", "--out", help="output HTML path (single input only)")
    args = p.parse_args()
    if args.out and len(args.inputs) > 1:
        p.error("-o requires exactly one input")

    skipped = False
    for src in args.inputs:
        out = args.out or str(Path(src).with_suffix(".html"))
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            render_to(src, out)
        except Exception as err:  # one bad file must not stop the batch
            print(f"⚠️  skipped {src}: {err}")
            skipped = True
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())