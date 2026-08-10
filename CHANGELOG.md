# Changelog

All notable changes to this repository. Format follows
[Keep a Changelog](https://keepachangelog.com/) conventions;
the project is pre-1.0, so the current series is `0.x.y`.

## [Unreleased]

## [0.2.0] — 2026-08-10 — HTML reports + fair Layer B measurement

HTML reporting for every benchmark, plus a set of Layer B measurement fixes
that make `transfer_delta` mean what it claims.

### Added

- `report_html.py` — **new.** Canonical dark theme (`BASE_CSS`) plus a
  self-contained agent-report renderer: no JavaScript, no CDN, native
  `<details>` for collapsing. `detect_kind()` dispatches on JSON shape and
  delegates quality/speed to the existing renderers rather than reimplementing
  them. CLI re-renders artifacts already on disk
  (`python report_html.py results/*.json`), skipping unparsable inputs with a
  non-zero exit.
  The agent report answers "what passed, what failed, why": how-to-read
  guidance, a measurement-definition card, rankings sorted by the console's own
  key, the full G0–G5 chain (unreached gates render as *not evaluated*, never as
  passes), axis bars, a task × seed matrix with the verifier's reason, per-episode
  failure traces, and Layer B transfer with per-run fairness columns.
- `benchmark_agent.py`: `--json` now writes an HTML report alongside the JSON;
  `--html` overrides the path and works without `--json`. New `--omp-context`.
- Layer B is **printed to the console** (it was computed and silently dropped
  unless `--json` was passed).
- `config` now records the measurement definition: `measurement_version`,
  `e2e_budget_s`, `omp_thinking`, `omp_context`, `harness_runtime_num_ctx`,
  `system_prompt_scope`, `verifier_policy`.
- Per-sample Layer B provenance: `budget_s`, `context_budget`, `thinking_level`,
  `compactions`, `server_context`, `final_text_chars`.

### Changed — Layer B fairness (**breaking for comparisons**)

`measurement_version: 2`. Runs from v0.1.0 are **not comparable** on
`transfer_delta` or the fabrication/grounding axes.

- **One budget for every harness** (`E2E_BUDGET_S = 300`). Was opencode 120 s
  vs omp 90 s.
- **omp's `--max-time` self-cap removed.** It made omp the only harness that
  stopped its own agent loop at the budget; 30/30 omp runs in the v5 report died
  at exactly 90.0 s while opencode ran to 398 s.
- **Budget kills kill the process tree.** `subprocess`'s timeout only kills the
  direct child; opencode's worker subtree kept running and finishing edits past
  its nominal budget.
- **omp thinking pinned to `off`.** `auto` resolved to `high` for a 27b model
  and consumed the whole budget before the first tool call; Layer A sends no
  thinking directive.
- **opencode's client context budget capped to `--num-ctx`** (was the model's
  full window).
- **Empty answers no longer pass** the fabrication/grounding verifiers. omp
  previously scored 5/5 on `g4_fabrication` with zero tool calls, because
  "no fabricated integer" was true of an empty answer. Affects existing
  fixtures (v3 seed 1, v4 seed 1, v5 seeds 1 and 4 were vacuous passes).

### Fixed

- **Harness event parsing was opencode-only.** omp's real schema
  (`tool_execution_start|end`, `message_end.message.content[]`,
  `message_end.message.usage`) is now parsed, verified against a live
  `omp --mode json` run: omp token counts went from `(-1, -1)` to real values.
- `_scan_tool_names` substring-matched tool names against the serialized event,
  so prose mentioning `finish` counted as a tool call. Now structural.
- `_scan_tokens` is key-scoped (`usage` / `tokens` blocks), so a tool argument
  named `input` can no longer be reported as a token count — while still
  reading opencode's `step_finish.part.tokens`.
- **Per-run artifact tags.** Every sample's `raw_stdout` pointed at a single
  overwritten file (tag was model-only), making per-run forensics impossible.
- `_write_json` creates parent directories (a 49-minute run was lost to
  `FileNotFoundError`).
- `benchmark_quality.py`: four unescaped interpolations in `save_html_report`
  (`eval_detail.reason`, the code-execution `call`/`expected`/`actual`/`reason`
  cells, and `t['name']`) could corrupt the report, since `actual` is literal
  model output. Replaced the three ad-hoc `.replace` chains with one `_esc()`.
- `report_html.py`: the "How to read this report" block was defined but never
  emitted.
- **`thinking_level` was misleading.** It records the level the harness was
  *asked* for; with `--thinking off` omp emits no level-change event, so every
  sample read `''` while the model kept emitting thinking blocks. Added
  `thinking_parts` (assistant reasoning parts for omp, non-zero reasoning
  tokens for opencode), rendered as a flagged *Reasoning parts* column.

### Verification

- `python -m pytest tests/ -q` → **67 passed** (43 → 67; new coverage for the
  renderer, escaping, the omp/opencode parsers, the omp context guard, the
  thinking detector, and a real process-tree kill that asserts a grandchild
  stops).
- Every pre-existing artifact still renders, including ones lacking the new
  fields.

#### Layer B, measured across four k=5 runs

| run | model | harness | pass^k | delta | timeouts |
|---|---|---|---|---|---|
| v5 (pre-v2) | vl-agent 27b | opencode / omp | 1.00 / 0.33 | +0.00 / −0.67 | 16/30 / **30/30** |
| v7 (v2) | vl-agent 27b | opencode / omp | 0.83 / 0.17 | +0.00 / −0.67 | 5/30 / **30/30** |
| v8 (v2) | coder 30b | opencode / omp | 1.00 / 0.50 | +0.17 / −0.33 | 0/30 / 0/30 |
| v9 (v2) | vl-agent 27b, `--omp-context 65536` | omp | 0.17 | −0.67 | 29/30 |

What the fixes established, and what they did **not**:

- **The budget fix works** where the model is fast enough to use it: a `k=1`
  run on `llama3.2:3b` put opencode and omp at an identical `0.33 / −0.17`
  with 0/6 timeouts each, and the v8 control ran 0/30 timeouts on both.
- **Budget was not the whole story for the 27b model.** Raising 90 s → 300 s
  left omp at 30/30 timeouts and `−0.67` unchanged. Root cause measured
  directly: a *trivial* one-file task through omp on that model takes
  **598 s** (exit 0, tool call succeeded), because the model keeps emitting
  reasoning even with `--thinking off`. This is a harness × model throughput
  interaction, not the asymmetry that was fixed.
- **The context hypothesis was tested and rejected.** v9 pinned omp's window
  to 65536: `pass^k` 0.17 → 0.17, timeouts 30/30 → 29/30, and `server_context`
  stayed at 112000 — confirming `OLLAMA_CONTEXT_LENGTH` moves omp's budgeting
  but not the window Ollama loads.
- **A genuine omp transfer gap remains, separate from all of the above.** On
  the v8 control (0/30 timeouts, no compaction, no reasoning) omp still lost
  `−0.33`: it fails the exact-content edit tasks (`e02` 1/5, `e03` 2/5,
  `a5_recovery` 4/5) where opencode passes 5/5, with
  `src/string_utils.py differs from expected content`. That is an edit-fidelity
  difference and the one real harness issue the instrumentation now isolates.

## [0.1.0] — agentic benchmark rewrite

This branch replaces the single-turn, prose-graded quality + dead agentic
scorers with a two-layer, gate-first agentic benchmark. Summary of the changes
below, split by audience: the "What changed" section explains the mechanism
(human), "Implementation notes" records the constraints agents must preserve.

### What changed

The repo now ships three complementary tools (in `README.md`):

- `benchmark_agent.py` — **new.** Gate-first agentic benchmark over **real**
  native tool calling (`/api/chat` with JSON-Schema tools), not prose prompts.
  Both layers grade **end state** (files, subprocesses, nonces) with the
  **same verifier**.
- `benchmark_quality.py` — rebuilt correctness screen (external suite only,
  seeded, capability-gated `--think`, truncation-aware).
- `benchmark_ollama.py` — speed only (TTFT/TPS); heuristic quality scoring
  removed.

#### Removed (dead / fabricated-metric code)

- `benchmark_agentic.py` (946 lines; unreachable, wrong-tool-pass, constant
  context-retention, hardcoded hallucination whitelist).
- `benchmark_harness_usability.py` (reported fabricated numbers: long-context
  suite never ran, N=10 percentiles, synthetic timeout totals, substring
  failure classification).
- `test_suite_extended.json` (duplicated v3's first 40 tests).
- `AGENTIC_FRAMEWORK_README.md`, `HARNESS_USABILITY_README.md`.

#### New modules (agentic benchmark)

- `ollama_client.py` — shared `/api/chat` client: `probe_model`,
  seeded `chat`, `warmup`, bounded `call_with_retry`, `effective_num_ctx`.
- `agent_workspace.py` — hermetic sandbox + `ToolRegistry` (7 tools:
  `read_file`, `list_dir`, `grep`, `write_file`, `edit_file`, `run_tests`,
  `finish`); `SAFE_ENV` subprocess isolation; `PathEscape` guard.
- `agent_loop.py` — native tool loop (`run_episode` → `Episode`) with step /
  wall budgets, schema validation, path-escape tracking.
- `agent_tasks.py` — 19 tasks (axes: probe, completion, edit, instruction,
  grounding, fabrication, recovery), end-state verifiers, ~60k-token grounding
  corpus, `HARNESS_SYSTEM_PROMPT` (measured ~7.5–8.5k tokens).
- `harness_drivers.py` — Layer B: `OpenCodeDriver`/`OmpDriver`/
  `ClineDriver`/`NativeDriver`, with a project-level `opencode.json` written
  per run (never touches user config).
- `gates.py` — G0–G5 gate evaluation, tier verdicts, reliability
  (pass@1, pass^k, Wilson CI).

#### Fixes applied this session

- `harness_drivers.py`: subprocess capture no longer uses `text=True`
  (locale-dependent cp1252 decode crashed on byte `0x9d` in llama3.2:3b
  output). Now captures raw bytes and decodes `utf-8, errors="replace"`. The
  same hardening was applied to `ClineDriver.prepare` and `harness_versions`.
- `agent_tasks.py`: the G4 fabrication verifier is **harness-agnostic** — an
  integer is legitimate iff it appears in **any** tool output (list_dir *or*
  a file read), so the same verifier grades the native loop and the real
  harnesses (which use their own `read`/`glob`/`bash` palettes).
- `benchmark_agent.py`: Layer B now runs on every model that reached the
  native stage (not only gate-passing tiers), so harness transfer is measured
  even for gate-blocked models; `native_e2e_subset_ppk` reads serialized
  episode dicts; `print_summary` sorts inline on nested keys.
- `tests/`: 43 offline self-tests (V1), including a regression for the
  non-cp1252 byte-capture path.

### Verification performed

- V1 offline self-tests: `python -m pytest tests/ -q` → **43 passed**.
- V2 G0 discrimination: tool-less `gemma3:27b` / `codestral:22b` blocked at G0;
  tool models advance.
- V3 Layer A e2e (qwen3-coder:30b, k=3): exactly 57 episodes; tier BLOCKED at
  G3 (completion/edit/probe/fabrication/recovery at 1.00; instruction and
  grounding 0.00 — genuine, not a harness artifact).
- V4 Layer B (openCode + omp over the 6 E2E tasks): opencode transferred 1:1
  (delta 0.0); omp dropped to 0.5 (2 edit-task end-state misses + recovery
  run-tests miscount) — a real harness-palette delta.
- V5 fabrication gate: fires deterministically on a fabricated integer
  (injected payload) and 5 live model configs refused to fabricate (G4 1.00).
- V6 quality determinism: two seeded `benchmark_quality.py coding` runs had
  byte-identical per-test pass/score/reason (only TPS/TTFT differed).
- V7 sandbox: `PathEscape` blocks `../` and absolute paths; subprocess env is
  `SAFE_ENV`-only (no host `SECRET_TEST` leak).

### Implementation notes (constraints agents must preserve)

- **Verifiers check end state, never free prose.** `task.verify(ws, ep)` must
  inspect files/subprocesses/nonces. The same function grades all 4 backends.
- **Gates before scores** — a model that fails a gate is `BLOCKED` with a
  reason, never a 0-score.
- **`HARNESS_SYSTEM_PROMPT` is measured, not estimated:** 7,500–8,500 tokens
  via `prompt_eval_count` (~4.44 chars/token; 33,963 chars).
- **G4 fabrication:** tools are `[list_dir, finish]` (no `read_file`/`grep`).
  Pass iff no integer beyond what appears in tool output is in `final_text`.
  A number seen in any read/list output is grounded, not fabricated.
- **Layer A is hermetic:** sandboxed workspace, no network, `SAFE_ENV` for
  `run_tests` (no `SECRET_TEST` leak, no home-dir writes).
- **Windows-locale trap:** never decode a harness child pipe with the locale
  codec; capture bytes and decode `utf-8, errors="replace"`.
- **Hosts are pinned:** `OLLAMA_HOST = http://192.168.0.149:11434` (rich
  fleet) and `127.0.0.1:11434` (secondary) differ in model sets.
- `results/` is git-ignored; run artifacts are regenerable.

## [Prior work]

Represents the old single-turn framework (`benchmark_quality.py`,
`benchmark_ollama.py`, `diagnose_ollama.py`, the three test suites). Superseded
by this branch; retained only as history.