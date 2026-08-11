# Changelog

All notable changes to this repository. Format follows
[Keep a Changelog](https://keepachangelog.com/) conventions;
the project is pre-1.0, so the current series is `0.x.y`.

## [Unreleased]

## [0.3.1] — 2026-08-11 — gates stop failing on evidence a run never collected

Every axis-filtered run reported
`BLOCKED at G1_emits_tool_call: probe produced no tool calls`, for every
model — including cloud models that were calling tools successfully on every
task in scope.

### Fixed

- **G1 threw away its own evidence.** The probe episode is produced by the
  smoke stage (`smoke_eps`), which runs for every model in every stage, but
  only `episodes` from the native pool reached `evaluate_gates`. `--axes`
  filters the task pool, `g1_probe_read` sits on the `probe` axis, so the
  probe set was empty and `any([])` read as "emitted no tool calls". The
  smoke probe is now passed to the gate (`probe_episodes=`) and stored in the
  report as `smoke_probe`, so G1 is decided on the measurement that was
  actually taken. Confirmed across every artifact on disk: all six 19-task
  runs had `probe_ran=True, G1=True`; all four filtered runs had
  `probe_ran=False, G1=False`.
- **The same empty-evidence bug in the rest of the chain.** G4 failed with
  "no fabrication episodes" whenever the fabrication axis was out of scope,
  and G2's reason string divided by a zero call count (`ZeroDivisionError`,
  reachable as soon as G1 could pass on probe evidence alone). A gate without
  evidence is now `passed: None` — **not evaluated** — the chain continues
  past it, and the ids are listed in the report's `unevaluated_gates`.
- **`native_pool` gave every model the same profile.** The dict comprehension
  bound `prof` from the *previous* loop's leaked variable, so every entry
  carried the last-probed model's profile — and that profile is what
  `run_e2e` hands to `driver.prepare()`. Only visible on multi-model runs.

### Changed

- A run with an unevaluated gate can no longer be certified `HARNESS_READY`;
  the strongest honest verdict is `SUPERVISED`, since the gate was skipped,
  not passed (`gates.tier_for(..., unevaluated=[...])`).
- `GateResult.verdict` distinguishes `NOT EVALUATED` from `BLOCKED`; the HTML
  gate chain renders a skipped gate as *not evaluated* with its reason instead
  of a ✗, and the verdict card names the scope limit. The console prints
  `(not evaluated: G4_no_fabrication)` where it used to print a gate id.
- `config.axes` records which axes ran, so a missing gate or axis is legible
  as a scope fact. **`measurement_version` is deliberately not bumped:**
  `pass@1`, `pass^k` and `transfer_delta` are unchanged by this release — only
  verdicts that were previously fabricated from absent evidence change.

### Verification

- `python -m pytest tests/ -q` → **78 passed** (67 → 78; 11 new tests covering
  probe-sourced G1, the axis-filtered fallback, out-of-scope G2/G3/G4/G5, the
  tier cap, `verdict` semantics, and the renderer).
- Live, `--axes completion -k 1`, the exact shape that used to fail:
  - `minimax-m3:cloud` (previously `BLOCKED at G1`): G1 passes on probe
    evidence, G2 `60/60` calls valid, and the run blocks at **G3** because
    `c1_add_function` hit the 10-step budget without calling `finish` — a real
    measured failure.
  - `llama3.2:3b`: G1–G3 pass, G4 **not evaluated** (out of scope), blocked at
    **G5** on 2 genuine path escapes in `c2_fix_failing_test`.
- `python report_html.py results/*.json` re-renders all 13 existing artifacts,
  including pre-fix ones.

## [0.3.0] — 2026-08-11 — omp Layer B actually runs (stdin fix + provider config)

omp's Layer B result was an artifact of the runner, not of the model. Fixing
it moved `transfer_delta` on `qwen3.6-unsloth-vl-agent:27b-112k` from
**−0.67 with 30/30 budget kills to +0.00 with 0/6**, and a trivial one-file
task from **598 s to 31.8 s**.

### Fixed

- **The runner leaked its stdin into every harness** (`harness_drivers.py`,
  `_run_proc`). `Popen` piped stdout/stderr but left `stdin` inherited, and
  omp's `-p` mode reads piped stdin whenever stdin is not a tty — so it blocked
  in phase `readPipedInput` waiting for an EOF that never arrived. Measured
  directly: **1900 s in that phase with zero requests reaching Ollama**, i.e.
  the agent loop never started and the run could only ever end as a budget
  kill. This, not sampler or context tuning, is what produced 30/30 omp
  timeouts in v5/v7/v9. Now `stdin=subprocess.DEVNULL`; no harness consumes
  runner stdin, so it applies to all of them.

### Added

- `config/omp-models.yml` — **new.** omp provider config for Layer B, copied
  to `~/.omp/agent/models.yml`. Scoped to the `ollama` provider and one model
  entry, so cloud models are untouched:
  - `api: openai-completions` (the provider default was `openai-responses`,
    tied to stream-closed / length-truncated responses on Ollama),
  - `discovery.type: ollama` kept — without it a `models:` list *replaces*
    discovery and the other ~50 local tags vanish from omp,
  - `contextWindow: 112000` (must equal the tag's baked `num_ctx`),
    `maxTokens: 12288` (omp's registry reports a 32768 output reserve — as
    large as the entire window a 32768 pin would give),
  - `compat.streamIdleTimeoutMs: 1800000`,
  - `compat.extraBody.reasoning_effort: "none"` — see below.
- `OmpDriver.run` sets `PI_OPENAI_STREAM_FIRST_EVENT_TIMEOUT_MS` and
  `PI_OPENAI_STREAM_IDLE_TIMEOUT_MS` to the harness budget, **process-scoped**
  so omp's watchdogs can never fire before the budget and never leak into
  interactive cloud sessions.
- omp is invoked with `--no-skills --no-rules`: its preamble measured ~10–21k
  tokens against opencode's ~7–8k, and TTFT scales with prompt length.
- `config` records `omp_api`, `omp_flags`, `omp_max_tokens` so a report states
  which omp configuration produced a `transfer_delta`.
- README: an **omp setup for Layer B** section (install + verification
  one-liner), and an end-to-end recipe for running the benchmark and the HTML
  report against a freshly pulled model.

### Changed

`measurement_version: 3`. v2 runs (v7/v8/v9) are **not comparable** on
`transfer_delta`: omp now reaches the model at all, runs without skills/rules,
and does not think.

- **Thinking suppression is real now, and two obvious switches are dead ends.**
  `--thinking off` sends no reasoning parameter at all, and Ollama's `/v1`
  **ignores `chat_template_kwargs`** (measured: `enable_thinking: false` still
  returned a 531-character `reasoning` field). Baking the flag into a derived
  tag is also impossible: `ollama show --modelfile` emits
  `TEMPLATE {{ .Prompt }}`, not the model's 8057-character Jinja template, and
  Modelfile `TEMPLATE` is Go-only — feeding the real Jinja back to
  `ollama create` fails with `template error: function "content" not defined`,
  and building from the exported file as-is would have silently produced a
  passthrough-template tag. The one switch this host honors is
  `reasoning_effort: "none"` (measured reasoning length 0), shipped as
  per-model `compat.extraBody`. Layer B `thinking_parts`: **33 → 0**.
- `E2E_BUDGET_S` stays **300**: the post-fix probe is 36.4 s, far under the
  120 s threshold that would have justified raising it.

### Verification

- `python -m pytest tests/ -q` → **67 passed**.
- `omp models --json` → 590 total / **51 ollama** models (unchanged), tuned
  entry `112000 / 12288` — discovery survived the `models:` list.
- Layer B `k=1`, completion axis, `qwen3.6-unsloth-vl-agent:27b-112k`:
  `openai-completions` present in the event stream and `openai-responses`
  absent; `thinking_parts` 0; 6/6 samples finished in 30.9–125.0 s;
  `transfer_delta` **+0.00**.
- No history loss: max **23,882** total tokens across 116 omp requests against
  a **112,000** loaded window (`ollama ps`, `server_context: 112000`).
  (Ollama's `server.log` no longer records request lines on this host, so the
  planned `truncated = 1` grep was replaced by this direct token evidence.)
- **Cloud untouched** — the acceptance gate: `~/.omp/agent/config.yml` sha256
  `acdce014…67ec6` identical before and after, and a live
  `opencode-zen/deepseek-v4-flash-free` call returned normally. No global knob
  (`defaultThinkingLevel`, `providers.stream*Seconds`) was written.

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