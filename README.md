# Ollama Model Benchmark Suites

Benchmarking toolkit for local [Ollama](https://ollama.com) deployments. Three
independent tools, each answering a different question:

| Tool | Question | Mechanism |
|---|---|---|
| `benchmark_agent.py` | **Can the model drive a coding agent?** | Hermetic Layer A tool-loop + real Layer B harnesses (`opencode` / `omp` / `cline`), gate-first verdicts |
| `benchmark_quality.py` | **Are the answers correct?** | Deterministic, machine-verifiable correctness tests (execution, numeric, tool-call JSON, facts) |
| `benchmark_ollama.py` | **How fast is inference?** | Throughput/latency only (TTFT, TPS) |
| `report_html.py` | **What passed, what failed, and why?** | Renders any result JSON as a self-contained dark-themed HTML report (no JS, no CDN) |

The three benchmarks are complementary: `benchmark_ollama.py` measures *speed*,
`benchmark_quality.py` measures *single-turn correctness*, and
`benchmark_agent.py` measures the *end-to-end agentic ability* — whether a
model can hold a multi-step tool loop over real code, and whether that
ability transfers to the actual CLI harnesses you would ship.
`report_html.py` turns any of their JSON outputs into a readable report.

Requirements: Python 3.10+, `pip install requests`, and a reachable Ollama
server (default `http://192.168.0.149:11434`). Layer B additionally needs the
harness binaries on `PATH`: `opencode`, `omp`, and/or `cline` (the last is
only present if you install it).

See [`CHANGELOG.md`](CHANGELOG.md) for the full history of this rewrite.
Benchmark run output is written to `results/` (git-ignored) via `--json`, with
an HTML report written alongside it automatically.

---

## 1. `benchmark_agent.py` — gate-first agentic benchmark

The centerpiece. It is a **two-layer** benchmark over the same fixtures and
verifiers, scored by **binary gates** followed by a **reliability profile** —
never a weighted mean.

### Layers

- **Layer A (native):** a hermetic tool loop against Ollama `/api/chat` with
  native tool schemas. A sandbox workspace and a 7-tool set — `read_file`,
  `list_dir`, `grep`, `write_file`, `edit_file`, `run_tests`, `finish` —
  feed each model a set of Python fixtures. The model must perform edits,
  run tests, and terminate — all graded on **end state** (files, subprocesses,
  nonces), never on free prose.
- **Layer B (real harnesses):** — the *same* fixtures and *same*
  `task.verify()` verdict, driven through the actual binaries:
  `opencode`, `omp`, and `cline`. Layer B grading never reads harness stdout;
  it re-runs `task.verify()` on the workspace.

Because Layer A and Layer B share fixtures *and* verifiers, the delta between
them is a clean measure of **harness transfer** (an opencode model that
fails in the native loop but passes in Layer B signals an environment
difference, not a capability difference).

### Stages (each runs up to and including itself)

| Stage | What it does | Why |
|---|---|---|
| `probe` | `/api/show` every model | cheap fleet filter, no GPU |
| `smoke` | `probe_read` + one completion task, k=1 | drops models that can't even emit one tool call (G1) |
| `native` | all 19 tasks × k samples, Layer A | computes G2–G5 and the axis reliability profile |
| `e2e` | the 6 E2E tasks × k × chosen harnesses | Layer B transfer on top of a native run |

### Gates (G0–G5)

Gates run in order on Layer A. A model is elevated to a tier only if
**all** gates pass; the first failing gate names the blocking reason.

| Gate | Check | Drops a model when… |
|---|---|---|
| G0 declares tools | `/api/show` exposes a `tools` template | the model can't do tool calling at all |
| G1 emits tool call | probe episode issued ≥ 1 tool call | the model never calls a tool |
| G2 schema-valid | ≥ 0.98 of tool calls parse against the schema | calls are malformed |
| G3 terminates | every episode hits `finish`, not step/wall budget | the loop never ends |
| G4 no fabrication | every fabrication sample grounds its number in tool output | it invents file contents without reading them |
| G5 workspace-safe | zero path escapes | it writes/reads outside the sandbox |

### Tiers (after gates)

- `HARNESS_READY` — all gates pass, overall `pass^k ≥ 0.80`, `edit ≥ 0.80`,
  `instruction ≥ 0.70`.
- `SUPERVISED` — passes all gates but misses one reliability bar.
- `NOT_RECOMMENDED` / `BLOCKED` — the model fails a gate; a **reason** is
  always reported (`failed_gate`), never a 0-score.

### Reliability profile

`pass@1` (fraction of episodes that pass), `pass^k` (fraction of tasks where
*all k* samples pass — the honest metric for "will it always work"), and a
Wilson 95% CI on `pass@1`. Reported per axis (`probe`, `completion`, `edit`,
`instruction`, `grounding`, `fabrication`, `recovery`) and overall.

### Running

```bash
# 19 tasks × 3 samples on one model → 57 episodes
python benchmark_agent.py --stage native --models qwen3-coder:30b -k 3

# Full fleet on all tasks, 5 samples each
python benchmark_agent.py --stage native -k 5

# Layer B on top of a native run (harness transfer)
python benchmark_agent.py --stage e2e --harness opencode --harness omp --models qwen3-coder:30b -k 3

# Probe only (cheap fleet filter)
python benchmark_agent.py --stage probe
```

The 6 E2E tasks (Layer B subset) are `c1_add_function`, `c2_fix_failing_test`,
`e02` (edit `src/string_utils.py`), `e03` (edit `src/report_gen.py`),
`g4_fabrication`, and `a5_recovery` — the same 6 are run per selected harness
in Layer B.

### Output

| Flag | Effect |
|---|---|
| `--json results/run.json` | full report (gates, per-task pass, reliability, Layer B transfer) **and** `results/run.html` alongside it |
| `--html path.html` | override the HTML path; works with or without `--json` |
| `--omp-context N` | pin omp's context budget (see the caveat below); refused if it doesn't clear omp's 32768 output reserve by 8192 |

Layer B is also summarised on the console now (it used to be computed and
silently dropped unless `--json` was passed).

### Reading a Layer B `transfer_delta` honestly

`transfer_delta` = harness `pass^k` − native `pass^k` on the same 6 tasks. It
is only a *harness* effect when the run was fair, so every sample records the
conditions it was measured under:

- **Budget.** All harnesses get the same wall budget
  (`harness_drivers.E2E_BUDGET_S`, 300 s) and a killed run kills the whole
  process tree. Previously opencode got 120 s while omp got 90 s *plus* its own
  `--max-time` self-cap, and opencode's worker subtree survived the kill — so
  omp lost on budget, not capability. A `timed_out` sample never finished:
  read it as "no verdict", not "failed".
- **Thinking.** omp resolves thinking from `auto`, which picked `high` for a
  27b reasoning model and burned the entire budget before the first tool call.
  It is now *requested* as `off` (`harness_drivers.OMP_THINKING`) to match
  Layer A, which sends no thinking directive. **A request is not a guarantee:**
  a thinking model may keep emitting reasoning anyway, so each sample also
  records `thinking_parts` — the reasoning the model actually produced. Trust
  that column over `thinking_level`.
- **Context.** Neither harness can set Ollama's runtime `num_ctx` — both speak
  an OpenAI-compatible API — so the harness phase runs at the model's **default
  window** while Layer A runs at `--num-ctx`. opencode's *client-side* budget is
  capped to `--num-ctx`; omp's is left alone by default because its window is
  coupled to a non-overridable 32768 output reserve (pinning the window to
  32768 leaves zero input budget and sends omp into a compaction loop that
  discards tool output). The window actually loaded is recorded per sample as
  `server_context`.
  Measured: `--omp-context 65536` moved omp's *budgeting* but left
  `server_context` at 112000 and changed the outcome not at all
  (`pass^k` 0.17 → 0.17), so context is usually **not** the binding constraint —
  check `timed_out` and `thinking_parts` first.
- **Compaction.** `compactions > 0` means the harness dropped earlier context,
  frequently the tool output the task depended on.
- **Throughput.** A harness can simply be too slow for a given model. Measured:
  a trivial one-file task through omp on `qwen3.6-unsloth-vl-agent:27b-112k`
  took **598 s**, so no task finished inside a 300 s budget (30/30 timeouts)
  while opencode completed the same tasks in 45–300 s.

`config.measurement_version` records this definition. **Runs with different
measurement versions are not comparable** on `transfer_delta` or on the
fabrication/grounding axes.

---

## 2. `benchmark_quality.py` — correctness benchmark

Deterministic correctness evaluation. Each response is graded by a
machine-verifiable method, never a heuristic:

| `eval_type` | Mechanism | Used for |
|---|---|---|
| `numeric` | regex-extract number; check within `tolerance` | math / reasoning |
| `code_execution` | extract code block; run in a subprocess; compare `repr(result)` | coding |
| `tool_call` | parse first JSON object; check tool name + required args | agentic / mcp |
| `contains_all` | all `required_facts` present (case-insensitive) | summarization |
| `exact` | exact string match | factual answers |
| `regex` | response matches pattern | structured output |

```bash
# Default: the standard 110-test suite
python benchmark_quality.py --suite test_suite_v3.json

# Only cert categories
python benchmark_quality.py --suite test_suite_v3.json --categories coding agentic

# Reproducible runs share a seed (default 42)
python benchmark_quality.py --suite test_suite_v3.json --seed 42

# Larger context for MCP / agentic tests
python benchmark_quality.py --suite test_suite_extreme.json --num-ctx 16384

# List models / quick preview
python benchmark_quality.py --list
python benchmark_quality.py --quick
```

Notes on the environment:
- Temperature is **0.0** and sampling is seeded for reproducibility.
- `--think` is **capability-gated**: it is only sent to models that actually
  support thinking, so a non-thinking model is never force‑400'd. The thinking
  stream is recorded separately from the answer.
- A response that puts its answer *only* in the `thinking` field is a
  **extraction failure** (fails), not a pass. A response truncated at
  `num_predict` is recorded with its truncation reason.

---

## 3. `benchmark_ollama.py` — speed benchmark (TPS/TTFT only)

Throughput and latency, nothing else. Heuristic quality scoring was removed;
content correctness is `benchmark_quality.py` / `benchmark_agent.py`'s job.

```bash
python benchmark_ollama.py
python benchmark_ollama.py --quick            # 1 test per category
python benchmark_ollama.py --models qwen3.5:9b deepseek-r1:32b
```

Tiers are TPS-only: EXCELLENT ≥30, GOOD ≥15, ADEQUATE ≥8, MARGINAL ≥3, POOR <3.

---

## 4. `report_html.py` — HTML reports

Turns any benchmark result JSON into a **single self-contained HTML file**:
inline CSS, no JavaScript, no CDN, no external assets. Collapsible sections use
native `<details>`, so the file works offline, prints, and survives being
emailed.

```bash
# render existing results (one command, many files; shell glob works)
python report_html.py results/*.json

# explicit output path (single input only)
python report_html.py -o report.html results/v5_e2e.json
```

The input shape is auto-detected, so the same command handles all three
benchmarks:

| Detected kind | Condition | Renderer |
|---|---|---|
| `agent` | top level has `config` + `models` | full agent report (below) |
| `quality` | every value has `tests` | delegates to `benchmark_quality.save_html_report` |
| `speed` | every value has `overall` | delegates to `benchmark_ollama.save_html_report` |

A file that fails to parse prints `⚠️  skipped …` and the batch continues;
the exit code is `1` if anything was skipped.

The **agent** report answers "what passed, what failed, and why" without
opening the JSON:

- *How to read this report* — tier meaning, gate-chain semantics,
  `pass@1` vs `pass^k`, what each axis probes, and when a `transfer_delta` is
  **not** a harness effect.
- *Measurement definition* card — budget, thinking level, context handling,
  verifier policy, `measurement_version`. Pre-v2 artifacts render an explicit
  "not comparable" warning instead.
- *Rankings* — sorted by the same key the console uses, so report and terminal
  always agree.
- *Per model* — verdict, the full G0–G5 chain (gates the run never reached are
  shown as *not evaluated*, never as passes), axis bars, and a
  **task × seed matrix** with the verifier's reason for the lowest failing seed.
- *Failure details* — per failing episode: budget/termination facts, the tool
  call trace with arguments and output, and the final text. Passing episodes
  stay in a compact collapsed table so the file doesn't balloon.
- *Layer B* — per-harness transfer with a signed delta, plus per-run
  `timed out` / `compactions` / `thinking` / `ctx budget` / `loaded ctx` /
  `answer chars` so an unfair run is visible at a glance.

Both the agent benchmark and the two older benchmarks write their HTML
automatically at run time; `report_html.py` exists to re-render artifacts you
already have (for example after a report improvement).

---

## Diagnostic

```bash
python diagnose_ollama.py --model qwen3.5:27b --prompt "What is 17*23?"
```
Dumps raw streaming API chunks to debug models writing to the wrong field.

---

## Test Suites

- `test_suite_v3.json` — standard, 110 tests (reasoning/coding/agentic/
  summarization × 25, mcp 10). **Recommended.**
- `test_suite_hard.json` — 40 harder tests (DP, Bayesian, nested tool schemas).
- `test_suite_extreme.json` — 40 hard-to-break tests drawn from HumanEval hard,
  AIME-style math, LeetCode hard.

Suite format: a `tests` array; each test has `id`, `name`, `category`,
`eval_type`, `difficulty`, `prompt`, `max_tokens`, plus eval-specific fields
(`expected_number`, `test_cases`, `required_args`, `expected_tool`, …).

### Ollama hosts

Two hosts are pinned explicitly as constants:

- `OLLAMA_HOST = "http://192.168.0.149:11434"` (the rich fleet — this is the
  default benchmark target)
- `http://127.0.0.1:11434` (secondary host with a different model set)

Each script takes `--host` to override.