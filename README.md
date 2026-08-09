# Ollama Model Benchmark Suites

Benchmarking toolkit for local [Ollama](https://ollama.com) deployments. Three
independent tools, each answering a different question:

| Tool | Question | Mechanism |
|---|---|---|
| `benchmark_agent.py` | **Can the model drive a coding agent?** | Hermetic Layer A tool-loop + real Layer B harnesses (`opencode` / `omp` / `cline`), gate-first verdicts |
| `benchmark_quality.py` | **Are the answers correct?** | Deterministic, machine-verifiable correctness tests (execution, numeric, tool-call JSON, facts) |
| `benchmark_ollama.py` | **How fast is inference?** | Throughput/latency only (TTFT, TPS) |

The three are complementary: `benchmark_ollama.py` measures *speed*,
`benchmark_quality.py` measures *single-turn correctness*, and
`benchmark_agent.py` measures the *end-to-end agentic ability* — whether a
model can hold a multi-step tool loop over real code, and whether that
ability transfers to the actual CLI harnesses you would ship.

Requirements: Python 3.10+, `pip install requests`, and a reachable Ollama
server (default `http://192.168.0.149:11434`). Layer B additionally needs the
harness binaries on `PATH`: `opencode`, `omp`, and/or `cline` (the last is
only present if you install it).

See [`CHANGELOG.md`](CHANGELOG.md) for the full history of this rewrite.
Benchmark run output is written to `results/` (git-ignored) via `--json`.

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

`--json results/candidate.json` writes the full report (gates, per-task pass,
reliability, per-harness Layer B transfer delta).

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