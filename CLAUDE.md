# CLAUDE.md

Guidance for working with this repository.

## Overview

Correctness-first benchmarking for local Ollama models, in three complementary
tools:

- `benchmark_agent.py` — **gate-first agentic benchmark**: a hermetic native
  tool loop (Layer A) plus the same fixtures/verifiers driven through the real
  `opencode` / `omp` / `cline` binaries (Layer B). Gates G0–G5, then a
  reliability profile; tier verdicts, never weighted averages.
- `benchmark_quality.py` — deterministic single-turn correctness benchmark
  (code execution, numeric, tool-call JSON, facts).
- `benchmark_ollama.py` — speed only (TTFT/TPS). Heuristic quality scoring
  was deliberately removed.

Default Ollama host: `http://192.168.0.149:11434`; `127.0.0.1:11434` is the
pinned secondary host. Deps: `requests` only. Python 3.10+.

## Module layout

| Module | Role |
|---|---|
| `ollama_client.py` | shared `/api/chat` client: probe (`ModelProfile`), seeded `chat`, warmup, bounded retries, `effective_num_ctx` |
| `agent_workspace.py` | hermetic sandbox (`Workspace`), `ToolRegistry` (7 tools), `SAFE_ENV` for subprocesses |
| `agent_loop.py` | native tool loop (`run_episode` → `Episode`) with step/wall budgets, schema validation, path-escape tracking |
| `agent_tasks.py` | 19 tasks (fixtures + end-state verifiers), grounding corpus (~60k tokens), `HARNESS_SYSTEM_PROMPT` (~7.5–8.5k tokens, measured) |
| `harness_drivers.py` | Layer B: `OpenCodeDriver` / `OmpDriver` / `ClineDriver` / `NativeDriver` |
| `gates.py` | G0–G5 evaluation, tier verdicts, reliability (pass@1, pass^k, Wilson CI) |
| `benchmark_agent.py` | CLI: `--stage probe\|smoke\|native\|e2e`, `--models`, `-k`, `--harness`, `--json`, `--html`, `--omp-context` |
| `benchmark_quality.py` | correctness suite runner (default suite `test_suite_v3.json`) |
| `benchmark_ollama.py` | speed benchmark (TTFT/TPS) |
| `report_html.py` | shared theme (`BASE_CSS`) + agent report renderer + `detect_kind` dispatch; CLI re-renders existing result JSON |

## Hard rules to preserve

- **Verifiers check end state, never free prose.** `task.verify(ws, ep)` must
  inspect files/subprocesses/nonces. Never grade the model's explanation.
- **Same verifier for all 4 backends** (native + opencode + omp + cline).
- **Gates before scores.** A model that fails a gate is `BLOCKED` with a
  reason — never a 0-score.
- **`HARNESS_SYSTEM_PROMPT` is measured, not estimated**: 7,500–8,500 tokens
  via `prompt_eval_count` on the target model (~4.44 chars/token).
- **Layer A is hermetic**: sandboxed workspace, no network, `SAFE_ENV` for
  `run_tests` (no `SECRET_TEST` leak, no home-dir writes).
- **G4 (fabrication)**: `read_file`/`grep` are deliberately absent (tools are
  `list_dir` + `finish`); pass iff no integer beyond what appears in **any**
  tool output (list_dir *or* a read) is in `final_text`. A number grounded in
  a read is legitimate — this is how the same verifier grades the native loop
  and the real harnesses (which use their own `read`/`glob`/`bash` palettes).
- **Windows-locale trap**: never decode a harness child pipe with the locale
  codec (`subprocess.run(..., text=True)` uses cp1252 and crashes on byte
  `0x9d`). Capture **bytes** and decode `utf-8, errors="replace"` yourself
  (see `harness_drivers._run_proc`).
- **Never let a harness inherit the runner's stdin.** `_run_proc` passes
  `stdin=subprocess.DEVNULL`. omp's `-p` mode reads piped stdin whenever stdin
  is not a tty and waits for an EOF an inherited pipe never delivers: measured
  1900 s stuck in phase `readPipedInput` with **no request reaching Ollama**,
  which is what produced the 30/30 omp "timeouts" in v5/v7/v9. A harness that
  looks pathologically slow is a plumbing suspect first.
- `SKIP_MODELS = {"qwen3-embedding:8b", "deepseek-ocr:latest"}` — never run
  these (non-generative).
- **Layer B must be measured fairly, or the delta is meaningless.** All
  harnesses share one budget (`harness_drivers.E2E_BUDGET_S`); no harness may
  get an *internal* self-cap that the others don't (omp's `--max-time` was
  removed for exactly this reason). A budget kill must kill the **process
  tree** — `subprocess`'s own timeout only kills the direct child, and the
  surviving worker subtree let opencode work for 398 s on a 120 s budget.
- **omp thinking is requested, not guaranteed** (`OMP_THINKING = "off"`):
  `auto` resolved to `high` on a 27b model and consumed the whole budget
  before the first tool call, while Layer A sends no thinking directive.
  `thinking_level` records what was *asked*; `thinking_parts` records what the
  model actually produced — never read the former alone as proof that a run
  did not reason. On Ollama, `--thinking off` alone sends no reasoning param
  and `/v1` ignores `chat_template_kwargs`; only
  `compat.extraBody.reasoning_effort: "none"` (see `config/omp-models.yml`)
  actually suppresses it. Baking it into a derived tag is not an option:
  `ollama show --modelfile` does not emit the model's Jinja template and
  Modelfile `TEMPLATE` is Go-only.
- **omp's provider config lives in `config/omp-models.yml`**, installed to
  `~/.omp/agent/models.yml`. Keep every local-model knob per-provider or
  per-model; `~/.omp/agent/config.yml` is global and shared with the user's
  cloud models, so `defaultThinkingLevel` and `providers.stream*Seconds` are
  off limits. Timeouts that must follow the harness budget go in
  `OmpDriver.run`'s process env (`PI_OPENAI_STREAM_*_TIMEOUT_MS`). A `models:`
  list without `discovery:` **replaces** discovery — keep `discovery.type`.
- **Never set omp's context from `num_ctx`.** omp couples its window to a
  32768 output reserve that its config surface does not let you lower, so
  `window == 32768` leaves zero input budget and triggers a compaction loop
  that silently discards tool output. Only the explicit `--omp-context` knob
  may move it, and `OmpDriver._safe_window` refuses anything that doesn't
  clear the reserve by `OMP_MIN_INPUT_BUDGET`.
- **No harness can set Ollama's runtime `num_ctx`** (both speak an
  OpenAI-compatible API). The harness phase therefore runs at the model's
  default window while Layer A runs at `--num-ctx`; record the truth via
  `server_context_length()` rather than claiming they match.
- **An empty answer is not abstention.** The grounding and fabrication
  verifiers require non-empty `final_text`; otherwise a killed or silent
  harness scores a free pass (omp once scored 5/5 on `g4_fabrication` with
  zero tool calls).
- **An absent evidence set is not a failed gate.** `--axes` filters the task
  pool, so a gate's task can be missing entirely (G4 without the fabrication
  axis, `g1_probe_read` without the probe axis). Such a gate is
  `passed=None` — *not evaluated* — the chain continues past it, its id lands
  in `unevaluated_gates`, and the tier is capped below `HARNESS_READY`. Never
  reintroduce `any([])`/`bool([]) and …` verdicts: they fabricate a negative
  result ("probe produced no tool calls") for a measurement nobody took. G1's
  evidence comes from the smoke stage, which always runs the probe, and is
  threaded in via `evaluate_gates(..., probe_episodes=...)`.
- **Parse harness events structurally, per schema.** opencode emits
  `part.tool` / `part.type == "text"` / `step_finish.part.tokens`; omp emits
  `tool_execution_start|end` / `message_end.message.content[]` /
  `message_end.message.usage`. Never substring-match tool names against a
  serialized event — prose mentioning `finish` counted as a call.
- **Per-run artifact tags.** `_run_proc`'s tag must include task+seed, or every
  sample's `raw_stdout` points at one overwritten file.
- **Record the measurement definition** in `config` (`measurement_version`,
  `e2e_budget_s`, `omp_thinking`, `verifier_policy`, `system_prompt_scope`).
  Changing what a metric means requires bumping `measurement_version` so old
  and new runs are never silently compared.
- **`HARNESS_SYSTEM_PROMPT` is applied on the instruction axis only**
  (`benchmark_agent.py`), so the other 6 axes measure *spontaneous*
  termination. Applying it everywhere would make G3/R2 trivially passable and
  break comparability — treat that as a measurement-definition change, not a
  bug fix.

## Running

```bash
python benchmark_agent.py --stage probe                          # fleet filter (no GPU)
python benchmark_agent.py --stage smoke --models <m>             # G1 drop
python benchmark_agent.py --stage native --models qwen3-coder:30b -k 3   # 57 episodes
python benchmark_agent.py --stage e2e --harness opencode --harness omp --models <m> -k 3
python benchmark_agent.py --stage e2e --models <m> -k 5 --json results/run.json  # + run.html
python report_html.py results/*.json                            # re-render existing artifacts
python benchmark_quality.py --suite test_suite_v3.json --seed 42
python benchmark_ollama.py --quick
python -m pytest tests/ -q                                      # offline self-tests
```

## Conventions

- Every module imports cleanly on Windows (`pathlib`, no shell-isms; `.cmd`
  resolution for harness binaries).
- **Layer B (e2e) runs on every model that reached the native stage**, even a
  gate-blocked one — `transfer_delta` is measured separately from the gate
  verdict. The 6 E2E task ids are `c1_add_function`, `c2_fix_failing_test`,
  `e02`, `e03`, `g4_fabrication`, `a5_recovery`.
- JSON output: `results/` (git-ignored; regenerable) with a `config` header
  (host, ollama version, suite sha, harness versions, and the measurement
  definition) so runs are reproducible. `--json` also writes an HTML report
  next to the JSON; `--html` overrides that path and works on its own. Layer B
  is printed to the console as well as persisted.
- Determinism: temperature 0.0 + explicit `seed` for quality; seeded task
  sampling for the agentic loop.
- `benchmark_quality.py` `--think` is capability-gated via `probe_model`
  (a model without the `thinking` capability is never sent `think: true`).
