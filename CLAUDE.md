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
| `benchmark_agent.py` | CLI: `--stage probe|smoke|native|e2e`, `--models`, `-k`, `--harness`, `--json` |
| `benchmark_quality.py` | correctness suite runner (default suite `test_suite_v3.json`) |
| `benchmark_ollama.py` | speed benchmark (TTFT/TPS) |

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
- `SKIP_MODELS = {"qwen3-embedding:8b", "deepseek-ocr:latest"}` — never run
  these (non-generative).

## Running

```bash
python benchmark_agent.py --stage probe                          # fleet filter (no GPU)
python benchmark_agent.py --stage smoke --models <m>             # G1 drop
python benchmark_agent.py --stage native --models qwen3-coder:30b -k 3   # 57 episodes
python benchmark_agent.py --stage e2e --harness opencode --harness omp --models <m> -k 3
python benchmark_quality.py --suite test_suite_v3.json --seed 42
python benchmark_ollama.py --quick
python -m pytest tests/ -q                                      # offline self-tests (V1)
```

## Conventions

- Every module imports cleanly on Windows (`pathlib`, no shell-isms; `.cmd`
  resolution for harness binaries).
- **Layer B (e2e) runs on every model that reached the native stage**, even a
  gate-blocked one — `transfer_delta` is measured separately from the gate
  verdict. The 6 E2E task ids are `c1_add_function`, `c2_fix_failing_test`,
  `e02`, `e03`, `g4_fabrication`, `a5_recovery`.
- JSON output: `results/` (git-ignored; regenerable) with a `config` header
  (host, ollama version, suite sha, harness versions) so runs are
  reproducible. Run verdicts are only persisted when you pass `--json` —
  including Layer B, which is **not** printed to the console.
- Determinism: temperature 0.0 + explicit `seed` for quality; seeded task
  sampling for the agentic loop.
- `benchmark_quality.py` `--think` is capability-gated via `probe_model`
  (a model without the `thinking` capability is never sent `think: true`).
