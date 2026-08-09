# Changelog

All notable changes to this repository. Format follows
[Keep a Changelog](https://keepachangelog.com/) conventions;
the project is pre-1.0, so this tracks a single unnumbered rewrite.

## [Unreleased] — agentic benchmark rewrite

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