# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Correctness-first benchmarking framework for local Ollama language models. Evaluates whether models produce correct answers (not just fast ones) by executing code, parsing JSON tool calls, and verifying facts. Targets ~34 models on a 24 GB VRAM / 64 GB RAM local Ollama deployment.

## Running Benchmarks

```bash
# Full benchmark (all models, all tests, default context 4096)
python benchmark_quality.py

# Specific models
python benchmark_quality.py --models llama3.2:3b qwen3-coder:30b

# Specific categories
python benchmark_quality.py --categories coding agentic

# Custom test suite
python benchmark_quality.py --suite test_suite_extreme.json

# Larger context (required for MCP tests which use 400-700 tokens for tool schemas)
python benchmark_quality.py --num-ctx 16384

# Thinking mode (for Qwen3, DeepSeek-R1, GLM-4 — not all models support it)
python benchmark_quality.py --num-ctx 48000 --think

# Quick preview (2 tests per category)
python benchmark_quality.py --quick

# List available models / export built-in suite
python benchmark_quality.py --list
python benchmark_quality.py --save-suite

# Speed-only benchmark
python benchmark_ollama.py

# Debug streaming issues
python diagnose_ollama.py --model <model-name>
```

## Architecture

Three single-file scripts with no package structure. Only external dependency is `requests`.

**`benchmark_quality.py`** — Main script (1,475 lines). Flow:
1. Load test suite JSON → get available models from Ollama API
2. For each model × test: call model via `/api/generate` (streaming JSONL), extract response, evaluate
3. Aggregate results → output JSON + CSV + HTML

**`benchmark_ollama.py`** — Speed benchmark measuring TTFT/TPS with heuristic quality scoring.

**`diagnose_ollama.py`** — Debugging utility for models that output to wrong response fields.

### Evaluation Types

| `eval_type` | Mechanism |
|---|---|
| `numeric` | Regex-extract number; check within tolerance |
| `code_execution` | Extract code block; run in Python subprocess (10s timeout); compare `repr(result)` |
| `tool_call` | Parse first JSON object; verify tool name + required args; substring match on expected values |
| `contains_all` | All required facts/keywords present (case-insensitive) |
| `exact` | Exact string match (case-insensitive) |
| `regex` | Response matches pattern |

### API Call Parameters

- Temperature: **0.05** (fixed, intentional for reproducibility)
- Default context: 4096 tokens
- MCP tests need 8192+; extreme suite needs 16384+
- Thinking mode sends `think: true` to Ollama

### Output Files

Named `quality_ctx{context}_{think|nothink}_{timestamp}.{json,csv,html}`. JSON contains full per-model, per-test results including pass/fail, TPS, TTFT, and model response text. CSV has ranked summary. HTML is an interactive dark-theme report.

### Test Suites

- `test_suite_v3.json` — 110 tests across 5 categories (recommended)
- `test_suite_hard.json` — 40 harder tests
- `test_suite_extreme.json` — 40 extreme tests (HumanEval Hard, AIME-style, LeetCode Hard)

Test schema: each test has `id`, `category`, `eval_type`, `difficulty`, `prompt`, `max_tokens`, plus eval-specific fields (`expected_number`, `test_cases`, `required_args`, etc.).

### Configuration

Edit constants at the top of each script:
- `OLLAMA_HOST` — default `http://192.168.0.149:11434`
- `SKIP_MODELS` — set of model names to exclude (embedding/OCR models)

Performance tiers: EXCELLENT (≥85%), GOOD (70–84%), ADEQUATE (50–69%), MARGINAL (30–49%), POOR (<30%).

## Known Issues

- **Thinking mode 400 errors**: Models like `qwen3-coder:30b`, `llama3.2:3b`, `granite4:small-h` don't support `think: true` — they return HTTP 400 and score 0% when `--think` is used.
- **Code execution timeout**: 10-second limit per test case; exponential-complexity algorithms will fail.
- **Streaming field issues**: Some models write to unexpected response fields — use `diagnose_ollama.py` to debug.
