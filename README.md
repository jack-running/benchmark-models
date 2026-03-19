# Ollama Model Benchmark Suite

A correctness-first benchmarking framework for [Ollama](https://ollama.com) language models. Unlike pure speed benchmarks, this suite evaluates whether models actually answer questions correctly — executing code, parsing tool calls, checking numeric precision, and verifying facts — across four capability domains.

Designed for local Ollama deployments (tested on 24 GB VRAM / 64 GB RAM hardware running ~34 models simultaneously).

---

## Files at a Glance

| File | Purpose |
|---|---|
| `benchmark_quality.py` | Main benchmark: correctness-based evaluation across all categories |
| `benchmark_ollama.py` | Speed + heuristic quality benchmark (TTFT, TPS, keyword scoring) |
| `diagnose_ollama.py` | Debug tool: dumps raw Ollama API chunk fields to diagnose streaming issues |
| `test_suite_v3.json` | Standard test suite — 110 tests across 5 categories |
| `test_suite_hard.json` | Hard test suite — 40 tests, DP algorithms, Bayesian math, complex tool calls |
| `test_suite_extreme.json` | Extreme test suite — 40 tests drawn from HumanEval hard, AIME-style math, LeetCode Hard |
| `test_suite_extended.json` | Earlier 40-test suite (superseded by v3) |
| `quality_ctx*_nothink_*.{json,csv,html}` | Benchmark run results with thinking mode OFF |
| `quality_ctx*_think_*.{json,csv,html}` | Benchmark run results with thinking mode ON |

---

## Quick Start

### Requirements

- Python 3.10+
- `pip install requests`
- An Ollama server running and accessible (default: `http://192.168.0.149:11434`)

### Run the quality benchmark (recommended)

```bash
# Benchmark all models using the built-in test suite
python benchmark_quality.py

# Use a custom test suite file
python benchmark_quality.py --suite test_suite_v3.json

# Test specific models only
python benchmark_quality.py --models llama3.2:3b qwen3-coder:30b --suite test_suite_v3.json

# Run only certain categories
python benchmark_quality.py --suite test_suite_v3.json --categories coding agentic

# Set context window (important for MCP/agentic tests with long system prompts)
python benchmark_quality.py --suite test_suite_extreme.json --num-ctx 16384

# Enable chain-of-thought reasoning mode
python benchmark_quality.py --suite test_suite_extreme.json --num-ctx 16384 --think

# Quick preview (2 tests per category)
python benchmark_quality.py --quick

# List available models without running any tests
python benchmark_quality.py --list

# Export the built-in test suite to a JSON file
python benchmark_quality.py --save-suite
```

### Run the speed benchmark

```bash
python benchmark_ollama.py
python benchmark_ollama.py --quick
python benchmark_ollama.py --models qwen3.5:9b deepseek-r1:32b
python benchmark_ollama.py --host http://localhost:11434
```

### Diagnose a failing model

```bash
python diagnose_ollama.py
python diagnose_ollama.py --model qwen3.5:27b --prompt "What is 17*23?"
```

---

## benchmark_quality.py

The primary benchmark. Evaluates models on correctness using deterministic, machine-verifiable methods — not heuristics or keyword matching.

### Output

Each run produces three files named by context window size, thinking mode, and timestamp:

```
quality_ctx16384_nothink_20260316_105636.csv   — ranking table (one row per model)
quality_ctx16384_nothink_20260316_105636.json  — full per-test results with reasons
quality_ctx16384_nothink_20260316_105636.html  — interactive HTML report
```

### Evaluation Methods

| `eval_type` | How it works | Used for |
|---|---|---|
| `numeric` | Extracts a number from the response; checks it is within `tolerance` of `expected_number` | Math and reasoning questions |
| `code_execution` | Extracts the code block from the response; runs each `test_case` call in a subprocess; compares `repr()` of the return value | Coding problems |
| `tool_call` | Parses the first JSON object in the response; checks `tool` name matches `expected_tool`; checks all `required_args` keys are present; checks each `expected_args` value appears (substring match) | Agentic and MCP tool-calling |
| `contains_all` | Checks that all strings in `required_facts` appear in the response (case-insensitive) | Summarization and factual recall |
| `exact` | Exact string match (case-insensitive by default) | Short factual answers |
| `regex` | Response must match a regular expression pattern | Structured output verification |

### Performance Tiers

| Tier | Pass Rate |
|---|---|
| EXCELLENT | ≥ 80% |
| GOOD | ≥ 60% |
| FAIR | ≥ 40% |
| POOR | < 40% |

### Thinking Mode (`--think`)

Ollama ≥ 0.7 and models like Qwen3, DeepSeek-R1, and GLM-4 support a reasoning mode where the model "thinks" before answering. Tokens generated during thinking appear in the `thinking` field of the API response, not `response`.

Key considerations:

- **Not all models support it.** Models that don't support `think: true` return HTTP 400, causing all their tests to fail. Check the result JSON's `reason` field for `API error: 400` to identify incompatible models.
- **Thinking consumes token budget.** The `num_predict` limit applies to thinking tokens + response tokens combined. The extreme suite uses `max_tokens` of 600–1400 to ensure enough headroom.
- **Thinking helps reasoning, can hurt coding/tool calls.** With thinking ON, models may produce verbose output that breaks the code extractor or JSON parser.
- **Default is OFF** (`--nothink`) for consistent, comparable results across models.

### Configuring the Ollama Host

Edit the `OLLAMA_HOST` constant at the top of the script:

```python
OLLAMA_HOST = "http://192.168.0.149:11434"
```

Or pass it at runtime:

```bash
python benchmark_quality.py --host http://localhost:11434
```

---

## benchmark_ollama.py

An earlier, speed-focused benchmark. Measures Time-to-First-Token (TTFT) and Tokens-per-Second (TPS), and assigns a heuristic quality score (0–100) based on keyword presence in responses.

Useful for a quick comparison of inference speed across models. For correctness evaluation, use `benchmark_quality.py` instead.

---

## diagnose_ollama.py

A debugging utility that dumps raw Ollama streaming API chunks to stdout. Use this when a model returns empty responses or behaves unexpectedly.

It runs three tests automatically:
1. Default payload (no explicit `think` setting)
2. Explicit `think: false`
3. Explicit `think: false` with a generous token budget

This helps identify whether a model is streaming its output to the `thinking` field instead of `response` — a common issue with Ollama ≥ 0.7 reasoning models.

---

## Test Suites

Test suites are JSON files with a `tests` array. Each test is a dictionary with fields that depend on its `eval_type`. All tests require: `id`, `name`, `category`, `eval_type`, `difficulty`, `prompt`, `max_tokens`.

### test_suite_v3.json — Standard (110 tests)

The main general-purpose suite. Recommended starting point.

| Category | Tests | Eval Types |
|---|---|---|
| `reasoning` | 25 | numeric, contains_all |
| `coding` | 25 | code_execution |
| `agentic` | 25 | tool_call |
| `summarization` | 25 | contains_all, regex |
| `mcp` | 10 | tool_call |

**Recommended:** `--num-ctx 8192`

```bash
python benchmark_quality.py --suite test_suite_v3.json --num-ctx 8192
```

### test_suite_hard.json — Hard (40 tests)

Harder versions of each category. Includes DP algorithms (LCS, edit distance, coin change), Bayesian probability, system-of-equations reasoning, and MCP tool schemas with nested parameters.

| Category | Tests | Highlights |
|---|---|---|
| `coding` | 10 | LCS, edit distance, rain water trapping, coin change, word break |
| `reasoning` | 10 | C(10,2), conditional probability, modular arithmetic, compound interest |
| `agentic` | 10 | 8-tool palette including `write_file`, `create_event`, `get_user_profile` |
| `mcp` | 10 | Nested dict params, boolean + datetime + integer required together, tool chaining |

**Recommended:** `--num-ctx 16384`

### test_suite_extreme.json — Extreme (40 tests)

Designed to break 100% pass rates on capable 27B models. Draws from HumanEval hard problems, LeetCode Hard, AIME-style competition math, and BFCL-style function calling.

| Category | Tests | Highlights |
|---|---|---|
| `extreme_coding` | 10 | Word Ladder BFS, N-Queens (must return 92 for n=8), Minimum Window Substring, LIS O(n log n), expression evaluator with parentheses, Hierholzer's itinerary reconstruction, max product subarray |
| `extreme_reasoning` | 10 | 7^(7^7) mod 100 = 43, right triangle area from inradius, Bayesian factory defect, Σk/2^k = 2, divisor count from prime factorization |
| `extreme_agentic` | 10 | 10-tool palette including `post_webhook` and `translate_text`; tests disambiguation between superficially similar tools |
| `extreme_mcp` | 10 | Nested `date_range` objects, `idempotency_key` required param, `compress_file` vs `resize_image` disambiguation, pagination offset |

**Recommended:** `--num-ctx 16384` (or `--num-ctx 48000` for thinking mode)

```bash
# Without thinking (faster, good for most models)
python benchmark_quality.py --suite test_suite_extreme.json --num-ctx 16384

# With thinking (better reasoning scores, but only for models that support it)
python benchmark_quality.py --suite test_suite_extreme.json --num-ctx 48000 --think
```

---

## Writing Custom Test Suites

Create a JSON file with the following structure:

```json
{
  "version": "1.0",
  "description": "My custom tests",
  "tests": [
    {
      "id": "my_r01",
      "name": "descriptive_name",
      "category": "reasoning",
      "eval_type": "numeric",
      "difficulty": "medium",
      "prompt": "What is 15% of 240? Give only the number.",
      "expected_number": 36,
      "tolerance": 0.5,
      "max_tokens": 20
    },
    {
      "id": "my_c01",
      "name": "reverse_string",
      "category": "coding",
      "eval_type": "code_execution",
      "difficulty": "easy",
      "prompt": "Write a Python function `reverse_str(s)` that returns the reverse of string s. Output only the function definition.",
      "function_name": "reverse_str",
      "test_cases": [
        {"call": "reverse_str('hello')", "expected": "olleh"},
        {"call": "reverse_str('')",      "expected": ""}
      ],
      "max_tokens": 200
    },
    {
      "id": "my_a01",
      "name": "search_not_profile",
      "category": "agentic",
      "eval_type": "tool_call",
      "difficulty": "hard",
      "system": "You are an AI agent. Output ONLY a JSON object: {\"tool\": \"<name>\", \"args\": {<key>: <value>}}\n\nAvailable tools:\n- search_web(query: str)\n- get_user_profile(user_id: str)",
      "prompt": "Find the latest news about quantum computing.",
      "expected_tool": "search_web",
      "expected_args": {"query": "quantum computing"},
      "required_args": ["query"],
      "max_tokens": 150
    },
    {
      "id": "my_s01",
      "name": "summarization_facts",
      "category": "summarization",
      "eval_type": "contains_all",
      "difficulty": "easy",
      "prompt": "Summarize: The Eiffel Tower was built in 1889 for the World's Fair in Paris. It stands 330 metres tall.",
      "required_facts": ["1889", "paris", "330"],
      "max_tokens": 100
    }
  ]
}
```

### Test Schema Reference

#### `numeric`
```json
{
  "eval_type": "numeric",
  "expected_number": 45,
  "tolerance": 0,
  "max_tokens": 30
}
```
`tolerance: 0` requires an exact match. Use small values like `0.001` for floating-point answers.

#### `code_execution`
```json
{
  "eval_type": "code_execution",
  "function_name": "my_func",
  "test_cases": [
    {"call": "my_func(arg1, arg2)", "expected": <python_value>}
  ],
  "max_tokens": 500
}
```
The model must output a code block containing only the function definition. The evaluator injects it into a subprocess and compares `repr(result)` against `repr(expected)`.

#### `tool_call`
```json
{
  "eval_type": "tool_call",
  "system": "<system prompt describing available tools>",
  "expected_tool": "tool_name",
  "expected_args": {"arg1": "expected_value"},
  "required_args": ["arg1", "arg2"],
  "max_tokens": 200
}
```
`required_args` is a list of argument names that must be present. `expected_args` is a dict of argument names to expected values; matching is done via case-insensitive substring check (either direction).

#### `contains_all`
```json
{
  "eval_type": "contains_all",
  "required_facts": ["fact1", "fact2"],
  "max_tokens": 200
}
```

#### `exact`
```json
{
  "eval_type": "exact",
  "expected_output": "exact answer",
  "max_tokens": 20
}
```

#### `regex`
```json
{
  "eval_type": "regex",
  "pattern": "\\d{4}-\\d{2}-\\d{2}",
  "max_tokens": 50
}
```

---

## Known Issues and Notes

**Thinking mode and 400 errors.** Models that do not support the `think` API parameter return HTTP 400 for every prompt, resulting in 0% scores when running with `--think`. This is a model compatibility issue, not a capability failure. Affected models in testing: `qwen3-coder:30b`, `llama3.2:3b`, `granite4:small-h`. Identify them by looking for `"API error: 400"` in the result JSON's `reason` field.

**Code execution timeout.** Each test case has a 10-second subprocess timeout. Algorithms with exponential complexity (e.g. naive N-Queens for large n) will time out. The tests in this suite are sized to complete within that window.

**Embedding and OCR models.** Models listed in `SKIP_MODELS` are excluded automatically. Add model names to this set in the script if you have additional non-generative models.

**Context window and MCP tests.** The MCP system prompts in v3 and the extreme suite include full JSON tool schemas, consuming 400–700 tokens before the model sees the actual question. Use `--num-ctx 8192` or higher for MCP category tests.

**Temperature.** All requests use `temperature: 0.05` for near-deterministic, reproducible results. This is intentional — benchmarking at higher temperatures adds noise without improving comparability.

---

## Result File Format

### JSON (`quality_ctx*_*.json`)

```json
{
  "model_name": {
    "overall": {
      "pass_rate": 75.0,
      "tier": "GOOD",
      "avg_tps": 98.7,
      "total_passed": 30,
      "total_tests": 40
    },
    "categories": {
      "extreme_coding": {"pass_rate": 80.0, "passed": 8, "total": 10},
      "extreme_reasoning": {"pass_rate": 20.0, "passed": 2, "total": 10}
    },
    "tests": [
      {
        "id": "e_c01",
        "name": "is_nested_brackets",
        "category": "extreme_coding",
        "pass": true,
        "eval_detail": {"pass": true, "reason": "3/3 test cases passed"},
        "response": "def is_nested(s):\n    ...",
        "tps": 98.0,
        "ttft": 0.12
      }
    ]
  }
}
```

### CSV (`quality_ctx*_*.csv`)

One row per model with columns: `rank`, `model`, `tier`, `overall_pct`, `avg_tps`, and one column per category.

---

## Hardware and Environment

The benchmark was developed and tested against an Ollama instance at `192.168.0.149:11434` running on a machine with:

- 24 GB VRAM
- 64 GB RAM
- ~34 models loaded simultaneously

Update `OLLAMA_HOST` in each script to point to your own Ollama server.
