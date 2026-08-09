#!/usr/bin/env python3
"""
Ollama Quality Benchmark Framework
====================================
Evaluates model CORRECTNESS — not just speed — using deterministic, verifiable tests:

  Reasoning     → Numeric answer extraction + tolerance check
  Coding        → Code is actually EXECUTED; outputs compared against expected values
  Agentic       → JSON tool-call is PARSED; tool name + arguments verified
  Summarization → Required facts / key phrases checked for presence

Usage examples:
  python benchmark_quality.py --quick                              # 2 tests/cat, fast preview
  python benchmark_quality.py --num-ctx 8192                      # set context window
  python benchmark_quality.py --models deepseek-r1:32b qwen3.5:27b
  python benchmark_quality.py --categories coding agentic
  python benchmark_quality.py --suite my_tests.json               # custom test suite
  python benchmark_quality.py --list                              # list models, don't benchmark
"""

import json
import time
import requests
import re
import subprocess
import sys
import os
import csv
import argparse
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Shared Ollama client (native tools, seed, warm-up, retries).
import ollama_client
from agent_workspace import SAFE_ENV

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OLLAMA_HOST     = "http://192.168.0.149:11434"
DEFAULT_NUM_CTX = 4096   # tokens; increase for long-context tests
REQUEST_TIMEOUT = 180    # seconds per prompt

SKIP_MODELS = {
    "qwen3-embedding:8b",   # embedding model, not generative
    "deepseek-ocr:latest",  # OCR specialist
}

# ─────────────────────────────────────────────────────────────
# OLLAMA API
# ─────────────────────────────────────────────────────────────

def get_models(host: str) -> list[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=15)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])
                if m["name"] not in SKIP_MODELS]
    except Exception as e:
        print(f"❌  Cannot connect to {host}: {e}")
        return []


def call_model(
    host: str,
    model: str,
    prompt: str,
    system: str = "",
    num_ctx: int = DEFAULT_NUM_CTX,
    max_tokens: int = 600,
    think: bool = False,
    seed: int = 0,
    keep_alive: str = ollama_client.DEFAULT_KEEP_ALIVE,
) -> dict:
    """Call the shared client (native tool API, temperature 0.0, bounded retries)."""
    result = {
        "response": "", "thinking": "", "ttft": None, "tps": None,
        "total_time": None, "error": None, "done_reason": "",
    }
    messages = (
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if system else [{"role": "user", "content": prompt}]
    )
    r = ollama_client.call_with_retry(
        host, model, messages,
        num_ctx=num_ctx, num_predict=max_tokens,
        temperature=0.0, seed=seed, think=think, keep_alive=keep_alive,
    )
    result["response"] = r.content
    result["thinking"] = r.thinking
    result["ttft"] = r.ttft_seconds
    result["total_time"] = r.wall_seconds
    result["error"] = r.error
    result["done_reason"] = r.done_reason
    if r.wall_seconds > 0:
        result["tps"] = r.completion_tokens / r.wall_seconds
    return result


# ─────────────────────────────────────────────────────────────
# EVALUATOR: NUMERIC ANSWER
# ─────────────────────────────────────────────────────────────

def extract_number(text: str) -> Optional[float]:
    """
    Pull the most likely "answer" number out of free-form text.
    Anchored patterns use their FIRST match (leftmost); the fallback "any
    number" pattern takes the last number on the last non-empty line, so
    that scattered digits do not mask the stated answer.
    """
    patterns = [
        # "= 391", "answer: 391", "result is 391", "is 80 km/h"
        r'(?:=|answer\s*:?|result\s*:?|is)\s*([-+]?\d[\d,]*(?:\.\d+)?)',
        # last standalone number in the text (common for short answers)
        r'([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:km/h|mph|%|items?|days?|years?)?\s*$',
    ]
    for pat in patterns:
        hits = re.findall(pat, text.strip(), re.IGNORECASE | re.MULTILINE)
        if hits:
            try:
                return float(hits[0].replace(",", ""))
            except ValueError:
                continue
    # fallback: any number, but take the last one on the last non-empty line
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]
        hits = re.findall(r'([-+]?\d[\d,]*(?:\.\d+)?)', last_line)
        if hits:
            try:
                return float(hits[-1].replace(",", ""))
            except ValueError:
                pass
    return None


def eval_numeric(response: str, expected: float, tolerance: float = 0.01) -> dict:
    """Pass if extracted number is within relative tolerance of expected."""
    if not response:
        return {"pass": False, "reason": "Empty response", "extracted": None}

    val = extract_number(response)
    if val is None:
        return {"pass": False, "reason": "No number found in response", "extracted": None}

    if expected == 0:
        passed = abs(val) < 1e-6
    else:
        passed = abs(val - expected) / abs(expected) <= tolerance

    return {
        "pass":      passed,
        "extracted": val,
        "expected":  expected,
        "reason":    f"Got {val}, expected {expected}" if not passed else f"Correct ({val})",
    }


# ─────────────────────────────────────────────────────────────
# EVALUATOR: EXACT TEXT MATCH
# ─────────────────────────────────────────────────────────────

def eval_exact(response: str, expected: str, case_sensitive: bool = False) -> dict:
    """Pass if expected string appears anywhere in the response."""
    if not isinstance(response, str) or not isinstance(expected, str):
        return {"pass": False, "reason": "non-string input to eval_exact"}
    r = response if case_sensitive else response.lower()
    e = expected if case_sensitive else expected.lower()
    ok = e in r
    return {
        "pass":   ok,
        "reason": f"Found '{expected}'" if ok else f"Expected '{expected}' not found",
    }


# ─────────────────────────────────────────────────────────────
# EVALUATOR: CODE EXECUTION
# ─────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> Optional[str]:
    """
    Extract first Python code block from LLM response.
    Handles ```python...```, ```...```, or bare def/class.
    Also strips __main__ guards to avoid double-execution issues.
    """
    # 1. Fenced python block
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1)
    else:
        # 2. Generic fenced block
        m = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
        code = m.group(1) if m else None

    if not code:
        # 3. Look for def/class with indented body (no fences)
        m = re.search(r'^((?:def |class )\w[\s\S]+?)(?:\n\n\n|\Z)', text, re.MULTILINE)
        code = m.group(1) if m else None

    if not code:
        return None

    # Strip if __name__ == "__main__": blocks to avoid side effects
    code = re.sub(
        r'\n?if\s+__name__\s*==\s*["\']__main__["\']\s*:.*',
        '',
        code,
        flags=re.DOTALL,
    ).strip()

    return code or None


def eval_code_execution(response: str, function_name: str, test_cases: list) -> dict:
    """
    Extract function from response, run each test case in a subprocess,
    compare stdout (repr of return value) against expected.
    """
    code = extract_code_block(response)
    if not code:
        return {
            "pass": False, "score": 0.0,
            "reason": "No code block found",
            "test_results": [], "extracted_code": None,
        }

    results = []
    for tc in test_cases:
        call_expr = tc["call"]
        expected  = tc["expected"]
        # Script: inject user code, call the function, print repr of result
        script = f"{code}\n\n_result = {call_expr}\nprint(repr(_result))"

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=10,
                env=dict(SAFE_ENV),
            )
            stdout_lines = proc.stdout.strip().splitlines()
            actual_str = stdout_lines[-1] if stdout_lines else ""
            stderr     = proc.stderr.strip()

            if proc.returncode != 0:
                passed = False
                reason = (stderr.splitlines()[-1] if stderr else "Runtime error")[:120]
            else:
                expected_repr = repr(expected)
                # compare against the LAST output line so an earlier debug print
                # does not false-fail a correct solution.
                if actual_str.startswith("[") and len(stdout_lines) > 1:
                    # serialized list may span lines; join them
                    actual_str = proc.stdout.strip()
                passed = (actual_str == expected_repr)
                reason = (
                    f"Got {actual_str!r}, expected {expected_repr!r}"
                    if not passed else "✓"
                )

            results.append({
                "call": call_expr, "expected": repr(expected),
                "actual": actual_str, "pass": passed, "reason": reason,
            })

        except subprocess.TimeoutExpired:
            results.append({
                "call": call_expr, "expected": repr(expected),
                "actual": None, "pass": False, "reason": "Execution timeout (10s)",
            })
        except Exception as e:
            results.append({
                "call": call_expr, "expected": repr(expected),
                "actual": None, "pass": False, "reason": str(e),
            })

    n_pass = sum(1 for r in results if r["pass"])
    score  = n_pass / len(results) if results else 0.0

    return {
        "pass":           score == 1.0,
        "score":          score,
        "passed":         n_pass,
        "total":          len(results),
        "test_results":   results,
        "extracted_code": code[:600],
        "reason":         f"{n_pass}/{len(results)} test cases passed",
    }


# ─────────────────────────────────────────────────────────────
# EVALUATOR: TOOL CALL (AGENTIC)
# ─────────────────────────────────────────────────────────────

def parse_json_from_text(text: str) -> Optional[dict]:
    """
    Find and parse the first valid JSON object in free-form text.
    Tries fenced blocks first, then bare {...} patterns.
    """
    # 1. ```json ... ``` or ``` { ... } ```
    for m in re.finditer(r'```(?:json)?\s*\n?(\{.*?\})\s*```', text, re.DOTALL):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Outermost {...} — greedy match walking from first { to last }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None  # try next {

    return None


def normalize_args(parsed: dict) -> dict:
    """Normalize tool call JSON — different models use different key names."""
    args = (
        parsed.get("args")
        or parsed.get("arguments")
        or parsed.get("parameters")
        or parsed.get("input")
        or {}
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args


def eval_tool_call(
    response: str,
    expected_tool: str,
    expected_args: dict,
    required_args: Optional[list] = None,
) -> dict:
    """
    Parse JSON tool call from model response, then verify:
      1. Correct tool name
      2. All required argument keys present
      3. Each expected_args value is contained in the actual value
         (substring, lower/whitespace-normalised, one direction only).

    A wrong tool name is a hard fail — there is no partial credit for
    'nearly the right' call, because in a harness a wrong call is ignored.
    """
    parsed = parse_json_from_text(response)
    if parsed is None:
        return {"pass": False, "reason": "No valid JSON object found in response", "parsed": None}

    # Normalize tool name field
    tool_name = (
        parsed.get("tool")
        or parsed.get("name")
        or parsed.get("action")
        or (parsed.get("function", {}) or {}).get("name")
        or ""
    )
    tool_name = str(tool_name).strip()

    if tool_name != expected_tool:
        return {
            "pass":   False,
            "reason": f"Wrong tool: expected '{expected_tool}', got '{tool_name}'",
            "parsed": parsed,
        }

    args       = normalize_args(parsed)
    check_keys = required_args or list(expected_args.keys())

    for key in check_keys:
        if key not in args:
            return {
                "pass":   False,
                "reason": f"Missing required argument: '{key}'",
                "parsed": parsed, "args_found": args,
            }

    for key, exp_val in expected_args.items():
        got = str(args.get(key, "")).lower()
        exp = str(exp_val).lower()
        # collapse whitespace so "quantum  computing" == "quantum computing"
        got = " ".join(got.split())
        exp = " ".join(exp.split())
        # one direction only: expected contained in actual. The reverse
        # ('got in exp') is the false-pass that let 'q' pass for 'quantum'.
        if got != exp and exp not in got:
            return {
                "pass":   False,
                "reason": f"Arg '{key}': expected value containing '{exp_val}', got '{args.get(key)}'",
                "parsed": parsed, "args_found": args,
            }

    return {
        "pass":       True,
        "reason":     f"Correct: {tool_name}({', '.join(f'{k}={v!r}' for k,v in args.items())})",
        "parsed":     parsed,
        "tool_found": tool_name,
        "args_found": args,
    }


# ─────────────────────────────────────────────────────────────
# EVALUATOR: CONTAINS ALL FACTS
# ─────────────────────────────────────────────────────────────

def eval_contains_all(
    response: str,
    required_facts: list[str],
    case_sensitive: bool = False,
) -> dict:
    """Pass if ALL required facts/keywords appear in the response."""
    r = response if case_sensitive else response.lower()
    missing = [f for f in required_facts
               if (f if case_sensitive else f.lower()) not in r]
    n_found = len(required_facts) - len(missing)
    ok      = len(missing) == 0
    return {
        "pass":    ok,
        "score":   n_found / len(required_facts) if required_facts else 1.0,
        "found":   n_found,
        "total":   len(required_facts),
        "missing": missing,
        "reason":  (
            f"All {n_found} facts found" if ok
            else f"{n_found}/{len(required_facts)} found, missing: {missing}"
        ),
    }


# ─────────────────────────────────────────────────────────────
# EVALUATOR: REGEX
# ─────────────────────────────────────────────────────────────

def eval_regex(response: str, pattern: str) -> dict:
    """Pass if the response matches the regex pattern."""
    try:
        m = re.search(pattern, response.strip(), re.IGNORECASE | re.MULTILINE)
        return {
            "pass":   bool(m),
            "match":  m.group(0) if m else None,
            "reason": f"Pattern matched: '{m.group(0)}'" if m else f"Pattern not found: {pattern!r}",
        }
    except re.error as e:
        return {"pass": False, "reason": f"Invalid regex: {e}"}


# ─────────────────────────────────────────────────────────────
# EVALUATOR DISPATCHER
# ─────────────────────────────────────────────────────────────

def evaluate(response: str, test: dict) -> dict:
    etype = test.get("eval_type", "")
    if etype == "numeric":
        return eval_numeric(
            response,
            float(test["expected_number"]),
            float(test.get("tolerance", 0.01)),
        )
    elif etype == "exact":
        return eval_exact(response, test["expected_text"],
                          bool(test.get("case_sensitive", False)))
    elif etype == "code_execution":
        return eval_code_execution(
            response,
            test.get("function_name", ""),
            test.get("test_cases", []),
        )
    elif etype == "tool_call":
        return eval_tool_call(
            response,
            test["expected_tool"],
            test.get("expected_args", {}),
            test.get("required_args"),
        )
    elif etype == "contains_all":
        return eval_contains_all(
            response,
            test.get("required_facts", []),
            bool(test.get("case_sensitive", False)),
        )
    elif etype == "regex":
        return eval_regex(response, test["pattern"])
    else:
        return {"pass": False, "reason": f"Unknown eval_type: '{etype}'"}


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────

_TIERS = [
    (85, "🟢 EXCELLENT", "#3fb950", "Keep as primary model"),
    (70, "🔵 GOOD",      "#58a6ff", "Keep for specific tasks"),
    (50, "🟡 ADEQUATE",  "#d29922", "Keep only if unique capability"),
    (30, "🟠 MARGINAL",  "#f0883e", "Consider removing"),
    ( 0, "🔴 POOR",      "#f85149", "Remove / replace"),
]

def classify(pass_rate: float) -> tuple[str, str, str]:
    """Return (label, hex_color, recommendation)."""
    for threshold, label, color, rec in _TIERS:
        if pass_rate >= threshold:
            return label, color, rec
    return _TIERS[-1][1], _TIERS[-1][2], _TIERS[-1][3]


# ─────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────

def run_benchmark(
    host: str,
    models: list[str],
    tests: list[dict],
    num_ctx: int,
    think: bool = False,
    seed: int = 42,
    keep_alive: str = ollama_client.DEFAULT_KEEP_ALIVE,
) -> dict:
    all_results = {}
    total = len(models) * len(tests)
    idx   = 0

    think_label = "ON (capability-gated)" if think else "OFF (direct answer mode)"
    print(f"\n{'='*72}")
    print(f"  Quality Benchmark  |  {len(models)} model(s)  |  {len(tests)} tests")
    print(f"  num_ctx = {num_ctx}  |  thinking = {think_label}  |  seed = {seed}")
    print(f"  Server: {host}")
    print(f"{'='*72}")

    for model in models:
        print(f"\n📦  {model}")
        # warm up once so the first real test isn't dominated by load time
        ollama_client.warmup(host, model)
        if think:
            try:
                prof = ollama_client.probe_model(host, model)
                model_think = "thinking" in (prof.capabilities or [])
            except Exception:
                model_think = False
            if not model_think:
                print(f"    (thinking requested but unsupported → sending think=False)")
        else:
            model_think = False
        all_results[model] = {"tests": [], "categories": {}, "overall": {}}

        for test in tests:
            idx += 1
            t_id  = test.get("id", "?")
            name  = test.get("name", t_id)
            cat   = test.get("category", "other")
            etype = test.get("eval_type", "?")

            print(f"  [{idx:>3}/{total}] {cat}/{name:<38}", end="", flush=True)

            resp = call_model(
                host, model,
                test["prompt"],
                system    = test.get("system", ""),
                num_ctx   = num_ctx,
                max_tokens= test.get("max_tokens", 600),
                think     = model_think,
                seed      = seed,
                keep_alive= keep_alive,
            )

            if resp["error"]:
                ev = {"pass": False, "reason": f"API error: {resp['error']}",
                      "score": 0.0}
                print(f"  ❌  {resp['error'][:60]}")
            else:
                eval_text = resp["response"]
                if not eval_text and resp.get("thinking"):
                    # model hid its answer in thinking only → extraction failure
                    ev = {"pass": False,
                          "reason": "answer only in thinking field (extraction failed)",
                          "score": 0.0}
                else:
                    ev = evaluate(eval_text, test)
                if ev["pass"] and resp.get("done_reason") in ("length", "max_tokens"):
                    # passed but was cut off at num_predict → not trustworthy
                    ev = {"pass": False,
                          "reason": "truncated at num_predict (pass reset to fail)",
                          "score": 0.0}
                icon = "✅" if ev["pass"] else "❌"
                tps  = f"{resp['tps']:.1f}t/s" if resp["tps"] else "N/A   "
                reason = ev.get("reason", "")[:55]
                print(f"  {icon}  {tps:<9} {reason}")

            score = ev.get("score", 1.0 if ev.get("pass") else 0.0)

            all_results[model]["tests"].append({
                "id":          t_id,
                "name":        name,
                "category":    cat,
                "eval_type":   etype,
                "difficulty":  test.get("difficulty", ""),
                "pass":        bool(ev.get("pass")),
                "score":       score,
                "eval_detail": ev,
                "response":    resp["response"][:800],
                "tps":         resp["tps"],
                "ttft":        resp["ttft"],
                "done_reason": resp.get("done_reason"),
                "thinking":    bool(think) and bool(resp.get("thinking")),
            })

        # ── Aggregate by category ──
        cats: dict = {}
        for t in all_results[model]["tests"]:
            c = t["category"]
            cats.setdefault(c, {"passed": 0, "total": 0, "scores": []})
            cats[c]["total"]  += 1
            cats[c]["scores"].append(t["score"])
            if t["pass"]:
                cats[c]["passed"] += 1

        for c, d in cats.items():
            d["pass_rate"] = d["passed"] / d["total"] * 100
            d["avg_score"] = statistics.mean(d["scores"]) * 100

        all_results[model]["categories"] = cats

        # ── Overall ──
        ts_all  = all_results[model]["tests"]
        n_pass  = sum(1 for t in ts_all if t["pass"])
        scores  = [t["score"] for t in ts_all]
        tps_all = [t["tps"] for t in ts_all if t["tps"]]

        overall_rate = n_pass / len(ts_all) * 100 if ts_all else 0
        label, color, rec = classify(overall_rate)

        all_results[model]["overall"] = {
            "passed":    n_pass,
            "total":     len(ts_all),
            "pass_rate": round(overall_rate, 2),
            "avg_score": round(statistics.mean(scores) * 100, 2) if scores else 0,
            "avg_tps":   round(statistics.mean(tps_all), 2) if tps_all else 0,
            "tier":      label,
        }

        print(f"  → Result: {n_pass}/{len(ts_all)} passed "
              f"({overall_rate:.1f}%)  {label}")

    return all_results


# ─────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────

def console_summary(results: dict, categories: list[str]):
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["overall"]["pass_rate"],
        reverse=True,
    )
    cats = categories

    def _cat_label(cat: str, width: int = 8) -> str:
        """Readable column label: strip shared prefix, abbreviate, uppercase acronyms."""
        part = cat.split("_")[-1]
        _map = {
            "mcp":           "MCP",
            "coding":        "Coding",
            "agentic":       "Agentic",
            "reasoning":     "Reason.",
            "summarization": "Summar.",
        }
        label = _map.get(part, part.upper() if len(part) <= 3 else part[:width].capitalize())
        return f"{label:>{width}}"

    cat_hdr = "".join(f"  {_cat_label(c)}" for c in cats)
    print(f"\n{'='*108}")
    print("  QUALITY BENCHMARK — FINAL RANKINGS  (% tests PASSED)")
    print(f"{'='*108}")
    print(f"{'#':<4} {'Model':<48} {'Tier':<12} {'Overall':>8} {'TPS':>6}{cat_hdr}")
    print("─" * 108)

    for rank, (model, data) in enumerate(ranked, 1):
        o     = data["overall"]
        short = model if len(model) <= 47 else model[:44] + "..."
        tier  = o["tier"].split()[-1]
        cat_cells = "".join(
            f"  {data['categories'].get(c, {}).get('pass_rate', 0):>6.0f}%"
            for c in cats
        )
        print(f"{rank:<4} {short:<48} {tier:<12} "
              f"{o['pass_rate']:>7.1f}%  {o['avg_tps']:>5.1f}{cat_cells}")

    print(f"\n{'─'*60}")
    print("  RECOMMENDATIONS")

    tier_buckets: dict[str, list] = defaultdict(list)
    for model, data in ranked:
        t_short = data["overall"]["tier"].split()[-1]
        tier_buckets[t_short].append((model, data["overall"]["pass_rate"]))

    advice = {
        "EXCELLENT": ("🟢", "Keep as primary model"),
        "GOOD":      ("🔵", "Keep for specific tasks"),
        "ADEQUATE":  ("🟡", "Keep only if unique capability"),
        "MARGINAL":  ("🟠", "Candidate for removal"),
        "POOR":      ("🔴", "Remove / replace"),
    }
    for tier_short in ["EXCELLENT", "GOOD", "ADEQUATE", "MARGINAL", "POOR"]:
        entries = tier_buckets.get(tier_short, [])
        if entries:
            dot, rec = advice[tier_short]
            print(f"\n  {dot} {tier_short} — {rec}:")
            for m, rate in entries:
                print(f"      • {m}  ({rate:.1f}%)")


# ─────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────

def save_html_report(results: dict, categories: list[str], num_ctx: int, path: str,
                     think: bool = False):
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["overall"]["pass_rate"],
        reverse=True,
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Summary table ──
    def _html_cat_label(cat: str) -> str:
        part = cat.split("_")[-1]
        _map = {"mcp": "MCP", "coding": "Coding", "agentic": "Agentic",
                "reasoning": "Reasoning", "summarization": "Summarization"}
        return _map.get(part, part.upper() if len(part) <= 3 else part.replace("_", " ").replace("_", " ").title())
    cat_th   = "".join(f"<th>{_html_cat_label(c)}</th>" for c in categories)
    rows_html = ""
    for rank, (model, data) in enumerate(ranked, 1):
        o     = data["overall"]
        label, color, _ = classify(o["pass_rate"])
        anchor  = re.sub(r'[^a-zA-Z0-9]', '_', model)
        cat_cells = "".join(
            f"<td>{data['categories'].get(c, {}).get('pass_rate', 0):.0f}%</td>"
            for c in categories
        )
        rows_html += (
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td><a href='#{anchor}'><strong>{model}</strong></a></td>"
            f"<td style='color:{color};font-weight:bold'>{label}</td>"
            f"<td><strong>{o['pass_rate']:.1f}%</strong>"
            f" <small>({o['passed']}/{o['total']})</small></td>"
            f"<td>{o['avg_tps']:.1f}</td>"
            f"{cat_cells}"
            f"</tr>\n"
        )

    # ── Per-model detail sections ──
    detail_html = ""
    for model, data in ranked:
        anchor  = re.sub(r'[^a-zA-Z0-9]', '_', model)
        o       = data["overall"]
        label, color, rec = classify(o["pass_rate"])

        test_rows = ""
        for t in data["tests"]:
            icon  = "✅" if t["pass"] else "❌"
            score = f"{t['score']*100:.0f}%"
            ev    = t["eval_detail"]
            reason = (ev.get("reason") or "")[:100]
            resp_esc = (t.get("response") or "")[:400].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            extra = ""

            # Code execution: show per-test-case table
            if t["eval_type"] == "code_execution":
                tc_rows = ""
                for r in ev.get("test_results", []):
                    ri = "✅" if r["pass"] else "❌"
                    tc_rows += (
                        f"<tr>"
                        f"<td><code>{r.get('call','')}</code></td>"
                        f"<td><code>{r.get('expected','')}</code></td>"
                        f"<td><code>{r.get('actual') or ''}</code></td>"
                        f"<td>{ri} {r.get('reason','')[:60]}</td>"
                        f"</tr>"
                    )
                code_esc = (ev.get("extracted_code") or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                extra = (
                    f"<details><summary>Test cases &amp; extracted code</summary>"
                    f"<table style='font-size:0.8em'>"
                    f"<tr><th>Call</th><th>Expected</th><th>Got</th><th>Result</th></tr>"
                    f"{tc_rows}</table>"
                    f"<pre style='margin-top:8px'>{code_esc}</pre>"
                    f"</details>"
                )

            # Tool call: show parsed JSON
            elif t["eval_type"] == "tool_call":
                parsed = ev.get("parsed")
                if parsed:
                    p_esc = json.dumps(parsed, indent=2).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    extra = f"<details><summary>Parsed JSON</summary><pre>{p_esc}</pre></details>"

            test_rows += (
                f"<tr>"
                f"<td style='font-size:1.2em'>{icon}</td>"
                f"<td><small style='color:#8b949e'>{t['category']}</small></td>"
                f"<td><strong>{t['name']}</strong>"
                f"  <small style='color:#8b949e'>[{t['eval_type']}] [{t.get('difficulty','')}]</small><br>"
                f"  {reason}<br>{extra}</td>"
                f"<td style='text-align:center'>{score}</td>"
                f"<td><details><summary style='color:#58a6ff;cursor:pointer'>show</summary>"
                f"<pre style='font-size:0.75em;white-space:pre-wrap'>{resp_esc}</pre>"
                f"</details></td>"
                f"</tr>\n"
            )

        detail_html += (
            f"<h2 id='{anchor}'>{model}"
            f"  <span style='color:{color};font-size:0.7em;margin-left:10px'>{label}</span>"
            f"  <span style='color:#8b949e;font-size:0.65em;margin-left:10px'>"
            f"{o['pass_rate']:.1f}% ({o['passed']}/{o['total']}) — {rec}</span></h2>\n"
            f"<table>"
            f"<tr><th>✓</th><th>Cat</th><th>Test &amp; Detail</th>"
            f"<th>Score</th><th>Response</th></tr>\n"
            f"{test_rows}"
            f"</table>\n"
        )

    # ── Tier legend ──
    tier_rows = ""
    for thr, label, color, rec in _TIERS:
        tier_rows += (
            f"<tr>"
            f"<td style='color:{color};font-weight:bold'>{label}</td>"
            f"<td>≥ {thr}% pass rate</td>"
            f"<td>{rec}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quality Benchmark – {now}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:#0d1117; color:#e6edf3; padding:32px 24px; line-height:1.5 }}
  h1   {{ color:#58a6ff; font-size:1.6em; margin-bottom:4px }}
  h2   {{ color:#79c0ff; font-size:1.05em; margin:36px 0 10px;
         border-bottom:1px solid #30363d; padding-bottom:6px }}
  p.meta {{ color:#8b949e; font-size:0.85em; margin-bottom:20px }}
  a {{ color:#58a6ff; text-decoration:none }} a:hover {{ text-decoration:underline }}
  table {{ width:100%; border-collapse:collapse; font-size:0.87em; margin:8px 0 }}
  th {{ background:#161b22; color:#79c0ff; padding:9px 10px; text-align:left;
       border:1px solid #30363d; white-space:nowrap }}
  td {{ padding:8px 10px; border:1px solid #30363d; vertical-align:top }}
  tr:nth-child(even) {{ background:#0d1117 }}
  tr:nth-child(odd)  {{ background:#111318 }}
  tr:hover {{ background:#1c2128 }}
  pre {{ background:#161b22; padding:8px; border-radius:4px;
        overflow-x:auto; font-size:0.82em; white-space:pre-wrap }}
  code {{ background:#161b22; padding:1px 5px; border-radius:3px; font-size:0.88em }}
  details summary {{ cursor:pointer; color:#58a6ff; user-select:none }}
  .tip {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
          padding:14px 18px; margin:16px 0; font-size:0.88em; color:#8b949e }}
  .tip strong {{ color:#e6edf3 }}
</style>
</head>
<body>
<h1>🎯 Ollama Quality Benchmark</h1>
<p class="meta">
  Generated: {now} &nbsp;|&nbsp;
  Server: {OLLAMA_HOST} &nbsp;|&nbsp;
  Context window: <strong>{num_ctx} tokens</strong> &nbsp;|&nbsp;
  Thinking mode: <strong>{'ON' if think else 'OFF (recommended)'}</strong> &nbsp;|&nbsp;
  Models tested: {len(results)}
</p>

<div class="tip">
  <strong>Scoring is deterministic and correctness-based:</strong><br>
  <strong>Reasoning</strong> — Numeric answer extracted from response and compared with tolerance.<br>
  <strong>Coding</strong> — Function extracted from response, <em>actually executed</em> in Python, outputs compared against expected values.<br>
  <strong>Agentic</strong> — JSON tool call parsed from response, tool name + required arguments verified.<br>
  <strong>Summarization</strong> — Required key facts / format patterns checked for presence.<br>
  Temperature = 0.05. Each test is binary pass/fail.
</div>

<h2>Rankings</h2>
<table>
<tr><th>#</th><th>Model</th><th>Tier</th><th>Overall Pass Rate ↓</th><th>TPS</th>{cat_th}</tr>
{rows_html}
</table>

<h2>Classification</h2>
<table>
<tr><th>Tier</th><th>Threshold</th><th>Recommendation</th></tr>
{tier_rows}
</table>

<h2>Per-Model Detail</h2>
{detail_html}
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📊  HTML report  → {path}")


# ─────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────

def save_csv(results: dict, categories: list[str], path: str):
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["overall"]["pass_rate"],
        reverse=True,
    )
    fieldnames = (
        ["rank", "model", "tier", "pass_rate", "passed", "total", "avg_tps"]
        + [f"{c}_pass_rate" for c in categories]
    )
    rows = []
    for rank, (model, data) in enumerate(ranked, 1):
        o = data["overall"]
        label, _, _ = classify(o["pass_rate"])
        row = {
            "rank":      rank,
            "model":     model,
            "tier":      label.split()[-1],
            "pass_rate": o["pass_rate"],
            "passed":    o["passed"],
            "total":     o["total"],
            "avg_tps":   o["avg_tps"],
        }
        for c in categories:
            row[f"{c}_pass_rate"] = round(
                data["categories"].get(c, {}).get("pass_rate", 0), 1
            )
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"📈  CSV results  → {path}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ollama Quality Benchmark — deterministic correctness evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host",       default=OLLAMA_HOST,
                        help=f"Ollama base URL (default: {OLLAMA_HOST})")
    parser.add_argument("--models",     nargs="+", metavar="MODEL",
                        help="Specific model names to test (default: all)")
    parser.add_argument("--categories", nargs="+",
                        metavar="CAT", help="Limit to specific categories (e.g. reasoning coding agentic summarization mcp)")
    parser.add_argument("--num-ctx",    type=int, default=DEFAULT_NUM_CTX,
                        dest="num_ctx",
                        help=f"Context window tokens passed to Ollama (default: {DEFAULT_NUM_CTX}). "
                             "Try 2048/4096/8192/16384 to test context sensitivity.")
    parser.add_argument("--suite",      default="test_suite_v3.json",
                        metavar="PATH",
                        help="Path to a test suite JSON file (default: test_suite_v3.json, the 110-test suite)")
    parser.add_argument("--quick",      action="store_true",
                        help="Quick mode: run only 2 tests per category (~8 tests total)")
    parser.add_argument("--seed",       type=int, default=42, metavar="N",
                        help="Sampling seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", default=".", metavar="DIR",
                        help="Directory for output files (default: current dir)")
    parser.add_argument("--list",       action="store_true",
                        help="List available models on the server and exit")
    parser.add_argument("--think",      action="store_true", default=False,
                        help="Enable thinking mode only for models that support it "
                             "(capability-checked; disabled for those that would 400).")
    args = parser.parse_args()

    # ── Load tests ──
    suite_path = args.suite
    with open(suite_path, encoding="utf-8") as f:
        suite = json.load(f)
    tests = suite["tests"]
    print(f"📋  Loaded {len(tests)} tests from {suite_path}")

    if args.categories:
        tests = [t for t in tests if t.get("category") in args.categories]
        print(f"🔎  Filtered to categories: {args.categories} → {len(tests)} tests")

    # ── Quick mode: 2 tests per category ──
    if args.quick:
        cat_counts: dict[str, int] = defaultdict(int)
        filtered = []
        for t in tests:
            c = t.get("category", "")
            if cat_counts[c] < 2:
                filtered.append(t)
                cat_counts[c] += 1
        tests = filtered
        print(f"⚡  Quick mode — {len(tests)} tests total")

    if not tests:
        print("No tests selected.")
        sys.exit(1)

    # ── Get models ──
    available = get_models(args.host)
    if not available:
        sys.exit(1)

    if args.list:
        print(f"\nModels on {args.host}  ({len(available)} total):")
        for m in available:
            print(f"  • {m}")
        return

    models = args.models or available
    unknown = [m for m in models if m not in available]
    if unknown:
        print(f"⚠️  Unknown models (skipping): {unknown}")
    models = [m for m in models if m in available]
    if not models:
        print("No valid models to benchmark.")
        sys.exit(1)

    # ── Categories present in selected tests ──
    categories = list(dict.fromkeys(t.get("category", "other") for t in tests))

    # ── Run ──
    results = run_benchmark(args.host, models, tests, args.num_ctx,
                            think=args.think, seed=args.seed)

    # ── Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx_tag = f"ctx{args.num_ctx}_{'think' if args.think else 'nothink'}"

    json_path = os.path.join(args.output_dir, f"quality_{ctx_tag}_{ts}.json")
    html_path = os.path.join(args.output_dir, f"quality_{ctx_tag}_{ts}.html")
    csv_path  = os.path.join(args.output_dir, f"quality_{ctx_tag}_{ts}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾  Raw JSON      → {json_path}")

    save_html_report(results, categories, args.num_ctx, html_path, think=args.think)
    save_csv(results, categories, csv_path)
    console_summary(results, categories)


if __name__ == "__main__":
    main()
