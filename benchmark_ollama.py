#!/usr/bin/env python3
"""
Ollama Model Speed Benchmark
=============================
Measures: Time-to-First-Token (TTFT), Tokens/sec (TPS), and end-to-end latency.
Heuristic quality scoring was REMOVED — correctness is assessed by
benchmark_quality.py (correctness) and benchmark_agent.py (agentic harness).

Usage:
  python benchmark_ollama.py                         # Benchmark all models
  python benchmark_ollama.py --quick                 # 1 test/category (fast preview)
  python benchmark_ollama.py --models qwen3.5:9b deepseek-r1:32b
  python benchmark_ollama.py --categories coding agentic
  python benchmark_ollama.py --host http://192.168.0.149:11434
"""

import json
import time
import requests
import statistics
import csv
import os
import sys
import argparse
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://192.168.0.149:11434"
REQUEST_TIMEOUT = 180  # seconds per prompt

# Models to skip (embedding / OCR, not generative)
SKIP_MODELS = {
    "qwen3-embedding:8b",
    "deepseek-ocr:latest",
}

# ─────────────────────────────────────────────────────────────
# BENCHMARK PROMPTS
# Each entry: name, prompt, max_tokens
# ─────────────────────────────────────────────────────────────
BENCHMARKS = {
    "reasoning": [
        {
            "name": "logic_puzzle",
            "prompt": (
                "Solve step by step:\n"
                "Alice, Bob, and Carol each have a different pet: cat, dog, or fish.\n"
                "Clues: Alice does NOT have the cat. Bob does NOT have the fish. Carol does NOT have the dog.\n"
                "Who has which pet? Show your full reasoning before giving the final answer."
            ),
            "max_tokens": 350,
        },
        {
            "name": "math_word_problem",
            "prompt": (
                "Solve step by step and show all calculations:\n"
                "A train travels 120 km in 1.5 hours, then stops for 30 minutes, "
                "then travels 80 km in 1 hour. What is the average speed for the "
                "entire journey including the stop? Give the answer in km/h."
            ),
            "max_tokens": 350,
        },
        {
            "name": "syllogism",
            "prompt": (
                "Analyze this argument logically:\n"
                "Premise 1: All Zorks are Blips.\n"
                "Premise 2: Some Blips are Flops.\n"
                "Premise 3: No Flops are Zorks.\n"
                "Question: What can we DEFINITELY conclude? What can we NOT conclude? "
                "Explain each conclusion and why."
            ),
            "max_tokens": 300,
        },
    ],
    "summarization": [
        {
            "name": "document_bullets",
            "prompt": (
                "Summarize the following passage in exactly 3 bullet points. "
                "Each bullet MUST start with the '•' character:\n\n"
                "The Industrial Revolution, beginning in Britain in the late 18th century, "
                "fundamentally transformed human society through mechanization and factory production. "
                "Steam power replaced human and animal labour, enabling mass production at unprecedented scales. "
                "Cities grew rapidly as workers migrated from rural areas, creating both economic opportunity "
                "and social challenges including poor working conditions and urban poverty. The revolution spread "
                "globally through the 19th century, reshaping economies from agriculture to industry. New technologies "
                "like railways and telegraphs connected markets and accelerated information flow. Child labour was common "
                "in factories until labour reforms gradually improved conditions. The period laid the groundwork for "
                "modern capitalism and the global trade networks that persist today."
            ),
            "max_tokens": 200,
        },
        {
            "name": "technical_summary",
            "prompt": (
                "In 2-3 concise sentences, explain what transformer neural networks are "
                "and why the attention mechanism is central to their success in NLP. "
                "Be precise and avoid unnecessary padding."
            ),
            "max_tokens": 180,
        },
        {
            "name": "extract_key_facts",
            "prompt": (
                "Read the following and list exactly 4 key facts as a numbered list (1. 2. 3. 4.):\n\n"
                "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement "
                "to process information in ways classical computers cannot. While classical bits are either 0 or 1, "
                "quantum bits (qubits) can exist in superposition of both states simultaneously. This enables quantum "
                "computers to evaluate many possible solutions in parallel for certain problem types. Current quantum "
                "computers are prone to errors from decoherence and require extreme cooling to near absolute zero. "
                "Quantum advantage has been demonstrated for specific tasks like factoring large numbers and simulating "
                "molecular structures, but practical general-purpose quantum computers remain years away."
            ),
            "max_tokens": 200,
        },
    ],
    "coding": [
        {
            "name": "implement_function",
            "prompt": (
                "Write a Python function `find_duplicates(lst: list) -> list` that returns "
                "a sorted list of all values that appear more than once in the input list. "
                "Requirements: include a proper docstring, handle edge cases (empty list, no duplicates), "
                "and add 3 example assertions at the bottom demonstrating correctness."
            ),
            "max_tokens": 400,
        },
        {
            "name": "debug_code",
            "prompt": (
                "Find ALL bugs in this Python code and provide the fully corrected version with comments explaining each fix:\n\n"
                "```python\n"
                "def calculate_stats(numbers):\n"
                "    total = 0\n"
                "    for n in numbers:\n"
                "        total =+ n\n"
                "    mean = total / len(numbers)\n"
                "    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers) - 1\n"
                "    return mean, variance\n\n"
                "print(calculate_stats([]))\n"
                "```"
            ),
            "max_tokens": 400,
        },
        {
            "name": "sql_query",
            "prompt": (
                "Write a SQL query for this requirement:\n"
                "Table: orders (order_id INT, customer_id INT, order_date DATE, amount DECIMAL)\n"
                "Task: Find the top 5 customers by total purchase amount in 2024. "
                "Show customer_id, total_amount, and order_count. Sort descending by total_amount."
            ),
            "max_tokens": 200,
        },
        {
            "name": "code_review",
            "prompt": (
                "Review this Python code and identify issues with security, performance, and style:\n\n"
                "```python\n"
                "import pickle, os\n\n"
                "def load_user_data(filename):\n"
                "    f = open(filename)\n"
                "    data = pickle.loads(f.read())\n"
                "    return data\n\n"
                "def search_users(db, query):\n"
                "    sql = 'SELECT * FROM users WHERE name = ' + query\n"
                "    return db.execute(sql)\n"
                "```\n"
                "List each issue with severity (HIGH/MEDIUM/LOW) and how to fix it."
            ),
            "max_tokens": 400,
        },
    ],
    "agentic": [
        {
            "name": "task_decomposition",
            "prompt": (
                "You are an AI agent. A user says: "
                "'Analyze last month's sales data from our database, compare it to the same month last year, "
                "generate a PDF report with charts, and email it to the sales team.'\n\n"
                "Break this into exactly 6 numbered steps (1-6) that an autonomous agent should take. "
                "For each step, name the tool/action used."
            ),
            "max_tokens": 350,
        },
        {
            "name": "tool_selection",
            "prompt": (
                "Available tools: web_search(query), read_file(path), write_file(path, content), "
                "execute_python(code), send_email(to, subject, body), query_database(sql).\n\n"
                "Task: 'The Q3 financial report is in /reports/q3_2024.csv. Calculate the total revenue, "
                "find the top 3 products by revenue, and email a summary to cfo@company.com.'\n\n"
                "List the exact tool calls in order with arguments. Be specific."
            ),
            "max_tokens": 350,
        },
        {
            "name": "error_handling",
            "prompt": (
                "You are an agent that called an external API and received this error:\n"
                "{'error': 'rate_limit_exceeded', 'retry_after': 60, 'requests_today': 100, 'limit': 100}\n\n"
                "Describe exactly 4 strategies you would implement to handle this gracefully "
                "without losing data or failing the user's task. Number each strategy (1-4) and be specific."
            ),
            "max_tokens": 350,
        },
        {
            "name": "context_following",
            "prompt": (
                "You are a document processing agent. The user gave you these constraints:\n"
                "- Only process files smaller than 10MB\n"
                "- Never overwrite original files\n"
                "- Log every action to audit.log\n"
                "- If a file contains PII (names, emails, SSNs), flag it and skip processing\n\n"
                "A file 'customer_data.xlsx' (8.3MB) is queued. You open it and see columns: "
                "customer_name, email, purchase_amount, product_id.\n\n"
                "What do you do? Walk through your decision process step by step."
            ),
            "max_tokens": 350,
        },
    ],
}


# ─────────────────────────────────────────────────────────────
# OLLAMA API
# ─────────────────────────────────────────────────────────────

def get_models(host: str) -> list[str]:
    """Fetch available models from Ollama server."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=15)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return [m for m in models if m not in SKIP_MODELS]
    except Exception as e:
        print(f"❌ Cannot connect to Ollama at {host}: {e}")
        return []


def run_prompt(host: str, model: str, prompt: str, max_tokens: int = 300) -> dict:
    """Stream a prompt and collect timing metrics only (speed benchmark)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
        },
    }

    result = {
        "ttft": None,
        "total_time": None,
        "tokens_generated": 0,
        "tps": None,
        "response": "",
        "error": None,
        "prompt_eval_count": 0,
    }

    start = time.perf_counter()
    first_token_at = None

    try:
        with requests.post(
            f"{host}/api/generate",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                chunk = json.loads(raw)

                token = chunk.get("response", "")
                if token and first_token_at is None:
                    first_token_at = time.perf_counter()
                    result["ttft"] = first_token_at - start

                result["response"] += token

                if chunk.get("done"):
                    result["total_time"] = time.perf_counter() - start
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    if eval_duration_ns > 0:
                        result["tps"] = eval_count / (eval_duration_ns / 1_000_000_000)
                    result["tokens_generated"] = eval_count
                    result["prompt_eval_count"] = chunk.get("prompt_eval_count", 0)
                    break

    except requests.exceptions.Timeout:
        result["error"] = "timeout"
    except requests.exceptions.ConnectionError as e:
        result["error"] = f"connection_error: {e}"
    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION (TPS-ONLY)
# Heuristic quality scoring was removed: content correctness is the
# job of benchmark_quality.py / benchmark_agent.py, not of a speed
# harness. This script reports throughput and latency, period.
# ─────────────────────────────────────────────────────────────

TPS_TIER_THRESHOLDS = [
    # (min_tps, tier_label, tier_short)
    (30, "🟢 EXCELLENT",  "EXCELLENT"),
    (15, "🔵 GOOD",       "GOOD"),
    ( 8, "🟡 ADEQUATE",   "ADEQUATE"),
    ( 3, "🟠 MARGINAL",   "MARGINAL"),
    ( 0, "🔴 POOR",       "POOR"),
]

def classify_tier(tps: float) -> tuple[str, str]:
    """Return (emoji_label, short_label) based on throughput (TPS) only."""
    for min_t, label, short in TPS_TIER_THRESHOLDS:
        if tps >= min_t:
            return label, short
    return "🔴 POOR", "POOR"


# ─────────────────────────────────────────────────────────────
# MAIN BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────

def run_benchmark(host: str, models: list[str], benchmark_config: dict) -> dict:
    """Run all benchmark prompts for all models."""
    all_results = {}
    total = len(models) * sum(len(v) for v in benchmark_config.values())
    idx = 0

    print(f"\n{'='*65}")
    print(f"  Ollama Benchmark Suite  |  {len(models)} models  |  {total} tests")
    print(f"  Server: {host}")
    print(f"{'='*65}")

    for model in models:
        print(f"\n📦  {model}")
        all_results[model] = {"categories": {}, "overall": {}}

        for category, tests in benchmark_config.items():
            cat_data = []
            for test in tests:
                idx += 1
                label = f"{category}/{test['name']}"
                print(f"  [{idx:>3}/{total}] {label:<40}", end="", flush=True)

                result = run_prompt(host, model, test["prompt"], test.get("max_tokens", 300))

                if result["error"]:
                    print(f"  ❌ {result['error']}")
                else:
                    tps_s  = f"{result['tps']:.1f} tok/s" if result["tps"] else "N/A     "
                    ttft_s = f"TTFT:{result['ttft']:.2f}s" if result["ttft"] else "TTFT:N/A"
                    print(f"  ✅ {tps_s} | {ttft_s}")

                cat_data.append({
                    "test":            test["name"],
                    "tps":             result["tps"],
                    "ttft":            result["ttft"],
                    "total_time":      result["total_time"],
                    "tokens":          result["tokens_generated"],
                    "response_length": len(result["response"]),
                    "error":           result["error"],
                })

            valid = [r for r in cat_data if r["error"] is None]
            tps_vals   = [r["tps"]  for r in valid if r["tps"]]
            ttft_vals  = [r["ttft"] for r in valid if r["ttft"]]

            all_results[model]["categories"][category] = {
                "tests":        cat_data,
                "avg_tps":      statistics.mean(tps_vals)  if tps_vals  else 0.0,
                "avg_ttft":     statistics.mean(ttft_vals) if ttft_vals else 0.0,
                "success_rate": len(valid) / len(cat_data) * 100 if cat_data else 0,
            }

        # Overall aggregates
        cats       = all_results[model]["categories"]
        all_tps    = [c["avg_tps"]    for c in cats.values() if c["avg_tps"]]
        all_ttft   = [c["avg_ttft"]   for c in cats.values() if c["avg_ttft"]]

        avg_tps     = statistics.mean(all_tps)  if all_tps  else 0.0
        tier, _     = classify_tier(avg_tps)

        all_results[model]["overall"] = {
            "avg_tps":      round(avg_tps, 2),
            "avg_ttft":     round(statistics.mean(all_ttft), 3) if all_ttft else 0.0,
            "tier":         tier,
        }

    return all_results


# ─────────────────────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────────────────────

def console_summary(results: dict):
    """Print ranked summary to stdout (throughput/latency only)."""
    ranked = sorted(
        results.items(),
        key=lambda x: (x[1]["overall"]["avg_tps"], -x[1]["overall"]["avg_ttft"]),
        reverse=True,
    )

    cats = ["reasoning", "summarization", "coding", "agentic"]
    separator = "=" * 100
    print(f"\n{separator}")
    print("  FINAL RANKINGS (speed only)")
    print(separator)
    hdr = f"{'#':<4} {'Model':<48} {'Tier':<12} {'TPS':>6} {'TTFT':>7}"
    for c in cats:
        hdr += f"  {c[:6].capitalize():>7}"
    print(hdr)
    print("-" * 100)

    for rank, (model, data) in enumerate(ranked, 1):
        o  = data["overall"]
        cs = data["categories"]
        short = model if len(model) <= 47 else model[:44] + "..."
        tier_short = o["tier"].split()[-1]
        row = (
            f"{rank:<4} {short:<48} {tier_short:<12} "
            f"{o['avg_tps']:>6.1f} {o['avg_ttft']:>6.2f}s"
        )
        for c in cats:
            t = cs.get(c, {}).get("avg_tps", 0)
            row += f"  {t:>7.1f}"
        print(row)

    # Recommendations
    print(f"\n{'─'*60}")
    print("  RECOMMENDATIONS")
    print(f"{'─'*60}")
    tiers = {"EXCELLENT": [], "GOOD": [], "ADEQUATE": [], "MARGINAL": [], "POOR": []}
    for model, data in ranked:
        _, short = classify_tier(data["overall"]["avg_tps"])
        tiers[short].append(model)

    labels = {
        "EXCELLENT": ("✅ Keep — primary models",           "🟢"),
        "GOOD":      ("✅ Keep — solid for specific tasks", "🔵"),
        "ADEQUATE":  ("⚠️  Keep only if unique capability", "🟡"),
        "MARGINAL":  ("🗑️  Consider removing",              "🟠"),
        "POOR":      ("❌ Remove / replace",                "🔴"),
    }
    for tier, models in tiers.items():
        if models:
            emoji, dot = labels[tier]
            print(f"\n{dot} {tier} — {emoji}:")
            for m in models:
                print(f"     • {m}")


def save_html_report(results: dict, path: str):
    """Write a dark-themed HTML report with sortable table (speed only)."""
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["overall"]["avg_tps"],
        reverse=True,
    )
    cats = ["reasoning", "summarization", "coding", "agentic"]
    tier_colors = {
        "EXCELLENT": "#3fb950",
        "GOOD":      "#58a6ff",
        "ADEQUATE":  "#d29922",
        "MARGINAL":  "#f0883e",
        "POOR":      "#f85149",
    }
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows_html = ""
    for rank, (model, data) in enumerate(ranked, 1):
        o  = data["overall"]
        cs = data["categories"]
        _, short_tier = classify_tier(o["avg_tps"])
        color = tier_colors.get(short_tier, "#8b949e")
        cat_cells = "".join(
            f"<td>{cs.get(c, {}).get('avg_tps', 0):.1f}</td>" for c in cats
        )
        rows_html += (
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td><strong>{model}</strong></td>"
            f"<td style='color:{color};font-weight:bold'>{o['tier']}</td>"
            f"<td>{o['avg_tps']:.1f}</td>"
            f"<td>{o['avg_ttft']:.2f}s</td>"
            f"</tr>\n"
        )

    legend_rows = ""
    for min_t, label, short in TPS_TIER_THRESHOLDS:
        color = tier_colors.get(short, "#8b949e")
        rec_map = {
            "EXCELLENT": "Primary throughput",
            "GOOD":      "Solid throughput",
            "ADEQUATE":  "Usable on small tasks",
            "MARGINAL":  "Latency-heavy",
            "POOR":      "Too slow to be useful",
        }
        legend_rows += (
            f"<tr>"
            f"<td style='color:{color};font-weight:bold'>{label}</td>"
            f"<td>TPS ≥ {min_t}</td>"
            f"<td>{rec_map[short]}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ollama Speed Benchmark Report – {now}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #e6edf3; padding: 32px 24px; line-height: 1.5; }}
  h1   {{ color: #58a6ff; font-size: 1.6em; margin-bottom: 4px; }}
  h2   {{ color: #79c0ff; font-size: 1.1em; margin: 32px 0 12px;
         border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
  p.meta {{ color: #8b949e; font-size: 0.85em; margin-bottom: 24px; }}
  table  {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ background: #161b22; color: #79c0ff; padding: 10px 12px;
       text-align: left; border: 1px solid #30363d; white-space: nowrap; }}
  td {{ padding: 9px 12px; border: 1px solid #30363d; vertical-align: middle; }}
  tr:nth-child(even) {{ background: #0d1117; }}
  tr:nth-child(odd)  {{ background: #111318; }}
  tr:hover           {{ background: #1c2128; }}
  .tip {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px 20px; margin: 20px 0; font-size: 0.9em; color: #8b949e; }}
  .tip strong {{ color: #e6edf3; }}
</style>
</head>
<body>
<h1>🤖 Ollama Model Speed Report</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Server: {OLLAMA_HOST} &nbsp;|&nbsp; Models tested: {len(results)}</p>

<div class="tip">
  This script measures <strong>throughput and latency only</strong>. Heuristic quality
  scoring was removed — correctness is assessed by <code>benchmark_quality.py</code>
  and <code>benchmark_agent.py</code>, not by a speed harness.
</div>

<h2>Overall Rankings</h2>
<table>
<tr>
  <th>#</th><th>Model</th><th>Tier</th>
  <th>TPS ↓</th><th>TTFT</th>
  <th>Reasoning</th><th>Summarize</th><th>Coding</th><th>Agentic</th>
</tr>
{rows_html}
</table>

<h2>Classification Criteria</h2>
<table>
<tr><th>Tier</th><th>Threshold</th><th>Recommendation</th></tr>
{legend_rows}
</table>

<h2>Notes</h2>
<div class="tip">
  <strong>Excluded from benchmarks:</strong> qwen3-embedding:8b (embedding model),
  deepseek-ocr:latest (OCR specialist — not a general generator).<br><br>
  <strong>Duplicate model groups detected:</strong><br>
  • <code>gemma3:27b</code> vs <code>hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_XL</code> — same base model<br>
  • <code>gpt-oss:20b</code> vs <code>hf.co/unsloth/gpt-oss-20b-GGUF:F16</code> — same base, different quant<br>
  • <code>qwen3-coder:30b</code> vs <code>hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL</code> — same base model<br>
  Keep the higher-throughput one from each pair and remove the other to reclaim disk space.
</div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📊  HTML report → {path}")


def save_csv(results: dict, path: str):
    """Write flat CSV of per-model aggregate speed metrics."""
    cats = ["reasoning", "summarization", "coding", "agentic"]
    fieldnames = ["rank", "model", "tier", "avg_tps", "avg_ttft"] + \
                 [f"{c}_tps" for c in cats]
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["overall"]["avg_tps"],
        reverse=True,
    )
    rows = []
    for rank, (model, data) in enumerate(ranked, 1):
        o  = data["overall"]
        cs = data["categories"]
        _, short_tier = classify_tier(o["avg_tps"])
        row = {
            "rank":      rank,
            "model":     model,
            "tier":      short_tier,
            "avg_tps":     round(o["avg_tps"], 2),
            "avg_ttft":    round(o["avg_ttft"], 3),
        }
        for c in cats:
            row[f"{c}_tps"] = round(cs.get(c, {}).get("avg_tps", 0), 2)
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
        description="Ollama Model Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host",       default=OLLAMA_HOST,
                        help=f"Ollama server URL (default: {OLLAMA_HOST})")
    parser.add_argument("--models",     nargs="+", metavar="MODEL",
                        help="Specific model names to test (default: all from server)")
    parser.add_argument("--categories", nargs="+",
                        choices=list(BENCHMARKS.keys()), metavar="CAT",
                        help="Limit to specific categories")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files (default: current dir)")
    parser.add_argument("--quick",      action="store_true",
                        help="Quick mode: run only 1 test per category (faster preview)")
    parser.add_argument("--list",       action="store_true",
                        help="Just list available models and exit")
    args = parser.parse_args()

    # Resolve benchmark config
    bench = {k: v for k, v in BENCHMARKS.items()
             if not args.categories or k in args.categories}
    if args.quick:
        bench = {k: v[:1] for k, v in bench.items()}
        print("⚡  Quick mode — 1 test per category")

    # Get models
    available_models = get_models(args.host)
    if not available_models:
        sys.exit(1)

    if args.list:
        print(f"\nModels on {args.host}:")
        for m in available_models:
            print(f"  • {m}")
        return

    models = args.models if args.models else available_models
    # Validate requested models exist
    unknown = [m for m in models if m not in available_models]
    if unknown:
        print(f"⚠️  Unknown models (will skip): {unknown}")
        models = [m for m in models if m in available_models]
    if not models:
        print("No valid models to benchmark.")
        sys.exit(1)

    # Run
    results = run_benchmark(args.host, models, bench)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(args.output_dir, f"benchmark_{ts}.json")
    html_path = os.path.join(args.output_dir, f"benchmark_{ts}.html")
    csv_path  = os.path.join(args.output_dir, f"benchmark_{ts}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾  Raw JSON      → {json_path}")

    save_html_report(results, html_path)
    save_csv(results, csv_path)
    console_summary(results)


if __name__ == "__main__":
    main()
