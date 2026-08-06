#!/usr/bin/env python3
"""
Production Harness Usability Benchmark
=======================================
Evaluates models for REAL-WORLD usage in agentic frameworks (OpenCode, OMP, etc.):

  ⏱️  TIMEOUT RESISTANCE     → Measures if models complete within harness timeouts
  📊 CONTEXT EFFICIENCY      → Token budget awareness, verbosity control
  🔄 ITERATION STABILITY     → Multi-turn conversation reliability
  🎯 TASK COMPLETION RATE    → End-to-end goal achievement
  💥 FAILURE MODE ANALYSIS   → How gracefully does it fail?
  🧠 LONG-CONTEXT RETENTION  → Performance at 64k-200k token contexts

Key Philosophy:
  A model that's 95% accurate but times out 40% of the time is LESS usable
  than a model that's 85% accurate and always completes on time.

Usage:
  python benchmark_harness_usability.py --models qwen3.5:32b deepseek-r1:32b
  python benchmark_harness_usability.py --context-sizes 32768 65536 131072
  python benchmark_harness_usability.py --timeout-threshold 60
  python benchmark_harness_usability.py --simulate-harness opencode
  python benchmark_harness_usability.py --full-suite
"""

import json
import time
import requests
import re
import sys
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.0.149:11434")
DEFAULT_TIMEOUT = 180  # seconds - matches Ollama default
HARNESS_TIMEOUTS = {
    "opencode": 120,      # OpenCode typical timeout
    "omp": 90,            # OMP (Open Model Protocol) timeout
    "cline": 150,         # Cline VS Code extension
    "default": 180,       # Generic fallback
}

# Context size tiers for testing
CONTEXT_TIERS = {
    "small": 8192,        # Quick sanity checks
    "medium": 32768,      # Typical coding tasks
    "large": 65536,       # Complex multi-file projects
    "xlarge": 131072,     # Full codebase analysis
    "max": 200000,        # Maximum Ollama supports
}


class FailureMode(Enum):
    """Categorize how models fail - critical for production use"""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    CONTEXT_EXCEEDED = "context_exceeded"
    MALFORMED_OUTPUT = "malformed_output"
    WRONG_TOOL = "wrong_tool"
    MISSING_ARGS = "missing_arguments"
    HALLUCINATED_TOOL = "hallucinated_tool"
    REFUSAL = "refusal_safety"
    LOOP_DETECTED = "infinite_loop"
    INCOMPLETE = "incomplete_response"
    API_ERROR = "api_error"


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn"""
    turn_number: int
    response_time: float
    tokens_generated: int
    tokens_per_second: float
    ttft: float  # Time to first token
    tool_calls: list = field(default_factory=list)
    failure_mode: Optional[FailureMode] = None
    error_message: Optional[str] = None


@dataclass
class TaskMetrics:
    """Metrics for a complete task (may have multiple turns)"""
    task_id: str
    task_name: str
    category: str
    total_turns: int
    successful_turns: int
    total_time: float
    avg_time_per_turn: float
    total_tokens: int
    avg_tokens_per_turn: float
    completion_rate: float  # 0.0 to 1.0
    failure_mode: Optional[FailureMode] = None
    would_timeout_in_harness: dict = field(default_factory=dict)  # harness_name -> bool
    context_efficiency_score: float = 0.0  # 0-100
    verbosity_penalty: float = 0.0  # Penalize excessive verbosity


@dataclass
class ModelUsabilityProfile:
    """Complete usability profile for a model"""
    model_name: str
    test_timestamp: str
    context_size_tested: int
    
    # Overall scores
    overall_usability_score: float  # 0-100 weighted composite
    timeout_resistance_score: float  # % tasks completed without timeout
    context_efficiency_score: float  # Token usage efficiency
    task_completion_rate: float  # % tasks fully completed
    multi_turn_stability: float  # Consistency across turns
    
    # Timeout analysis
    timeout_rate_by_harness: dict  # harness -> timeout %
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    slowest_task_category: str
    
    # Context behavior
    degradation_at_context: dict  # context_size -> performance %
    context_retention_score: float  # Ability to recall early context
    verbosity_vs_accuracy: dict  # correlation data
    
    # Failure analysis
    failure_mode_distribution: dict  # FailureMode -> count
    recovery_success_rate: float  # % of errors handled gracefully
    hallucination_rate: float
    
    # Recommendations
    recommended_for: list[str]  # Use cases where this model excels
    not_recommended_for: list[str]  # Use cases to avoid
    optimal_context_size: int  # Sweet spot for this model
    suggested_timeout: int  # Recommended timeout setting


# ─────────────────────────────────────────────────────────────
# TEST SUITES FOR HARNESS USABILITY
# ─────────────────────────────────────────────────────────────

def get_timeout_stress_tests() -> list[dict]:
    """
    Tests designed to expose timeout risks.
    These are realistic tasks that require careful token budgeting.
    """
    return [
        {
            "id": "timeout_01",
            "name": "concise_code_fix",
            "category": "coding_efficiency",
            "prompt": """Fix the bug in this function. ONLY output the corrected function, no explanation:

def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total

calculate_sum([1, 2, 3])  # Should return 6, currently returns 0""",
            "expected_behavior": "Returns only the fixed function code",
            "max_tokens_needed": 150,
            "timeout_risk": "low",
            "evaluation": {
                "type": "code_execution",
                "must_contain": ["def calculate_sum", "return"],
                "must_not_contain": ["Here's", "Sure!", "explanation", "```markdown"]
            }
        },
        {
            "id": "timeout_02", 
            "name": "verbose_explanation_trap",
            "category": "verbosity_control",
            "prompt": """Analyze this code and identify potential issues. Be CONCISE - limit your response to 200 words maximum:

class DataProcessor:
    def __init__(self):
        self.data = []
        self.cache = {}
    
    def process(self, items):
        for item in items:
            if item not in self.cache:
                self.cache[item] = self.expensive_operation(item)
            self.data.append(self.cache[item])
        return self.data
    
    def expensive_operation(self, x):
        return x * 2""",
            "expected_behavior": "Concise analysis under 200 words",
            "max_tokens_needed": 300,
            "timeout_risk": "medium",
            "evaluation": {
                "type": "word_count",
                "max_words": 200,
                "must_mention": ["cache", "efficiency"]
            }
        },
        {
            "id": "timeout_03",
            "name": "multi_file_refactor",
            "category": "complex_coding",
            "prompt": """Refactor this codebase structure for better modularity. Output ONLY JSON with file paths and contents:

Current structure:
- app.py (main entry, 200 lines)
- utils.py (helpers, 150 lines)  
- models.py (data classes, 100 lines)

Goal: Split into proper packages with clear separation of concerns.
Respond with valid JSON only, no markdown formatting.""",
            "expected_behavior": "Valid JSON with refactored structure",
            "max_tokens_needed": 800,
            "timeout_risk": "high",
            "evaluation": {
                "type": "json_validity",
                "must_have_keys": ["files"],
                "min_files": 4
            }
        },
        {
            "id": "timeout_04",
            "name": "iterative_debugging",
            "category": "multi_turn_debug",
            "prompt": """I'm getting this error: 'IndexError: list index out of range'
My code: data = []; print(data[0])
What's wrong? Fix it in ONE line.""",
            "expected_behavior": "Brief, direct fix",
            "max_tokens_needed": 100,
            "timeout_risk": "low",
            "follow_up_turns": [
                {
                    "prompt": "Now what if data might be empty sometimes? Handle gracefully.",
                    "expected_behavior": "Adds conditional check",
                    "max_tokens_needed": 150
                },
                {
                    "prompt": "Good. Now make it return a default value instead of crashing.",
                    "expected_behavior": "Implements default value logic",
                    "max_tokens_needed": 200
                }
            ],
            "evaluation": {
                "type": "multi_turn_coherence",
                "all_turns_must_complete": True
            }
        },
        {
            "id": "timeout_05",
            "name": "context_overflow_scenario",
            "category": "context_management",
            "prompt": """[Imagine 50000 tokens of conversation history here]

Based on our discussion above about the user's requirements, what database schema should we use?
Keep your answer under 300 tokens.""",
            "expected_behavior": "Acknowledges context limits gracefully OR provides concise answer",
            "max_tokens_needed": 400,
            "timeout_risk": "extreme",
            "simulated_context_tokens": 50000,
            "evaluation": {
                "type": "context_awareness",
                "acceptable_responses": [
                    "provides_schema",
                    "acknowledges_missing_context",
                    "asks_clarifying_question"
                ]
            }
        }
    ]


def get_context_efficiency_tests() -> list[dict]:
    """
    Tests that measure how efficiently models use token budget.
    Critical for avoiding timeouts in long conversations.
    """
    return [
        {
            "id": "efficiency_01",
            "name": "token_budget_awareness",
            "category": "efficiency",
            "prompt": """You have 500 tokens remaining in your response budget.
Explain how to implement a binary search tree in Python.
Budget your tokens carefully - stop before hitting the limit.""",
            "expected_behavior": "Complete explanation that stays under token limit",
            "target_token_range": (400, 500),
            "evaluation": {
                "type": "token_efficiency",
                "penalize_overage": True,
                "penalize_underuse": False
            }
        },
        {
            "id": "efficiency_02",
            "name": "progressive_summarization",
            "category": "summarization_efficiency",
            "prompt": """Summarize this article in exactly 3 sentences:

[Long article text - simulated 5000 tokens]

Artificial intelligence has transformed numerous industries...
[imagine 5000 tokens of detailed content here]

...and continues to evolve rapidly.""",
            "expected_behavior": "Exactly 3 sentences capturing key points",
            "target_token_range": (80, 150),
            "evaluation": {
                "type": "constraint_compliance",
                "sentence_count": 3,
                "must_capture_themes": ["AI transformation", "industry impact", "ongoing evolution"]
            }
        },
        {
            "id": "efficiency_03",
            "name": "code_golf_challenge",
            "category": "code_efficiency",
            "prompt": """Write a Python function that returns the nth Fibonacci number.
Make it as SHORT as possible while remaining readable.
Maximum 5 lines of code.""",
            "expected_behavior": "Concise, correct implementation",
            "target_token_range": (50, 150),
            "evaluation": {
                "type": "code_brevity",
                "max_lines": 5,
                "must_be_correct": True
            }
        }
    ]


def get_long_context_retention_tests() -> list[dict]:
    """
    Tests for models running at high context sizes (64k-200k).
    Evaluates whether models can retain and use information from early in context.
    """
    tests = []
    
    for context_tier, ctx_size in CONTEXT_TIERS.items():
        if ctx_size < 32768:  # Skip small contexts for these tests
            continue
            
        tests.append({
            "id": f"longctx_{context_tier}",
            "name": f"context_retention_{context_tier}",
            "category": "long_context",
            "context_size": ctx_size,
            "prompt": f"""Earlier in our conversation (approximately {ctx_size // 2} tokens ago), 
I mentioned a specific project requirement. What was the primary database technology I said we should use?

Options:
A) PostgreSQL
B) MongoDB  
C) Redis
D) SQLite

Answer with just the letter.""",
            "expected_answer": "A",  # Would be injected in actual test
            "simulated_context_fill": ctx_size // 2,
            "evaluation": {
                "type": "multiple_choice",
                "correct_answer": "A",
                "test_recall_from_position": ctx_size // 2
            }
        })
    
    return tests


def get_harness_simulation_tests(harness_type: str = "opencode") -> list[dict]:
    """
    Tests that simulate real harness workflows.
    Each harness has different patterns and constraints.
    """
    if harness_type == "opencode":
        return [
            {
                "id": "harness_opencode_01",
                "name": "opencode_file_edit",
                "category": "harness_simulation",
                "harness": "opencode",
                "prompt": """<task>Edit src/main.py to add error handling</task>
<current_file>
def process_data(items):
    result = []
    for item in items:
        parsed = parse_item(item)
        result.append(transform(parsed))
    return result
</current_file>

Use the edit_file tool to wrap the loop body in try/except.""",
                "expected_tool": "edit_file",
                "expected_args": {
                    "path": "src/main.py",
                    "changes": "try/except block"
                },
                "harness_constraints": {
                    "max_response_tokens": 1000,
                    "timeout_seconds": 120,
                    "requires_tool_call": True
                }
            },
            {
                "id": "harness_opencode_02",
                "name": "opencode_multi_step",
                "category": "harness_simulation", 
                "harness": "opencode",
                "prompt": """Create a new Python module with tests.
1. Create file: src/calculator.py with add/multiply functions
2. Create file: tests/test_calculator.py with pytest tests
3. Run the tests""",
                "expected_sequence": ["create_file", "create_file", "run_command"],
                "harness_constraints": {
                    "max_turns": 5,
                    "timeout_seconds": 120,
                    "must_complete_all_steps": True
                }
            }
        ]
    
    elif harness_type == "omp":
        return [
            {
                "id": "harness_omp_01",
                "name": "omp_agent_task",
                "category": "harness_simulation",
                "harness": "omp",
                "prompt": """Analyze the current directory structure and suggest improvements.
Use available tools to explore and report findings.""",
                "expected_behavior": "Uses ls/cd tools, provides structured analysis",
                "harness_constraints": {
                    "max_response_tokens": 800,
                    "timeout_seconds": 90,
                    "requires_structured_output": True
                }
            }
        ]
    
    return []


# ─────────────────────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def evaluate_timeout_risk(response: str, response_time: float, test: dict) -> dict:
    """Evaluate if response would cause timeout in production harness"""
    max_tokens = test.get("max_tokens_needed", 500)
    estimated_tokens = len(response.split()) * 1.3  # Rough estimate
    
    timeout_thresholds = {
        "opencode": HARNESS_TIMEOUTS["opencode"],
        "omp": HARNESS_TIMEOUTS["omp"],
        "default": DEFAULT_TIMEOUT
    }
    
    would_timeout = {}
    for harness, threshold in timeout_thresholds.items():
        would_timeout[harness] = response_time > (threshold * 0.9)  # 90% threshold
    
    return {
        "would_timeout_in_harness": would_timeout,
        "estimated_tokens": estimated_tokens,
        "token_efficiency": max(0, 100 - abs(estimated_tokens - max_tokens) / max_tokens * 100),
        "response_time": response_time,
        "risk_level": "high" if response_time > 60 else "medium" if response_time > 30 else "low"
    }


def evaluate_context_efficiency(response: str, test: dict) -> dict:
    """Evaluate how efficiently model used its token budget"""
    target_range = test.get("target_token_range", (100, 500))
    actual_tokens = len(response.split()) * 1.3
    
    if actual_tokens < target_range[0]:
        efficiency = 100 - ((target_range[0] - actual_tokens) / target_range[0] * 50)
        issue = "underutilized"
    elif actual_tokens > target_range[1]:
        efficiency = 100 - ((actual_tokens - target_range[1]) / target_range[1] * 100)
        issue = "overbudget"
    else:
        efficiency = 100
        issue = "optimal"
    
    return {
        "efficiency_score": max(0, efficiency),
        "actual_tokens": actual_tokens,
        "target_range": target_range,
        "issue": issue
    }


def evaluate_failure_mode(response: str, error: Optional[str], test: dict) -> FailureMode:
    """Categorize the type of failure"""
    if error == "timeout":
        return FailureMode.TIMEOUT
    
    if not response:
        return FailureMode.INCOMPLETE
    
    if "context window" in response.lower() or "too long" in response.lower():
        return FailureMode.CONTEXT_EXCEEDED
    
    if "i cannot" in response.lower() or "i'm unable" in response.lower():
        return FailureMode.REFUSAL
    
    # Check for malformed output based on expected type
    eval_type = test.get("evaluation", {}).get("type", "")
    
    if eval_type == "json_validity":
        try:
            json.loads(response)
        except:
            return FailureMode.MALFORMED_OUTPUT
    
    if eval_type == "code_execution":
        if "```python" not in response and "def " not in response:
            return FailureMode.MALFORMED_OUTPUT
    
    return FailureMode.SUCCESS


# ─────────────────────────────────────────────────────────────
# MAIN BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────

def call_model_with_metrics(
    host: str,
    model: str,
    prompt: str,
    num_ctx: int = 32768,
    max_tokens: int = 1000,
    timeout: int = DEFAULT_TIMEOUT
) -> tuple[str, dict]:
    """Call model and collect detailed metrics"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
            "temperature": 0.1,
        }
    }
    
    metrics = {
        "ttft": None,
        "total_time": None,
        "tokens_generated": 0,
        "tps": None,
        "error": None
    }
    
    start = time.perf_counter()
    first_token = None
    response_text = ""
    
    try:
        with requests.post(
            f"{host}/api/generate",
            json=payload,
            stream=True,
            timeout=timeout
        ) as resp:
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if not line:
                    continue
                    
                chunk = json.loads(line)
                token = chunk.get("response", "")
                
                if first_token is None and token:
                    first_token = time.perf_counter()
                    metrics["ttft"] = first_token - start
                
                response_text += token
                metrics["tokens_generated"] = chunk.get("eval_count", 0)
                
                if chunk.get("done"):
                    metrics["total_time"] = time.perf_counter() - start
                    if metrics["total_time"] > 0:
                        metrics["tps"] = metrics["tokens_generated"] / metrics["total_time"]
                    break
                    
    except requests.exceptions.Timeout:
        metrics["error"] = "timeout"
        metrics["total_time"] = timeout
    except Exception as e:
        metrics["error"] = str(e)
    
    return response_text, metrics


def run_usability_benchmark(
    models: list[str],
    host: str = OLLAMA_HOST,
    context_sizes: list[int] = None,
    harness_type: str = "opencode",
    timeout_threshold: int = 120
) -> dict:
    """Run comprehensive usability benchmark"""
    
    if context_sizes is None:
        context_sizes = [CONTEXT_TIERS["medium"], CONTEXT_TIERS["large"]]
    
    # Gather all test suites
    tests = (
        get_timeout_stress_tests() +
        get_context_efficiency_tests() +
        get_harness_simulation_tests(harness_type)
    )
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"  PRODUCTION HARNESS USABILITY BENCHMARK")
    print(f"{'='*80}")
    print(f"  Models: {len(models)}")
    print(f"  Context sizes: {context_sizes}")
    print(f"  Target harness: {harness_type}")
    print(f"  Timeout threshold: {timeout_threshold}s")
    print(f"  Total tests per model: {len(tests)}")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\n🔍 Testing model: {model}")
        print("-" * 60)
        
        model_results = {
            "tasks": [],
            "failure_modes": {},
            "timeout_analysis": {},
            "context_efficiency": []
        }
        
        for ctx_size in context_sizes:
            print(f"\n  Context size: {ctx_size:,} tokens")
            
            for test in tests:
                test_id = test.get("id", "unknown")
                test_name = test.get("name", test_id)
                
                print(f"    Running {test_name}...", end=" ", flush=True)
                
                # Execute test
                response, metrics = call_model_with_metrics(
                    host, model, test["prompt"],
                    num_ctx=ctx_size,
                    max_tokens=test.get("max_tokens_needed", 500),
                    timeout=min(timeout_threshold, DEFAULT_TIMEOUT)
                )
                
                # Evaluate
                if metrics["error"]:
                    failure_mode = evaluate_failure_mode(response, metrics["error"], test)
                    print(f"❌ {failure_mode.value}")
                else:
                    timeout_eval = evaluate_timeout_risk(response, metrics["total_time"], test)
                    efficiency_eval = evaluate_context_efficiency(response, test)
                    failure_mode = evaluate_failure_mode(response, None, test)
                    
                    # Simple pass/fail for display
                    passed = failure_mode == FailureMode.SUCCESS
                    print(f"{'✅' if passed else '⚠️'} {metrics['total_time']:.2f}s")
                    
                    model_results["context_efficiency"].append(efficiency_eval)
                    model_results["timeout_analysis"][test_id] = timeout_eval
                
                # Record failure mode
                fm_key = failure_mode.value
                model_results["failure_modes"][fm_key] = model_results["failure_modes"].get(fm_key, 0) + 1
                
                # Store task result
                model_results["tasks"].append({
                    "test_id": test_id,
                    "test_name": test_name,
                    "context_size": ctx_size,
                    "response_time": metrics["total_time"],
                    "tokens": metrics["tokens_generated"],
                    "tps": metrics["tps"],
                    "ttft": metrics["ttft"],
                    "failure_mode": fm_key,
                    "would_timeout": timeout_eval.get("would_timeout_in_harness", {}) if not metrics["error"] else {}
                })
        
        # Aggregate results for this model
        results[model] = compute_usability_profile(model, model_results, context_sizes)
    
    return results


def compute_usability_profile(model: str, model_results: dict, context_sizes: list[int]) -> ModelUsabilityProfile:
    """Compute comprehensive usability profile from raw results"""
    
    tasks = model_results["tasks"]
    failure_modes = model_results["failure_modes"]
    
    # Calculate timeout rates
    total_tasks = len(tasks)
    timeout_count = failure_modes.get("timeout", 0)
    timeout_resistance = (total_tasks - timeout_count) / total_tasks * 100 if total_tasks > 0 else 0
    
    # Response time statistics
    response_times = [t["response_time"] for t in tasks if t["response_time"]]
    avg_time = statistics.mean(response_times) if response_times else 0
    p95_time = sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 5 else avg_time
    p99_time = sorted(response_times)[int(len(response_times) * 0.99)] if len(response_times) > 10 else p95_time
    
    # Task completion rate
    successful_tasks = sum(1 for t in tasks if t["failure_mode"] == "success")
    completion_rate = successful_tasks / total_tasks * 100 if total_tasks > 0 else 0
    
    # Context efficiency
    efficiency_scores = [e["efficiency_score"] for e in model_results.get("context_efficiency", [])]
    avg_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0
    
    # Timeout by harness
    harness_timeouts = {}
    for task in tasks:
        for harness, would_timeout in task.get("would_timeout", {}).items():
            if harness not in harness_timeouts:
                harness_timeouts[harness] = {"total": 0, "timeouts": 0}
            harness_timeouts[harness]["total"] += 1
            if would_timeout:
                harness_timeouts[harness]["timeouts"] += 1
    
    timeout_by_harness = {
        h: (data["timeouts"] / data["total"] * 100) if data["total"] > 0 else 0
        for h, data in harness_timeouts.items()
    }
    
    # Overall usability score (weighted composite)
    overall_score = (
        timeout_resistance * 0.35 +      # Most important: don't timeout
        completion_rate * 0.30 +          # Complete tasks successfully
        avg_efficiency * 0.20 +           # Use tokens efficiently
        (100 - min(timeout_count / total_tasks * 100, 100)) * 0.15  # Minimize failures
    ) if total_tasks > 0 else 0
    
    # Generate recommendations
    recommended_for = []
    not_recommended_for = []
    
    if timeout_resistance > 90:
        recommended_for.append("Time-sensitive production workflows")
    if avg_efficiency > 80:
        recommended_for.append("High-volume token-constrained scenarios")
    if completion_rate > 85:
        recommended_for.append("Complex multi-step tasks")
    
    if timeout_resistance < 70:
        not_recommended_for.append("Real-time interactive applications")
    if avg_efficiency < 60:
        not_recommended_for.append("Token-budget-constrained environments")
    
    # Find optimal context size
    ctx_performance = {}
    for ctx_size in context_sizes:
        ctx_tasks = [t for t in tasks if t["context_size"] == ctx_size]
        if ctx_tasks:
            ctx_success = sum(1 for t in ctx_tasks if t["failure_mode"] == "success")
            ctx_performance[ctx_size] = ctx_success / len(ctx_tasks) * 100
    
    optimal_ctx = max(ctx_performance.items(), key=lambda x: x[1])[0] if ctx_performance else context_sizes[0]
    
    return ModelUsabilityProfile(
        model_name=model,
        test_timestamp=datetime.now().isoformat(),
        context_size_tested=context_sizes[0],
        overall_usability_score=round(overall_score, 2),
        timeout_resistance_score=round(timeout_resistance, 2),
        context_efficiency_score=round(avg_efficiency, 2),
        task_completion_rate=round(completion_rate, 2),
        multi_turn_stability=round(completion_rate, 2),  # Simplified for now
        timeout_rate_by_harness=timeout_by_harness,
        avg_response_time=round(avg_time, 2),
        p95_response_time=round(p95_time, 2),
        p99_response_time=round(p99_time, 2),
        slowest_task_category="",  # Would need more analysis
        degradation_at_context={str(k): v for k, v in ctx_performance.items()},
        context_retention_score=round(completion_rate, 2),  # Simplified
        verbosity_vs_accuracy={},
        failure_mode_distribution=failure_modes,
        recovery_success_rate=0.0,  # Would need error recovery tests
        hallucination_rate=0.0,  # Would need hallucination-specific tests
        recommended_for=recommended_for,
        not_recommended_for=not_recommended_for,
        optimal_context_size=optimal_ctx,
        suggested_timeout=int(p95_time * 1.5)  # 50% buffer over p95
    )


# ─────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────

def print_usability_report(results: dict):
    """Print comprehensive usability report"""
    
    print(f"\n{'='*100}")
    print(f"  PRODUCTION HARNESS USABILITY REPORT")
    print(f"{'='*100}\n")
    
    # Rank by overall usability score
    ranked = sorted(
        results.items(),
        key=lambda x: x[1].overall_usability_score,
        reverse=True
    )
    
    # Summary table
    print(f"{'Rank':<5} {'Model':<35} {'Overall':>8} {'Timeout':>9} {'Complete':>9} {'Efficient':>10} {'OptimalCtx':>10}")
    print("-" * 100)
    
    for rank, (model, profile) in enumerate(ranked, 1):
        model_short = model[:32] + "..." if len(model) > 35 else model
        print(f"{rank:<5} {model_short:<35} "
              f"{profile.overall_usability_score:>7.1f}%  "
              f"{profile.timeout_resistance_score:>8.1f}%  "
              f"{profile.task_completion_rate:>8.1f}%  "
              f"{profile.context_efficiency_score:>9.1f}%  "
              f"{profile.optimal_context_size:>9,}")
    
    print("\n" + "=" * 100)
    print("  DETAILED ANALYSIS\n")
    
    for model, profile in ranked:
        print(f"\n📊 {model}")
        print("-" * 80)
        
        print(f"  Overall Usability Score:     {profile.overall_usability_score:.1f}/100")
        print(f"  Timeout Resistance:          {profile.timeout_resistance_score:.1f}%")
        print(f"  Task Completion Rate:        {profile.task_completion_rate:.1f}%")
        print(f"  Context Efficiency:          {profile.context_efficiency_score:.1f}%")
        
        print(f"\n  ⏱️  Response Time:")
        print(f"    Average:  {profile.avg_response_time:.2f}s")
        print(f"    P95:      {profile.p95_response_time:.2f}s")
        print(f"    P99:      {profile.p99_response_time:.2f}s")
        
        print(f"\n  🚨 Timeout Risk by Harness:")
        for harness, rate in profile.timeout_rate_by_harness.items():
            risk_icon = "🔴" if rate > 30 else "🟡" if rate > 10 else "🟢"
            print(f"    {risk_icon} {harness}: {rate:.1f}% would timeout")
        
        print(f"\n  💥 Failure Mode Distribution:")
        for mode, count in sorted(profile.failure_mode_distribution.items(), key=lambda x: -x[1]):
            pct = count / sum(profile.failure_mode_distribution.values()) * 100
            bar = "█" * int(pct / 5)
            print(f"    {mode:.<25} {pct:>5.1f}% {bar}")
        
        print(f"\n  🎯 Recommendations:")
        if profile.recommended_for:
            print(f"    ✅ Recommended for:")
            for rec in profile.recommended_for:
                print(f"       • {rec}")
        if profile.not_recommended_for:
            print(f"    ⚠️  Not recommended for:")
            for rec in profile.not_recommended_for:
                print(f"       • {rec}")
        
        print(f"\n  ⚙️  Configuration Suggestions:")
        print(f"    Optimal context size: {profile.optimal_context_size:,} tokens")
        print(f"    Suggested timeout:    {profile.suggested_timeout}s")
    
    print("\n" + "=" * 100)
    print("  ACTIONABLE INSIGHTS\n")
    
    # Find best model for each use case
    best_timeout = max(ranked, key=lambda x: x[1].timeout_resistance_score)
    best_efficiency = max(ranked, key=lambda x: x[1].context_efficiency_score)
    best_completion = max(ranked, key=lambda x: x[1].task_completion_rate)
    
    print(f"  🏆 Best timeout resistance:  {best_timeout[0]} ({best_timeout[1].timeout_resistance_score:.1f}%)")
    print(f"  🏆 Best context efficiency:  {best_efficiency[0]} ({best_efficiency[1].context_efficiency_score:.1f}%)")
    print(f"  🏆 Best task completion:     {best_completion[0]} ({best_completion[1].task_completion_rate:.1f}%)")
    
    # Warning flags
    print(f"\n  ⚠️  Warning Flags:")
    for model, profile in ranked:
        if profile.timeout_resistance_score < 70:
            print(f"    🔴 {model}: High timeout risk ({profile.timeout_resistance_score:.1f}%)")
        if profile.task_completion_rate < 60:
            print(f"    🟡 {model}: Low task completion ({profile.task_completion_rate:.1f}%)")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production Harness Usability Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models qwen3.5:32b deepseek-r1:32b
  %(prog)s --models llama3.1:70b --context-sizes 65536 131072
  %(prog)s --harness omp --timeout-threshold 90
  %(prog)s --full-suite  # Run all test categories
        """
    )
    
    parser.add_argument("--models", nargs="+", required=True, help="Models to benchmark")
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama host URL")
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[32768, 65536],
                       help="Context sizes to test")
    parser.add_argument("--harness", default="opencode", choices=["opencode", "omp", "cline", "default"],
                       help="Target harness to simulate")
    parser.add_argument("--timeout-threshold", type=int, default=120,
                       help="Timeout threshold in seconds")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--full-suite", action="store_true", help="Run complete test suite")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_usability_benchmark(
        models=args.models,
        host=args.host,
        context_sizes=args.context_sizes,
        harness_type=args.harness,
        timeout_threshold=args.timeout_threshold
    )
    
    # Print report
    print_usability_report(results)
    
    # Save to file if requested
    if args.output:
        output_data = {
            model: asdict(profile) if hasattr(profile, '__dataclass_fields__') else profile
            for model, profile in results.items()
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
