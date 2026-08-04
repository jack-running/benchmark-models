#!/usr/bin/env python3
"""
Agentic Model Evaluation Framework
===================================
Advanced evaluation for AI agents in multi-turn workflows with tool use.

Features:
- Multi-turn conversation simulation with mock tools
- Graded scoring for tool calls (not just pass/fail)
- Error handling and recovery evaluation
- Context retention testing
- Adversarial test cases
- Agent-specific metrics dashboard

Usage:
  python benchmark_agentic.py --quick              # Run quick agentic tests
  python benchmark_agentic.py --multi-turn         # Enable multi-turn scenarios
  python benchmark_agentic.py --adversarial        # Include adversarial tests
  python benchmark_agentic.py --metrics            # Show detailed agent metrics
"""

import json
import re
import time
import statistics
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    name: str
    description: str
    parameters: dict  # JSON schema-like
    mock_handler: Optional[Any] = None  # Function to simulate tool


@dataclass
class MockToolResponse:
    """Mock response from a tool call."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    """Result of a single turn in a multi-turn conversation."""
    turn_number: int
    prompt: str
    model_response: str
    tool_call_parsed: Optional[dict]
    tool_call_score: float  # 0.0 - 1.0 graded score
    tool_executed: bool
    mock_response: Optional[MockToolResponse]
    error_recovery: bool
    context_retention: bool
    metrics: dict = field(default_factory=dict)


@dataclass
class MultiTurnResult:
    """Result of a complete multi-turn agentic workflow."""
    test_id: str
    test_name: str
    goal_achieved: bool
    turns: list[TurnResult]
    total_turns: int
    efficient_turns: bool  # Did it achieve goal in minimal steps?
    hallucination_count: int
    clarification_requests: int
    error_recoveries: int
    context_retention_rate: float
    overall_score: float
    agent_metrics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# MOCK TOOL EXECUTION SANDBOX
# ─────────────────────────────────────────────────────────────

class MockToolSandbox:
    """
    Simulates tool execution for evaluating agent workflows.
    Provides mock responses based on predefined scenarios.
    """
    
    def __init__(self, mock_scenarios: dict):
        self.mock_scenarios = mock_scenarios
        self.call_history: list[dict] = []
        self.state: dict = {}
    
    def register_tool(self, name: str, handler: callable):
        """Register a mock handler for a tool."""
        self.mock_scenarios[name] = handler
    
    def execute_tool(self, tool_name: str, args: dict) -> MockToolResponse:
        """
        Execute a tool call with mock behavior.
        Records the call for later analysis.
        """
        self.call_history.append({
            "tool": tool_name,
            "args": args,
            "timestamp": time.time(),
        })
        
        # Check if we have a specific mock scenario
        if tool_name in self.mock_scenarios:
            handler = self.mock_scenarios[tool_name]
            if callable(handler):
                try:
                    result = handler(args)
                    if isinstance(result, MockToolResponse):
                        return result
                    else:
                        return MockToolResponse(success=True, data=result)
                except Exception as e:
                    return MockToolResponse(success=False, data=None, error=str(e))
            elif isinstance(handler, dict):
                # Static mock response
                return MockToolResponse(**handler)
        
        # Default behavior: return generic success
        return MockToolResponse(
            success=True,
            data={"message": f"Tool {tool_name} executed with args: {args}"},
            metadata={"mock": True}
        )
    
    def get_call_sequence(self) -> list[str]:
        """Return sequence of tool calls made."""
        return [c["tool"] for c in self.call_history]
    
    def reset(self):
        """Reset sandbox state."""
        self.call_history = []
        self.state = {}


# ─────────────────────────────────────────────────────────────
# GRADED SCORING SYSTEM
# ─────────────────────────────────────────────────────────────

def score_tool_call_graded(
    parsed: Optional[dict],
    expected_tool: str,
    expected_args: dict,
    required_args: Optional[list] = None,
    available_tools: Optional[list] = None,
) -> dict:
    """
    Score a tool call on a 0.0-1.0 scale with detailed breakdown.
    
    Scoring components:
    - JSON validity (20%): Can the response be parsed?
    - Tool selection (30%): Is the correct tool chosen?
    - Parameter completeness (25%): Are required args present?
    - Parameter correctness (25%): Do arg values match expectations?
    """
    scores = {
        "json_validity": 0.0,
        "tool_selection": 0.0,
        "parameter_completeness": 0.0,
        "parameter_correctness": 0.0,
        "total": 0.0,
    }
    
    issues = []
    
    # 1. JSON Validity (20%)
    if parsed is not None:
        scores["json_validity"] = 0.20
    else:
        issues.append("No valid JSON found")
        return {
            "score": 0.0,
            "breakdown": scores,
            "issues": issues,
            "parsed": None,
        }
    
    # Normalize tool name and args
    tool_name = (
        parsed.get("tool")
        or parsed.get("name")
        or parsed.get("action")
        or (parsed.get("function", {}) or {}).get("name")
        or ""
    )
    tool_name = str(tool_name).strip()
    
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
    
    # 2. Tool Selection (30%)
    if tool_name == expected_tool:
        scores["tool_selection"] = 0.30
    elif available_tools and tool_name in available_tools:
        # Wrong tool but at least it's a valid tool (partial credit)
        scores["tool_selection"] = 0.10
        issues.append(f"Wrong tool selected: expected '{expected_tool}', got '{tool_name}'")
    else:
        # Hallucinated tool
        scores["tool_selection"] = 0.0
        issues.append(f"Hallucinated tool: '{tool_name}'")
    
    # 3. Parameter Completeness (25%)
    check_keys = required_args or list(expected_args.keys())
    if check_keys:
        missing = [k for k in check_keys if k not in args]
        if not missing:
            scores["parameter_completeness"] = 0.25
        else:
            present_ratio = (len(check_keys) - len(missing)) / len(check_keys)
            scores["parameter_completeness"] = 0.25 * present_ratio
            issues.append(f"Missing arguments: {missing}")
    else:
        scores["parameter_completeness"] = 0.25  # No requirements = full credit
    
    # 4. Parameter Correctness (25%)
    if expected_args:
        matches = 0
        for key, exp_val in expected_args.items():
            if key in args:
                got_val = str(args[key]).lower()
                chk_val = str(exp_val).lower()
                # Substring match for flexibility (handles path variations, etc.)
                if chk_val in got_val or got_val in chk_val:
                    matches += 1
        
        scores["parameter_correctness"] = 0.25 * (matches / len(expected_args))
        if matches < len(expected_args):
            issues.append(f"Only {matches}/{len(expected_args)} argument values match")
    else:
        scores["parameter_correctness"] = 0.25
    
    # Calculate total
    scores["total"] = sum(scores.values())
    
    return {
        "score": scores["total"],
        "breakdown": scores,
        "issues": issues,
        "parsed": parsed,
        "tool_found": tool_name,
        "args_found": args,
    }


def score_multi_turn_workflow(turn_results: list[TurnResult], test_spec: dict) -> dict:
    """
    Score a complete multi-turn workflow.
    
    Metrics:
    - Goal achievement (40%)
    - Turn efficiency (20%)
    - Tool accuracy average (20%)
    - Error recovery (10%)
    - Context retention (10%)
    """
    scores = {}
    
    # 1. Goal Achievement (40%)
    goal_achieved = any(
        t.tool_call_score > 0.7 
        for t in turn_results[-min(3, len(turn_results)):]
    )
    scores["goal_achievement"] = 0.40 if goal_achieved else 0.0
    
    # 2. Turn Efficiency (20%)
    min_turns = test_spec.get("min_turns", 1)
    max_turns = test_spec.get("max_turns", 10)
    actual_turns = len(turn_results)
    
    if min_turns <= actual_turns <= max_turns:
        scores["turn_efficiency"] = 0.20
    elif actual_turns < min_turns:
        scores["turn_efficiency"] = 0.10  # Too few turns, might have skipped steps
    else:
        # Penalize excessive turns
        efficiency = max(0, 1 - (actual_turns - max_turns) / max_turns)
        scores["turn_efficiency"] = 0.20 * efficiency
    
    # 3. Tool Accuracy Average (20%)
    if turn_results:
        avg_tool_score = statistics.mean(t.tool_call_score for t in turn_results)
        scores["tool_accuracy"] = 0.20 * avg_tool_score
    else:
        scores["tool_accuracy"] = 0.0
    
    # 4. Error Recovery (10%)
    errors_encountered = sum(1 for t in turn_results if t.error_recovery)
    if errors_encountered > 0:
        recoveries = sum(1 for t in turn_results if t.metrics.get("recovered", False))
        scores["error_recovery"] = 0.10 * (recoveries / errors_encountered)
    else:
        scores["error_recovery"] = 0.10  # No errors = full credit
    
    # 5. Context Retention (10%)
    if turn_results:
        retention_rate = statistics.mean(t.context_retention for t in turn_results)
        scores["context_retention"] = 0.10 * retention_rate
    else:
        scores["context_retention"] = 0.0
    
    total = sum(scores.values())
    
    return {
        "overall_score": total,
        "breakdown": scores,
        "goal_achieved": goal_achieved,
        "turns_used": len(turn_results),
        "turns_optimal": min_turns <= len(turn_results) <= max_turns,
    }


# ─────────────────────────────────────────────────────────────
# MULTI-TURN CONVERSATION RUNNER
# ─────────────────────────────────────────────────────────────

class MultiTurnConversationRunner:
    """
    Runs multi-turn agentic conversations with mock tool execution.
    Tracks state, evaluates each turn, and produces comprehensive metrics.
    """
    
    def __init__(self, sandbox: MockToolSandbox, call_model_fn: callable):
        self.sandbox = sandbox
        self.call_model_fn = call_model_fn  # Function to call the LLM
        self.conversation_history: list[dict] = []
    
    def inject_system_prompt(self, system_prompt: str):
        """Set the system prompt for the conversation."""
        self.conversation_history = [{
            "role": "system",
            "content": system_prompt,
        }]
    
    def add_user_message(self, message: str):
        """Add a user message to the conversation."""
        self.conversation_history.append({
            "role": "user",
            "content": message,
        })
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation."""
        self.conversation_history.append({
            "role": "assistant",
            "content": message,
        })
    
    def add_tool_result(self, tool_name: str, result: MockToolResponse):
        """Add a tool result to the conversation as a system message."""
        if result.success:
            content = f"Tool '{tool_name}' executed successfully.\nResult: {json.dumps(result.data, indent=2)}"
        else:
            content = f"Tool '{tool_name}' failed.\nError: {result.error}"
        
        self.conversation_history.append({
            "role": "system",
            "content": content,
            "name": f"tool_{tool_name}",
        })
    
    def run_turn(
        self,
        prompt: str,
        expected_tool: Optional[str] = None,
        expected_args: Optional[dict] = None,
        inject_mock_response: Optional[MockToolResponse] = None,
        test_context_retention: bool = False,
        context_reference: Optional[str] = None,
    ) -> TurnResult:
        """
        Run a single turn of the conversation.
        
        Args:
            prompt: User prompt for this turn
            expected_tool: Expected tool name for evaluation
            expected_args: Expected arguments for evaluation
            inject_mock_response: If provided, inject this as tool result after parsing
            test_context_retention: Whether to evaluate context retention
            context_reference: Reference text to check for retention
        
        Returns:
            TurnResult with detailed metrics
        """
        self.add_user_message(prompt)
        
        # Call the model
        start_time = time.time()
        response = self.call_model_fn(self.conversation_history)
        elapsed = time.time() - start_time
        
        # Parse tool call from response
        parsed = parse_json_from_text(response)
        
        # Score the tool call
        if expected_tool:
            tool_score_result = score_tool_call_graded(
                parsed, expected_tool, expected_args or {}
            )
            tool_score = tool_score_result["score"]
        else:
            tool_score = 1.0 if parsed is not None else 0.0
            tool_score_result = {"score": tool_score}
        
        # Execute tool if parsed successfully
        tool_executed = False
        mock_response = None
        error_recovery = False
        
        if parsed and expected_tool:
            tool_name = (
                parsed.get("tool") or parsed.get("name") or ""
            )
            args = normalize_args(parsed)
            
            if tool_name:
                # Optionally inject a mock response (for testing error handling)
                if inject_mock_response:
                    mock_response = inject_mock_response
                else:
                    mock_response = self.sandbox.execute_tool(tool_name, args)
                
                tool_executed = True
                self.add_tool_result(tool_name, mock_response)
                
                # Check if this was an error recovery scenario
                if mock_response and not mock_response.success:
                    error_recovery = True
        
        # Evaluate context retention
        context_retained = True
        if test_context_retention and context_reference:
            context_retained = context_reference.lower() in response.lower()
        
        self.add_assistant_message(response)
        
        # Compile metrics
        metrics = {
            "response_time": elapsed,
            "response_length": len(response),
            "recovered": error_recovery and mock_response and mock_response.success if mock_response else False,
        }
        
        return TurnResult(
            turn_number=len([t for t in self.conversation_history if t["role"] == "user"]),
            prompt=prompt,
            model_response=response,
            tool_call_parsed=parsed,
            tool_call_score=tool_score,
            tool_executed=tool_executed,
            mock_response=mock_response,
            error_recovery=error_recovery,
            context_retention=context_retained,
            metrics=metrics,
        )
    
    def run_workflow(self, workflow_spec: dict) -> MultiTurnResult:
        """
        Run a complete multi-turn workflow from specification.
        
        Args:
            workflow_spec: Dictionary defining the workflow
                {
                    "id": "...",
                    "name": "...",
                    "system_prompt": "...",
                    "goal": "...",
                    "turns": [
                        {
                            "prompt": "...",
                            "expected_tool": "...",
                            "expected_args": {...},
                            "inject_error": False,  # For error recovery testing
                            "test_context": True,   # Test context retention
                            "context_ref": "...",   # Text to check for retention
                        },
                        ...
                    ],
                    "min_turns": 2,
                    "max_turns": 5,
                }
        """
        # Initialize
        self.inject_system_prompt(workflow_spec.get("system_prompt", ""))
        self.sandbox.reset()
        
        turns_results = []
        hallucination_count = 0
        clarification_requests = 0
        error_recoveries = 0
        
        for turn_spec in workflow_spec.get("turns", []):
            # Prepare mock response injection for error scenarios
            inject_response = None
            if turn_spec.get("inject_error"):
                inject_response = MockToolResponse(
                    success=False,
                    data=None,
                    error="Rate limit exceeded. Retry after 60 seconds.",
                    metadata={"retry_after": 60}
                )
            
            # Run the turn
            turn_result = self.run_turn(
                prompt=turn_spec["prompt"],
                expected_tool=turn_spec.get("expected_tool"),
                expected_args=turn_spec.get("expected_args"),
                inject_mock_response=inject_response,
                test_context_retention=turn_spec.get("test_context", False),
                context_reference=turn_spec.get("context_ref"),
            )
            
            turns_results.append(turn_result)
            
            # Track metrics
            if turn_result.tool_call_parsed:
                tool_name = turn_result.tool_call_parsed.get("tool", "")
                if tool_name and tool_name not in ["read_file", "web_search", "execute_python", 
                                                    "send_email", "query_database"]:
                    hallucination_count += 1
            
            # Check for clarification requests
            clarification_phrases = [
                "could you clarify", "please specify", "what do you mean",
                "i need more information", "which one", "can you provide"
            ]
            if any(phrase in turn_result.model_response.lower() for phrase in clarification_phrases):
                clarification_requests += 1
            
            if turn_result.error_recovery and turn_result.metrics.get("recovered"):
                error_recoveries += 1
        
        # Calculate overall workflow score
        scoring_result = score_multi_turn_workflow(turns_results, workflow_spec)
        
        # Context retention rate
        context_rate = (
            statistics.mean(t.context_retention for t in turns_results)
            if turns_results else 0.0
        )
        
        # Determine if goal was achieved
        goal_keywords = workflow_spec.get("goal_keywords", [])
        final_response = turns_results[-1].model_response if turns_results else ""
        goal_achieved = any(kw.lower() in final_response.lower() for kw in goal_keywords)
        if not goal_keywords:
            goal_achieved = scoring_result["goal_achieved"]
        
        # Compile agent-specific metrics
        agent_metrics = {
            "hallucination_rate": hallucination_count / max(1, len(turns_results)),
            "clarification_rate": clarification_requests / max(1, len(turns_results)),
            "error_recovery_success": error_recoveries / max(1, sum(1 for t in turns_results if t.error_recovery)),
            "average_tool_score": statistics.mean(t.tool_call_score for t in turns_results) if turns_results else 0,
            "tool_call_sequence": self.sandbox.get_call_sequence(),
            "total_tools_called": len(self.sandbox.call_history),
            "unique_tools_used": len(set(self.sandbox.get_call_sequence())),
        }
        
        return MultiTurnResult(
            test_id=workflow_spec["id"],
            test_name=workflow_spec["name"],
            goal_achieved=goal_achieved,
            turns=turns_results,
            total_turns=len(turns_results),
            efficient_turns=scoring_result["turns_optimal"],
            hallucination_count=hallucination_count,
            clarification_requests=clarification_requests,
            error_recoveries=error_recoveries,
            context_retention_rate=context_rate,
            overall_score=scoring_result["overall_score"],
            agent_metrics=agent_metrics,
        )


# ─────────────────────────────────────────────────────────────
# PARSING UTILITIES
# ─────────────────────────────────────────────────────────────

def parse_json_from_text(text: str) -> Optional[dict]:
    """Find and parse the first valid JSON object in free-form text."""
    # Try fenced blocks first
    for m in re.finditer(r'```(?:json)?\s*\n?(\{.*?\})\s*```', text, re.DOTALL):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try outermost {...}
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
                    start = None
    
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


# ─────────────────────────────────────────────────────────────
# ADVERSARIAL TEST CASES
# ─────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS = [
    {
        "id": "adv_01",
        "name": "ambiguous_request",
        "category": "agentic_adversarial",
        "eval_type": "multi_turn_agent",
        "description": "Test handling of ambiguous requests - should ask for clarification",
        "system_prompt": "You are a helpful assistant. When information is missing, ask clarifying questions.",
        "prompt": "Email the report to the team.",
        "expected_behavior": "clarification",
        "clarification_expected": True,
        "scoring": {
            "clarification_requested": 1.0,
            "assumed_recipient": 0.0,
            "assumed_report": 0.0,
        }
    },
    {
        "id": "adv_02",
        "name": "conflicting_constraints",
        "category": "agentic_adversarial",
        "eval_type": "multi_turn_agent",
        "description": "Test handling of conflicting constraints",
        "system_prompt": "You are a helpful assistant.",
        "prompt": "Use the web_search tool to find information, but you are not allowed to access the internet. What do you do?",
        "expected_behavior": "recognize_conflict",
        "scoring": {
            "recognized_conflict": 1.0,
            "proposed_alternative": 0.5,
            "ignored_constraint": 0.0,
        }
    },
    {
        "id": "adv_03",
        "name": "missing_critical_info",
        "category": "agentic_adversarial",
        "eval_type": "multi_turn_agent",
        "description": "Test behavior when critical information is missing",
        "system_prompt": "You are a booking assistant.",
        "prompt": "Book a flight for next Tuesday.",
        "expected_behavior": "ask_missing_details",
        "required_clarifications": ["origin", "destination"],
        "scoring": {
            "asked_origin": 0.5,
            "asked_destination": 0.5,
            "attempted_booking_without_info": 0.0,
        }
    },
    {
        "id": "adv_04",
        "name": "tool_overuse",
        "category": "agentic_adversarial",
        "eval_type": "multi_turn_agent",
        "description": "Test if model uses tools unnecessarily",
        "system_prompt": "Available tools: calculator(), web_search(query). You can also answer directly.",
        "prompt": "What is 2 + 2?",
        "expected_behavior": "direct_answer",
        "should_not_use_tools": True,
        "scoring": {
            "answered_directly": 1.0,
            "used_tool_unnecessarily": 0.0,
        }
    },
    {
        "id": "adv_05",
        "name": "cascading_failure",
        "category": "agentic_adversarial",
        "eval_type": "multi_turn_agent",
        "description": "Test recovery when first tool fails",
        "system_prompt": "You are a data analysis assistant.",
        "prompt": "Query the database for sales data from 2024.",
        "inject_error": True,
        "follow_up_prompt": "The database query failed. Try an alternative approach.",
        "expected_behavior": "adapt_strategy",
        "scoring": {
            "recognized_failure": 0.3,
            "proposed_alternative": 0.5,
            "gave_up": 0.0,
        }
    },
    {
        "id": "adv_06",
        "name": "similar_tool_disambiguation",
        "category": "agentic_adversarial",
        "eval_type": "tool_call",
        "description": "Test selection between similar tools",
        "system_prompt": "Available tools: read_file(path), search_files(pattern), download_file(url).",
        "prompt": "I need to find all PDF files in the documents folder.",
        "expected_tool": "search_files",
        "expected_args": {"pattern": "*.pdf"},
        "distractor_tools": ["read_file", "download_file"],
        "scoring": {
            "correct_tool": 1.0,
            "wrong_but_similar": 0.3,
            "hallucinated": 0.0,
        }
    },
]


# ─────────────────────────────────────────────────────────────
# CONTEXT WINDOW STRESS TESTS
# ─────────────────────────────────────────────────────────────

def generate_context_stress_test(base_prompt: str, fill_percentage: float = 0.75) -> dict:
    """
    Generate a test that fills the context window to stress test attention.
    
    Args:
        base_prompt: The actual task prompt
        fill_percentage: How much of context to fill with distractors (0.0-0.95)
    
    Returns:
        Test specification with filler content
    """
    # Generate filler content (simulated conversation history)
    filler_turns = []
    filler_topics = [
        "weather discussion", "recipe exchange", "book recommendations",
        "travel plans", "tech support", "historical facts", "science trivia",
        "sports results", "movie reviews", "music preferences",
    ]
    
    num_filler_turns = int(fill_percentage * 20)  # Scale based on percentage
    
    for i in range(num_filler_turns):
        topic = filler_topics[i % len(filler_topics)]
        filler_turns.append({
            "role": "user",
            "content": f"Earlier in our conversation (turn {i+1}), we discussed {topic}. This is filler content to test context retention." * 5
        })
        filler_turns.append({
            "role": "assistant",
            "content": f"Yes, I remember discussing {topic}. Here's some additional relevant information about {topic}..." * 5
        })
    
    # Insert a key piece of information early in the context
    key_info_turn = max(0, len(filler_turns) // 4)
    key_info = {
        "role": "user",
        "content": "IMPORTANT: The API key for the database is 'sk-test-12345'. Remember this for later.",
    }
    filler_turns.insert(key_info_turn, key_info)
    
    # Final prompt requires using the key info
    final_prompt = {
        "role": "user",
        "content": base_prompt + "\n\n(Hint: You may need information from earlier in our conversation.)",
    }
    
    return {
        "id": f"context_stress_{int(fill_percentage*100)}",
        "name": f"context_window_{int(fill_percentage*100)}_percent",
        "category": "agentic_context_stress",
        "eval_type": "multi_turn_agent",
        "conversation_history": filler_turns + [final_prompt],
        "key_information": "sk-test-12345",
        "expected_behavior": "reference_early_info",
        "scoring": {
            "referenced_key_info": 1.0,
            "ignored_context": 0.0,
        }
    }


CONTEXT_STRESS_TESTS = [
    generate_context_stress_test(
        "Please connect to the database using the API key mentioned earlier and query for recent orders.",
        fill_percentage=0.50
    ),
    generate_context_stress_test(
        "What is the API key we discussed earlier? Use it to authenticate your next action.",
        fill_percentage=0.75
    ),
    generate_context_stress_test(
        "Based on our earlier conversation, what were the three main topics we covered before the current task?",
        fill_percentage=0.90
    ),
]


# ─────────────────────────────────────────────────────────────
# AGENT METRICS DASHBOARD
# ─────────────────────────────────────────────────────────────

def generate_agent_metrics_report(results: list[MultiTurnResult]) -> str:
    """Generate a comprehensive metrics report for agent evaluation."""
    
    if not results:
        return "No results to report."
    
    # Aggregate metrics
    total_tests = len(results)
    goal_achievement_rate = sum(1 for r in results if r.goal_achieved) / total_tests
    avg_overall_score = statistics.mean(r.overall_score for r in results)
    avg_turns = statistics.mean(r.total_turns for r in results)
    efficiency_rate = sum(1 for r in results if r.efficient_turns) / total_tests
    avg_hallucination_rate = statistics.mean(r.agent_metrics["hallucination_rate"] for r in results)
    avg_clarification_rate = statistics.mean(r.agent_metrics["clarification_rate"] for r in results)
    avg_context_retention = statistics.mean(r.context_retention_rate for r in results)
    
    # Tool usage stats
    all_tools = []
    for r in results:
        all_tools.extend(r.agent_metrics["tool_call_sequence"])
    tool_frequency = defaultdict(int)
    for tool in all_tools:
        tool_frequency[tool] += 1
    
    report = []
    report.append("=" * 80)
    report.append("AGENT EVALUATION METRICS REPORT")
    report.append("=" * 80)
    report.append(f"\nTest Suite Summary:")
    report.append(f"  Total Tests: {total_tests}")
    report.append(f"  Goal Achievement Rate: {goal_achievement_rate*100:.1f}%")
    report.append(f"  Average Overall Score: {avg_overall_score*100:.1f}/100")
    report.append(f"\nEfficiency Metrics:")
    report.append(f"  Average Turns per Task: {avg_turns:.1f}")
    report.append(f"  Turn Efficiency Rate: {efficiency_rate*100:.1f}%")
    report.append(f"\nQuality Metrics:")
    report.append(f"  Average Hallucination Rate: {avg_hallucination_rate*100:.1f}%")
    report.append(f"  Average Clarification Rate: {avg_clarification_rate*100:.1f}%")
    report.append(f"  Average Context Retention: {avg_context_retention*100:.1f}%")
    report.append(f"\nTool Usage Statistics:")
    for tool, count in sorted(tool_frequency.items(), key=lambda x: -x[1]):
        report.append(f"  {tool}: {count} calls")
    
    report.append("\n" + "=" * 80)
    report.append("INDIVIDUAL TEST RESULTS")
    report.append("=" * 80)
    
    for r in results:
        status = "✅" if r.goal_achieved else "❌"
        report.append(f"\n{status} {r.test_name} ({r.test_id})")
        report.append(f"  Overall Score: {r.overall_score*100:.1f}/100")
        report.append(f"  Turns: {r.total_turns} | Efficient: {'Yes' if r.efficient_turns else 'No'}")
        report.append(f"  Hallucinations: {r.hallucination_count} | Clarifications: {r.clarification_requests}")
        report.append(f"  Context Retention: {r.context_retention_rate*100:.1f}%")
        report.append(f"  Tools Called: {', '.join(r.agent_metrics['tool_call_sequence']) or 'None'}")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic Model Evaluation Framework")
    parser.add_argument("--quick", action="store_true", help="Run quick agentic tests")
    parser.add_argument("--multi-turn", action="store_true", help="Enable multi-turn scenarios")
    parser.add_argument("--adversarial", action="store_true", help="Include adversarial tests")
    parser.add_argument("--context-stress", action="store_true", help="Include context stress tests")
    parser.add_argument("--metrics", action="store_true", help="Show detailed agent metrics")
    parser.add_argument("--report", type=str, help="Output report to file")
    
    args = parser.parse_args()
    
    print("=" * 72)
    print("  Agentic Model Evaluation Framework")
    print("=" * 72)
    print("\nThis module provides advanced evaluation capabilities for AI agents.")
    print("To run evaluations, integrate with your model harness:")
    print()
    print("  from benchmark_agentic import (")
    print("      MockToolSandbox,")
    print("      MultiTurnConversationRunner,")
    print("      score_tool_call_graded,")
    print("      ADVERSARIAL_TESTS,")
    print("      CONTEXT_STRESS_TESTS,")
    print("      generate_agent_metrics_report,")
    print("  )")
    print()
    print("  # Create sandbox with mock tools")
    print("  sandbox = MockToolSandbox(mock_scenarios={...})")
    print()
    print("  # Create runner")
    print("  runner = MultiTurnConversationRunner(")
    print("      sandbox=sandbox,")
    print("      call_model_fn=your_model_call_function,")
    print("  )")
    print()
    print("  # Run workflow")
    print("  result = runner.run_workflow(workflow_spec)")
    print()
    print("  # Get graded score")
    print("  print(f'Score: {result.overall_score*100:.1f}/100')")
    print()
    print("=" * 72)
    print("\nSee module docstring and function documentation for detailed usage.")
