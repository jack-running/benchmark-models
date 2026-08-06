# Agentic Model Evaluation Framework - Implementation Summary

## Overview

This implementation adds comprehensive agentic evaluation capabilities to the Ollama benchmark framework, addressing the critical gaps in evaluating models for **real-world agent workflows**.

## Files Created/Modified

### New File: `benchmark_agentic.py` (946 lines)

A complete agentic evaluation framework with:

#### Core Components

1. **Mock Tool Execution Sandbox** (`MockToolSandbox`)
   - Simulates tool execution without real API calls
   - Supports both callable handlers and static mock responses
   - Tracks call history for analysis
   - Enables testing of error scenarios

2. **Multi-Turn Conversation Runner** (`MultiTurnConversationRunner`)
   - Runs complete multi-turn agentic workflows
   - Maintains conversation state across turns
   - Injects mock tool results into conversation
   - Evaluates each turn and overall workflow

3. **Graded Scoring System**
   - `score_tool_call_graded()`: 0-100% scoring with breakdown
     - JSON validity (20%)
     - Tool selection accuracy (30%)
     - Parameter completeness (25%)
     - Parameter correctness (25%)
   - `score_multi_turn_workflow()`: Overall workflow scoring
     - Goal achievement (40%)
     - Turn efficiency (20%)
     - Tool accuracy average (20%)
     - Error recovery (10%)
     - Context retention (10%)

4. **Pre-built Test Suites**
   - `ADVERSARIAL_TESTS`: 6 adversarial test cases
     - Ambiguous requests (should ask for clarification)
     - Conflicting constraints
     - Missing critical information
     - Tool overuse detection
     - Cascading failure recovery
     - Similar tool disambiguation
   
   - `CONTEXT_STRESS_TESTS`: 3 context window stress tests
     - 50% context fill
     - 75% context fill
     - 90% context fill

5. **Metrics Dashboard**
   - `generate_agent_metrics_report()`: Comprehensive reporting
   - Agent-specific metrics:
     - Hallucination rate
     - Clarification rate
     - Error recovery success
     - Tool usage statistics
     - Context retention rate

### Modified File: `benchmark_quality.py`

Enhanced with agentic framework integration:

1. **Import Integration**
   ```python
   from benchmark_agentic import (
       MockToolSandbox,
       MultiTurnConversationRunner,
       score_tool_call_graded,
       ADVERSARIAL_TESTS,
       CONTEXT_STRESS_TESTS,
       generate_agent_metrics_report,
   )
   ```

2. **Enhanced `eval_tool_call()` Function**
   - Added `use_graded_scoring` parameter
   - When enabled, uses advanced graded scoring from `benchmark_agentic`
   - Provides partial credit instead of binary pass/fail
   - Backward compatible with existing tests

## Key Features Implemented

### 1. Multi-Turn Agentic Scenarios ✅

**Before**: Single tool call evaluation
**After**: Complete workflow evaluation with state tracking

```python
workflow_spec = {
    'id': 'customer_analysis',
    'turns': [
        {'prompt': 'Query database for sales', 'expected_tool': 'query_database'},
        {'prompt': 'Read detailed report', 'expected_tool': 'read_file'},
        {'prompt': 'Email summary to team', 'expected_tool': 'send_email'},
    ],
}
```

### 2. Mock Tool Execution Sandbox ✅

**Before**: Only checked JSON structure
**After**: Simulates actual tool execution with mock responses

```python
sandbox = MockToolSandbox({
    'query_database': lambda args: MockToolResponse(
        success=True, 
        data=[{'id': 1, 'revenue': 1000}]
    ),
})
```

### 3. Graded Scoring System ✅

**Before**: Binary pass/fail
**After**: Nuanced 0-100% scoring with detailed breakdown

| Scenario | Old Score | New Score |
|----------|-----------|-----------|
| Perfect match | 100% pass | 100% |
| Wrong but valid tool | 0% fail | 30% |
| Missing some args | 0% fail | 50-75% |
| Partial arg values | 0% fail | 60-80% |
| No JSON | 0% fail | 0% |

### 4. Adversarial Test Cases ✅

Tests designed to expose common agent failure modes:

- **Ambiguous Requests**: "Email the report" → Should ask "Which report? To whom?"
- **Conflicting Constraints**: "Search online but don't access internet" → Should recognize conflict
- **Missing Info**: "Book flight for Tuesday" → Should ask origin/destination
- **Tool Overuse**: "What is 2+2?" → Should answer directly, not use calculator
- **Cascading Failures**: First tool fails → Should adapt strategy
- **Similar Tools**: Choose between `read_file`, `search_files`, `download_file`

### 5. Context Window Stress Tests ✅

Evaluates attention degradation with long contexts:

- Fills 50%/75%/90% of context with distractors
- Hides key information early in conversation
- Tests if model can reference early information

### 6. Agent-Specific Metrics ✅

New metrics beyond simple pass/fail:

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| Hallucination Rate | % of calls to non-existent tools | 0% |
| Clarification Rate | % of turns asking for clarification | Task-dependent |
| Error Recovery Success | % of errors handled appropriately | 100% |
| Turn Efficiency | Achieved goal in optimal turns | Yes |
| Context Retention | References earlier info correctly | 100% |

### 7. Capability vs Compliance Separation ✅

The graded scoring system separates:
- **Capability**: Can it technically produce correct tool calls?
- **Compliance**: Does it follow constraints appropriately?

Example: A model that refuses a harmful request gets high compliance score even if it doesn't make the requested tool call.

## Philosophy Shifts Implemented

### 1. From "Can it call tools?" → "Can it accomplish goals using tools?" ✅

Workflows are evaluated on **goal achievement**, not just individual API calls.

### 2. From Static Tests → Dynamic Workflows ✅

The `MultiTurnConversationRunner` executes conversation loops with state tracking.

### 3. From Isolated Calls → Integrated Pipelines ✅

Tests evaluate chains of tool calls with dependencies between steps.

### 4. From Perfect Information → Uncertainty Handling ✅

Adversarial tests give incomplete prompts and reward appropriate clarification requests.

### 5. From Single Metrics → Multi-Dimensional Profiles ✅

Reports show capability profiles across multiple dimensions, not just overall score.

## Usage Examples

### Basic Integration

```python
from benchmark_agentic import (
    MockToolSandbox,
    MultiTurnConversationRunner,
    score_tool_call_graded,
)

# Create sandbox
sandbox = MockToolSandbox(mock_scenarios={...})

# Create runner
runner = MultiTurnConversationRunner(
    sandbox=sandbox,
    call_model_fn=your_model_function,
)

# Run workflow
result = runner.run_workflow(workflow_spec)
print(f"Score: {result.overall_score*100:.1f}/100")
```

### Graded Tool Call Scoring

```python
parsed = {"tool": "read_file", "args": {"path": "/data.csv"}}
result = score_tool_call_graded(
    parsed, 
    expected_tool="read_file",
    expected_args={"path": "/data/file.csv"},
)
print(f"Score: {result['score']*100:.0f}%")
# Output: Score: 83% (partial credit for path substring match)
```

### Error Recovery Testing

```python
workflow_spec = {
    'turns': [
        {
            'prompt': 'Query the database',
            'inject_error': True,  # Forces tool failure
        },
        {
            'prompt': 'The query failed. Try alternative.',
            'expected_tool': 'read_file',  # Should adapt
        },
    ],
}
```

## Testing Performed

All components tested successfully:

✅ Graded scoring system (5 test scenarios)
✅ Mock tool sandbox (dynamic and static handlers)
✅ Multi-turn conversation runner
✅ Error recovery scenarios
✅ Agent metrics report generation
✅ Adversarial test suite availability
✅ Context stress test generation
✅ Integration with benchmark_quality.py

## Next Steps for Users

To integrate with your agentic harness:

1. **Import the framework**:
   ```python
   from benchmark_agentic import *
   ```

2. **Define your mock tools**:
   ```python
   def my_tool_handler(args):
       # Your mock logic
       return MockToolResponse(success=True, data={...})
   
   sandbox = MockToolSandbox({'my_tool': my_tool_handler})
   ```

3. **Create workflow specifications**:
   ```python
   workflow = {
       'id': 'my_workflow',
       'name': 'My Agentic Task',
       'system_prompt': '...',
       'turns': [...],
   }
   ```

4. **Run and evaluate**:
   ```python
   runner = MultiTurnConversationRunner(sandbox, call_model_fn)
   result = runner.run_workflow(workflow)
   print(generate_agent_metrics_report([result]))
   ```

## Benefits for Agentic Work

This implementation enables you to:

1. **Identify weak models** before deploying them in production agent systems
2. **Compare models** across multiple dimensions relevant to agentic work
3. **Test error handling** without risking real API failures
4. **Evaluate cost efficiency** via turn optimization metrics
5. **Detect hallucinations** before they cause production issues
6. **Validate context retention** for long-running conversations
7. **Ensure safety compliance** through adversarial testing
