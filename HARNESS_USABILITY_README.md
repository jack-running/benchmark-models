# Production Harness Usability Benchmark

## Overview

This benchmark evaluates models for **real-world production use** in agentic frameworks like OpenCode, OMP (Open Model Protocol), Cline, and similar harnesses. 

### Key Philosophy

> **A model that's 95% accurate but times out 40% of the time is LESS usable than a model that's 85% accurate and always completes on time.**

Traditional benchmarks focus on correctness alone. This framework recognizes that **usability = correctness + reliability + efficiency**.

---

## What We Measure

### ⏱️ 1. Timeout Resistance
- **Why it matters**: Harnesses have hard timeouts (90-150s typical)
- **What we test**: Response times under various load conditions
- **Metrics**: 
  - % tasks completed without timeout
  - P95/P99 response times
  - Timeout risk per harness (opencode, omp, cline)

### 📊 2. Context Efficiency  
- **Why it matters**: Token budgets are finite; verbose models waste budget
- **What we test**: Ability to stay within token constraints
- **Metrics**:
  - Token usage vs. target range
  - Verbosity penalty score
  - Budget awareness compliance

### 🔄 3. Multi-Turn Stability
- **Why it matters**: Real work happens over conversation turns
- **What we test**: Consistency across iterative workflows
- **Metrics**:
  - Turn-to-turn coherence
  - Error recovery success rate
  - Context retention across turns

### 🎯 4. Task Completion Rate
- **Why it matters**: Partial completions waste user time
- **What we test**: End-to-end goal achievement
- **Metrics**:
  - % tasks fully completed
  - Steps completed / total steps
  - Abandonment rate

### 💥 5. Failure Mode Analysis
- **Why it matters**: How a model fails determines debugging difficulty
- **What we test**: Categorization of failure types
- **Failure modes tracked**:
  - `timeout` - Exceeded time limit
  - `context_exceeded` - Hit context window limit
  - `malformed_output` - Invalid JSON/code structure
  - `hallucinated_tool` - Called non-existent tool
  - `refusal_safety` - Overly cautious refusal
  - `incomplete_response` - Cut off mid-response
  - `infinite_loop` - Repetitive/looping behavior

### 🧠 6. Long-Context Retention (64k-200k)
- **Why it matters**: Large projects require large contexts
- **What we test**: Recall from early conversation turns
- **Metrics**:
  - Performance degradation at context size
  - Information recall accuracy
  - Optimal context size per model

---

## Test Categories

### Timeout Stress Tests
Realistic tasks designed to expose timeout risks:

| Test ID | Name | Risk Level | Description |
|---------|------|------------|-------------|
| `timeout_01` | concise_code_fix | Low | Simple bug fix, should be fast |
| `timeout_02` | verbose_explanation_trap | Medium | Explicit word limit constraint |
| `timeout_03` | multi_file_refactor | High | Complex output, many tokens needed |
| `timeout_04` | iterative_debugging | Low | Multi-turn debugging session |
| `timeout_05` | context_overflow_scenario | Extreme | Simulated large context |

### Context Efficiency Tests
Measures token budget awareness:

| Test ID | Name | Target Tokens | Constraint |
|---------|------|---------------|------------|
| `efficiency_01` | token_budget_awareness | 400-500 | Stay under limit |
| `efficiency_02` | progressive_summarization | 80-150 | Exactly 3 sentences |
| `efficiency_03` | code_golf_challenge | 50-150 | Max 5 lines of code |

### Harness Simulation Tests
Mimics real harness workflows:

#### OpenCode Mode
- File editing with structured XML prompts
- Multi-step file creation sequences
- Tool call requirements

#### OMP Mode  
- Directory exploration tasks
- Structured analysis output
- Shorter timeout constraints (90s)

### Long Context Retention Tests
Tests at multiple context sizes:
- 32k tokens (medium complexity)
- 65k tokens (large projects)
- 131k tokens (full codebases)
- 200k tokens (maximum Ollama)

---

## Usage

### Basic Usage

```bash
# Benchmark two models with default settings
python benchmark_harness_usability.py --models qwen3.5:32b deepseek-r1:32b

# Test specific context sizes
python benchmark_harness_usability.py \
  --models llama3.1:70b \
  --context-sizes 65536 131072

# Simulate OMP harness (stricter timeouts)
python benchmark_harness_usability.py \
  --models qwen3.5:32b \
  --harness omp \
  --timeout-threshold 90

# Save results to JSON
python benchmark_harness_usability.py \
  --models qwen3.5:32b \
  --output results.json
```

### Advanced Options

```bash
# Full test suite (all categories)
python benchmark_harness_usability.py \
  --models qwen3.5:32b \
  --full-suite

# Custom Ollama host
python benchmark_harness_usability.py \
  --models qwen3.5:32b \
  --host http://localhost:11434

# Aggressive timeout testing
python benchmark_harness_usability.py \
  --models qwen3.5:32b \
  --timeout-threshold 60
```

---

## Output Interpretation

### Summary Table

```
Rank  Model                            Overall   Timeout  Complete  Efficient  OptimalCtx
-------------------------------------------------------------------------------------------
1     qwen3.5:32b                       87.2%     95.0%     85.0%      82.5%     65,536
2     deepseek-r1:32b                   72.4%     75.0%     70.0%      68.0%     32,768
```

**Columns:**
- **Overall**: Weighted composite score (35% timeout + 30% completion + 20% efficiency + 15% failures)
- **Timeout**: % of tasks completed without timing out
- **Complete**: % of tasks fully completed (all steps)
- **Efficient**: Average token efficiency score
- **OptimalCtx**: Best performing context size for this model

### Detailed Analysis Sections

#### Response Time Statistics
```
⏱️  Response Time:
    Average:  23.45s
    P95:      45.20s
    P99:      67.80s
```
**Use P95 for timeout configuration** (covers 95% of cases)

#### Timeout Risk by Harness
```
🚨 Timeout Risk by Harness:
    🟢 opencode: 5.0% would timeout
    🟡 omp: 15.0% would timeout
    🔴 cline: 35.0% would timeout
```
- 🟢 Green (<10%): Safe for production
- 🟡 Yellow (10-30%): Monitor closely
- 🔴 Red (>30%): High risk, consider alternatives

#### Failure Mode Distribution
```
💥 Failure Mode Distribution:
    success.................. 75.0% ███████████████
    timeout.................. 15.0% ███
    malformed_output.......... 5.0% █
    incomplete_response....... 5.0% █
```

#### Recommendations
```
🎯 Recommendations:
    ✅ Recommended for:
       • Time-sensitive production workflows
       • High-volume token-constrained scenarios
       • Complex multi-step tasks
    
    ⚠️  Not recommended for:
       • Real-time interactive applications
       • Token-budget-constrained environments

⚙️  Configuration Suggestions:
    Optimal context size: 65,536 tokens
    Suggested timeout:    68s
```

---

## Scoring Methodology

### Overall Usability Score

Weighted composite of four dimensions:

```python
overall_score = (
    timeout_resistance * 0.35 +      # Most important
    task_completion_rate * 0.30 +    
    context_efficiency * 0.20 +      
    failure_prevention * 0.15        
)
```

**Rationale**: A model that times out frequently is unusable regardless of accuracy.

### Timeout Resistance Score
```python
timeout_resistance = (total_tasks - timeout_count) / total_tasks * 100
```

### Context Efficiency Score
```python
if actual_tokens < target_min:
    efficiency = 100 - ((target_min - actual) / target_min * 50)
elif actual_tokens > target_max:
    efficiency = 100 - ((actual - target_max) / target_max * 100)
else:
    efficiency = 100  # Perfect
```

### Task Completion Rate
Simple percentage of fully completed tasks:
```python
completion_rate = successful_tasks / total_tasks * 100
```

---

## Integration with Existing Benchmarks

This harness usability benchmark **complements** (doesn't replace) the existing benchmarks:

| Benchmark | Focus | Use Together When |
|-----------|-------|-------------------|
| `benchmark_quality.py` | Correctness, accuracy | Selecting models for quality-critical tasks |
| `benchmark_agentic.py` | Tool calling, multi-turn workflows | Evaluating agent capabilities |
| `benchmark_harness_usability.py` | **Production readiness** | **Deploying to real users** |

### Recommended Workflow

1. **Initial Screening**: Run `benchmark_quality.py --quick` to filter out poor performers
2. **Agent Capability**: Run `benchmark_agentic.py` on remaining candidates
3. **Production Readiness**: Run `benchmark_harness_usability.py` on finalists
4. **Final Selection**: Choose based on weighted priorities for your use case

---

## Best Practices

### For Model Selection

1. **Prioritize timeout resistance** for interactive applications
2. **Check optimal context size** - don't assume bigger is better
3. **Review failure modes** - some are easier to handle than others
4. **Consider harness-specific performance** - a model good for OpenCode might fail in OMP

### For Configuration

1. **Set timeouts using P95**, not average (covers 95% of cases)
2. **Add 50% buffer** to P95 for safety margin
3. **Test at your expected context size** - performance varies significantly
4. **Monitor timeout rates in production** - adjust if >10%

### For Interpretation

1. **Look at distributions**, not just averages
2. **Failure mode patterns** reveal model personality
3. **Context size sweet spots** vary by model architecture
4. **Harness-specific results** may differ - test your target

---

## Example Decision Matrix

```
Scenario: Building a production coding assistant with OpenCode

Requirements:
- Must respond within 120s (OpenCode timeout)
- Handles 50+ requests/hour
- Works with large codebases (64k+ context)

Candidate Models:
┌─────────────┬──────────┬────────────┬─────────────┬──────────────┐
│ Model       │ Usability│ Timeout    │ Optimal Ctx│ Verdict      │
│             │ Score    │ @120s      │             │              │
├─────────────┼──────────┼────────────┼─────────────┼──────────────┤
│ qwen3.5:32b │ 87%      │ 95%        │ 65k         │ ✅ SELECT    │
│ deepseek:32b│ 72%      │ 75%        │ 32k         │ ⚠️  Backup   │
│ llama3:70b  │ 68%      │ 65%        │ 131k        │ ❌ Reject    │
└─────────────┴──────────┴────────────┴─────────────┴──────────────┘

Decision Rationale:
- qwen3.5:32b has best timeout resistance (critical for UX)
- deepseek:32b acceptable but needs monitoring
- llama3:70b too slow despite larger context capability
```

---

## Limitations

1. **Simulated contexts**: Long-context tests use simulated fill, not real conversation history
2. **Single-turn focus**: Some tests are single-turn; real work is multi-turn
3. **No network effects**: Doesn't account for network latency variations
4. **Static workloads**: Real harnesses have dynamic, unpredictable patterns

## Future Enhancements

- [ ] Real multi-turn conversation simulation
- [ ] Network latency injection testing
- [ ] Concurrent request load testing
- [ ] Cost-per-task analysis
- [ ] Integration with actual harnesses (OpenCode, OMP)
- [ ] Historical trend tracking

---

## Troubleshooting

### "All models show high timeout rates"
- Check Ollama server resources (CPU/RAM)
- Reduce context size
- Increase timeout threshold
- Consider model quantization

### "Context efficiency scores are low"
- Models may need explicit token budget instructions
- Try adding "Be concise" to system prompts
- Consider smaller/faster models

### "Results vary between runs"
- Expected due to model stochasticity
- Run multiple iterations and average
- Use `temperature=0` for more deterministic results

---

## Contributing

To add new test categories:

1. Create test function following existing pattern
2. Add to test suite aggregation in `run_usability_benchmark()`
3. Update evaluation logic if needed
4. Document in this README

To add harness support:

1. Add harness config to `HARNESS_TIMEOUTS`
2. Implement `get_harness_simulation_tests()` for new harness
3. Update CLI choices
4. Add harness-specific metrics if needed
