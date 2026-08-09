"""Offline tests for benchmark_quality evaluators (no server)."""
import benchmark_quality as bq


def test_extract_number_anchored_first():
    # anchored pattern ("= N" / "answer: N") wins even if later digits exist
    assert bq.extract_number("result = 42, then 7") == 42.0
    assert bq.extract_number("the answer: 391 tokens") == 391.0
    assert bq.extract_number("80 km/h") == 80.0
    assert bq.extract_number("no numbers here") is None


def test_eval_exact_non_string_is_false():
    res = bq.eval_exact(12345, "12345")
    assert res["pass"] is False
    res = bq.eval_exact("hello world", "hello")
    assert res["pass"] is True


def test_extract_code_block_none_on_missing():
    assert bq.extract_code_block("no code here") is None
    assert bq.extract_code_block("```python\nx = 1\n```") == "x = 1"


def test_eval_tool_call_wrong_tool_no_typo():
    res = bq.eval_tool_call(
        '{"tool": "web_search", "args": {"query": "x"}}',
        expected_tool="read_file",
        expected_args={"path": "a.py"},
    )
    assert res["pass"] is False
    assert "Wrong tool" in res["reason"]  # not "Wro ng tool"


def test_eval_tool_call_one_way_substring_only():
    # 'q' must NOT match expected 'quantum' (the old bidirectional bug)
    res = bq.eval_tool_call(
        '{"tool": "search_web", "args": {"query": "q"}}',
        expected_tool="search_web",
        expected_args={"query": "quantum computing"},
    )
    assert res["pass"] is False
    res = bq.eval_tool_call(
        '{"tool": "search_web", "args": {"query": "quantum computing today"}}',
        expected_tool="search_web",
        expected_args={"query": "quantum computing"},
    )
    assert res["pass"] is True


def test_eval_tool_call_required_args():
    res = bq.eval_tool_call(
        '{"tool": "read_file", "args": {"content": "x"}}',
        expected_tool="read_file",
        expected_args={},
        required_args=["path"],
    )
    assert res["pass"] is False
    assert "Missing required argument" in res["reason"]


def test_parse_json_from_text_handles_prose():
    text = "Sure! Here is the call:\n```json\n{\"tool\": \"finish\", \"args\": {\"summary\": \"done\"}}\n```"
    parsed = bq.parse_json_from_text(text)
    assert parsed == {"tool": "finish", "args": {"summary": "done"}}
    assert bq.parse_json_from_text("no json") is None


def test_eval_code_execution_runs_subprocess():
    code = "def triple(x):\n    return x * 3\n"
    res = bq.eval_code_execution(code, "triple",
                                 [{"call": "triple(4)", "expected": 12}])
    assert res["pass"] is True


def test_eval_code_execution_fails_on_wrong_output():
    code = "def triple(x):\n    return x + 3\n"
    res = bq.eval_code_execution(code, "triple",
                                 [{"call": "triple(4)", "expected": 12}])
    assert res["pass"] is False