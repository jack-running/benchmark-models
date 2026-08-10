#!/usr/bin/env python3
"""
Agentic benchmark tasks: end-state-verified, not prose-graded.

Every verify() asserts on workspace state (file contents, subprocess results,
test exit codes) or a nonce, never on free prose. The same verifier grades
all four backends (native loop, opencode, omp, cline).
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent_workspace import SAFE_ENV, Workspace

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

RUN_TESTS_TOOLS = ["read_file", "list_dir", "grep", "write_file",
                   "edit_file", "run_tests", "finish"]


def run_pytest(ws: Workspace) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(ws.root),
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(SAFE_ENV),
        )
        return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()
    except FileNotFoundError:
        return 127, "pytest unavailable"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def py_eval(ws: Workspace, code: str) -> tuple[int, str]:
    """Run `code` inside the workspace with the safe env."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ws.root),
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(SAFE_ENV),
        )
        return proc.returncode, proc.stdout.strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""


def _episode_read_files(ep) -> list[str]:
    return [tc["arguments"].get("path") for tc in ep.tool_calls
            if tc["name"] == "read_file" and tc["arguments"].get("path")]


def check_harness_rules(task: "Task", ep) -> dict:
    """R1/R2/R3 machine-checkable adherence. Each reported separately."""
    fixture_keys = set(task.fixture.keys())
    r1_ok = True
    r1_detail = ""
    for tc in ep.tool_calls:
        if tc["name"] == "write_file" and tc["arguments"].get("path") in fixture_keys:
            r1_ok = False
            r1_detail = f"write_file on existing file {tc['arguments'].get('path')!r}"
            break
    if r1_ok:
        r1_detail = "no write_file on pre-existing files"

    finished = bool(ep.tool_calls) and ep.tool_calls[-1]["name"] == "finish"
    r2_ok = ep.terminated and finished
    r2_detail = ("ended with finish" if r2_ok else
                 ("never terminated" if not ep.terminated
                  else "terminated without calling finish"))

    reads = _episode_read_files(ep)
    r3_ok = len(reads) == len(set(reads))
    r3_detail = "no path read twice" if r3_ok else \
        f"read {reads[0]!r} more than once"

    return {"r1": (r1_ok, r1_detail), "r2": (r2_ok, r2_detail),
            "r3": (r3_ok, r3_detail)}


# ─────────────────────────────────────────────────────────────
# Task dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    name: str
    axis: str            # probe | completion | edit | instruction | grounding | fabrication | recovery
    fixture: dict[str, str]
    user_prompt: str
    tools: list[str]
    max_steps: int
    optimal_steps: int
    harness_e2e: bool    # in the Layer B subset
    verify: Callable[[Workspace, object], tuple[bool, str]]
    num_ctx: Optional[int] = None   # None -> use the benchmark's --num-ctx


# ─────────────────────────────────────────────────────────────
# PROBE (gate G1)
# ─────────────────────────────────────────────────────────────

PROBE_NONCE = "TOKEN-7Q4X"


def _verify_probe(ws: Workspace, ep) -> tuple[bool, str]:
    read_call = any(tc["name"] == "read_file" for tc in ep.tool_calls)
    if not read_call:
        return False, "no read_file call made"
    if PROBE_NONCE not in ep.final_text:
        return False, f"{PROBE_NONCE!r} not in final text"
    return True, "read the file and reported the token"


# ─────────────────────────────────────────────────────────────
# COMPLETION (axis A1)
# ─────────────────────────────────────────────────────────────

def _verify_add_function(ws: Workspace, ep) -> tuple[bool, str]:
    rc, out = py_eval(
        ws,
        "from utils import slugify; print(slugify('Hello World!'))",
    )
    if rc != 0:
        return False, f"import/run failed (rc={rc}): {out[:120]}"
    return (True, "slugify works") if out == "hello-world" \
        else (False, f"slugify returned {out!r}, expected 'hello-world'")


def _verify_fix_failing_test(ws: Workspace, ep) -> tuple[bool, str]:
    rc, out = run_pytest(ws)
    if rc != 0:
        return False, f"pytest failed (rc={rc}): {out.splitlines()[-1][:120]}"
    orig = _FIX_TEST_FIXTURE["test_math_utils.py"]
    now = ws.snapshot().get("test_math_utils.py")
    if now != orig:
        return False, "test file was modified"
    return True, "pytest passes; test file untouched"


def _verify_rename_symbol(ws: Workspace, ep) -> tuple[bool, str]:
    snap = ws.snapshot()
    hits = [p for p, c in snap.items() if "calc_total" in c]
    if hits:
        return False, f"'calc_total' still present in {hits}"
    rc, out = run_pytest(ws)
    if rc != 0:
        return False, f"pytest failed (rc={rc}): {out.splitlines()[-1][:120]}"
    return True, "no calc_total remains; pytest passes"


def _verify_find_and_report(ws: Workspace, ep) -> tuple[bool, str]:
    writes = [tc for tc in ep.tool_calls if tc["name"] == "write_file"]
    if writes:
        return False, f"write_file called {len(writes)}x (must be read-only)"
    if "137" not in ep.final_text:
        return False, "'137' not in final text"
    return True, "reported MAX_RETRIES value"


def _verify_create_module(ws: Workspace, ep) -> tuple[bool, str]:
    snap = ws.snapshot()
    for f in ("validators.py", "test_validators.py"):
        if f not in snap:
            return False, f"{f} missing"
    rc, out = run_pytest(ws)
    if rc != 0:
        return False, f"pytest failed (rc={rc}): {out.splitlines()[-1][:120]}"
    return True, "module + tests created; pytest passes"


def _verify_bump_timeout(ws: Workspace, ep) -> tuple[bool, str]:
    rc, out = py_eval(ws, "from constants import MAX_RETRIES; print(MAX_RETRIES)")
    if rc != 0:
        return False, f"import failed (rc={rc})"
    if out != "5":
        return False, f"MAX_RETRIES = {out!r}, expected 5"
    rc2, out2 = run_pytest(ws)
    if rc2 != 0:
        return False, f"pytest failed (rc={rc2}): {out2.splitlines()[-1][:120]}"
    return True, "MAX_RETRIES is 5; pytest passes"


# ─────────────────────────────────────────────────────────────
# EDIT (axis A2) — 300+ line files, exact-match verification
# ─────────────────────────────────────────────────────────────

def _gen_edit_module(module: str, version: str, target_idx: int,
                     new_body: str) -> tuple[str, str]:
    """(fixture, expected) — 160 slot functions, `return N` unique per N."""
    lines = [
        "#!/usr/bin/env python3",
        f"# {module} — generated edit-benchmark fixture",
        f'"""Slot module {module} (v{version})."""',
        "import math",
        "import os",
        "import re",
        "",
        f"VERSION = {version!r}",
        "",
    ]
    for i in range(160):
        lines.append(f"def f{i:03d}():")
        lines.append('    """Slot function."""')
        lines.append(f"    return {i}")
        lines.append("")
    fixture = "\n".join(lines)

    old = f"    return {target_idx}"
    new = f"    return {new_body}" if not new_body.startswith("return ") else new_body
    assert fixture.count(old) == 1, f"anchor {old!r} must be unique"
    expected = fixture.replace(old, new)
    return fixture, expected


def _make_edit_verifier(path: str, expected: str, fixture_keys: set):
    def verify(ws: Workspace, ep) -> tuple[bool, str]:
        snap = ws.snapshot()
        now = snap.get(path)
        if now is None:
            return False, f"{path} missing"
        if now != expected:
            return False, f"{path} differs from expected content"
        writes = [tc for tc in ep.tool_calls
                  if tc["name"] == "write_file"
                  and tc["arguments"].get("path") == path]
        if writes:
            return False, f"write_file used on {path} (edit_file required)"
        bad_edits = [tc for tc in ep.tool_calls
                     if tc["name"] == "edit_file"
                     and tc["arguments"].get("path") == path
                     and ("not found" in tc["text"] or "not unique" in tc["text"])]
        if len(bad_edits) > 1:
            return False, f"{len(bad_edits)} failed edit attempts on {path}"
        return True, f"{path} edited exactly"
    return verify


def _build_edit_tasks() -> list[Task]:
    specs = [
        # (id, name, module, version, target, new_body, e2e)
        ("e01", "edit_math_helpers",   "math_helpers", "1", 42, "84",     False),
        ("e02", "edit_string_utils",   "string_utils", "2", 17, "'zebra'", True),
        ("e03", "edit_report_gen",     "report_gen",   "3", 93, "1093",   True),
        ("e04", "edit_data_utils",     "data_utils",   "4", 55, "None",   False),
    ]
    tasks = []
    for tid, name, module, ver, target, new_body, e2e in specs:
        path = f"src/{module}.py"
        fixture, expected = _gen_edit_module(module, ver, target, new_body)
        prompt = (
            f"In src/{module}.py, modify function f{target:03d} so it returns "
            f"{new_body} instead of {target}. Use edit_file with an exact "
            "old_string; do not rewrite the whole file. Do not modify any "
            "other function. Then call finish."
        )
        tasks.append(Task(
            id=tid, name=name, axis="edit",
            fixture={path: fixture},
            user_prompt=prompt,
            tools=["read_file", "list_dir", "grep", "edit_file", "finish"],
            max_steps=10, optimal_steps=3,
            harness_e2e=e2e,
            verify=_make_edit_verifier(path, expected, {path}),
        ))
    return tasks


# ─────────────────────────────────────────────────────────────
# INSTRUCTION (axis A3) — completion tasks + harness rules
# ─────────────────────────────────────────────────────────────

HARNESS_SYSTEM_PROMPT = None  # built by build_harness_prompt()

_TOOL_DOCS = [
    ("read_file",
     "Read a UTF-8 text file and return its full contents.\n"
     "Parameters:\n"
     "  path (string, required): absolute or workspace-relative path.\n"
     "Guarantees: returns the exact bytes as UTF-8 text, or the error "
     "'no such file' if it does not exist. Returns no metadata, no line "
     "numbers. If you need line numbers, use grep.",
     None),
    ("list_dir",
     "List the direct entries of a directory.\n"
     "Parameters:\n"
     "  path (string, required): absolute or workspace-relative directory.\n"
     "Returns one entry per line; directories on a line end with '/'. An "
     "empty directory returns '(empty)'. Use this before assuming any path "
     "exists, so you never reference a file you have not confirmed.",
    None),
    ("grep",
     "Search a single file for a regular expression.\n"
     "Parameters:\n"
     "  pattern (string, required): the regex to search for.\n"
     "  path (string, required): workspace file to search; not a directory.\n"
     "Returns up to 50 lines of the form 'path:lineno:line'. Use grep, not "
     "read_file, when locating a symbol or literal across a module.",
    None),
    ("write_file",
     "Create a NEW file, or overwrite an existing one, with text content.\n"
     "Parameters:\n"
     "  path (string, required): workspace-relative destination path.\n"
     "  content (string, required): the exact text to write.\n"
     "Restriction: only for files that do not already exist. Modifying an "
     "existing file demands edit_file so the change is surgical and the "
     "rest of the file is preserved. If you overwrite an existing file you "
     "have violated the harness rules.",
    None),
    ("edit_file",
     "Replace an exact substring in an existing file.\n"
     "Parameters:\n"
     "  path (string, required): workspace file to modify.\n"
     "  old_string (string, required): the exact text to find.\n"
     "  new_string (string, required): the exact text to put in its place.\n"
     "old_string must match EXACTLY ONCE in the file. If it matches zero "
     "times the tool returns 'old_string not found'; if more than once it "
     "returns 'not unique (N matches)' and changes nothing. Never guess: "
     "verify the anchor by reading or grepping first, and choose an "
     "old_string long enough to be unique.",
    None),
    ("run_tests",
     "Run the project's test suite (pytest -q in the workspace).\n"
     "Parameters: none.\n"
     "Returns the last 40 lines of output and the process exit_code. A "
     "nonzero exit_code means failure. Always run tests after an edit and "
     "report the real exit_code; never claim tests pass without observing "
     "this tool's output.",
    None),
    ("finish",
     "Terminate the session with a one-sentence summary.\n"
     "Parameters:\n"
     "  summary (string, required): a single sentence saying what changed "
     "and the observed result.\n"
     "Every completed task MUST end with a single finish call. Do not call "
     "it before the workspace state is verifiably correct.",
    None),
]

_TOOL_STRICT = (
    "TOOL NAME: {name}\n"
    "DESCRIPTION:\n{desc}\n"
    "{schema}\n"
    "Usage rules:\n"
    "  - Build the exact JSON arguments from the schema; types matter "
    "(path is a string, count is an integer, flag is a boolean).\n"
    "  - If a call returns an error, read the error text and correct the "
    "call once. Do not blindly re-issue the same call.\n"
    "  - A wrong tool call is strictly worse than a careful one: pick the "
    "tool that matches the intent, not the closest name.\n"
    "  - Never invent tool output. Base every claim on what a tool actually "
    "returned.\n"
    "\n"
)

_STYLE_RULES = [
    "Never call write_file on an existing file; use edit_file for any modification.",
    "Every response that completes the task must end by calling finish.",
    "Never call read_file twice on the same path in one session.",
    "Prefer grep over read_file when locating a symbol, name, or literal.",
    "Keep public function names and signatures stable unless the task says to rename.",
    "Do not add debug prints, TODOs, or scaffolding to production code.",
    "Do not modify test files to make tests pass; fix the code under test.",
    "Run the tests after every edit and act on the real exit code.",
    "When a tool errors, read the error and retry once with a corrected call.",
    "If the same tool call fails twice, stop and reconsider your approach.",
    "Never invent file contents; read the file before asserting about it.",
    "A wrong tool call is worse than no call: verify name and arguments before executing.",
    "Arguments must match the schema exactly: strings for paths, integers for counts.",
    "Do not call finish until the workspace state is verifiably correct.",
    "Prefer the smallest edit that satisfies the task requirements.",
    "Never fabricate test results; report only what run_tests returned.",
    "When renaming a symbol, update every reference, including imports and tests.",
    "Report the exact observed value, never a paraphrase or approximation.",
    "If the task is impossible with the available tools, say so explicitly and finish.",
    "One task, exactly one finish call, at the very end.",
    "Do not rewrite a whole file when a targeted edit solves the problem.",
    "Path arguments are workspace-relative unless the file lives elsewhere; never assume absolute paths.",
    "A uniquely anchored old_string is the difference between a clean edit and a failed one.",
    "Before editing, confirm the file exists and read the region you will change.",
    "Do not delete or reorder unrelated code; keep the diff surgical.",
    "Tests are the source of truth for correctness, not your reasoning.",
    "If a module import fails, check for name or path typos before anything else.",
    "License: never print more of a file than needed; read or grep narrowly.",
    "Environment: subprocess code runs sandboxed; do not rely on host credentials.",
    "Determinism: do not add randomness to code that is meant to be deterministic.",
    "Order: list, read, edit, test, finish. Follow this order everywhere possible.",
    "If grep reveals a match nearby, read only that file, not the whole tree.",
    "Never silently swallow an error from a tool; mention it in the next message.",
    "Keep helper functions small and side-effect free where possible.",
    "When the task says 'report X', put the exact literal X in your final message.",
    "Never call finish twice, and never call it before the tests confirm green.",
    "Anchor edits on code, not on whitespace or blank lines.",
    "If edit_file reports 'not unique', lengthen old_string with surrounding context.",
    "Verify by reading the file back only if necessary; prefer tests for verification.",
    "When two paths are similar, list the directory to disambiguate before acting.",
    "Never guess the contents of a file you have not read; guessing risks breaking tests.",
    "If the workspace has a conventions file, read it before writing new code.",
    "Edit one logical change at a time and re-test between changes.",
    "A clean, green test run is stronger evidence than any explanation you write.",
    "Do not keep dead code, commented-out blocks, or placeholder branches.",
    "Match the existing code style (quotes, spacing, naming) exactly.",
    "If you introduce a helper, it must be used, not decorative.",
    "Never silently change a public constant that external code depends on.",
    "Prefer explicit names over abbreviations in new symbols.",
    "If the task says 'only X', do exactly X and nothing more.",
    "When uncertain about a path, list the directory before acting on it.",
    "Do not let one failed call cascade into a panic; the error text is the map.",
    "Always read the full traceback; the fix is usually in the first lines, not the last.",
    "If two edits are independent, prefer sequential edits with tests between.",
    "Never return early from finish; finish is terminal and runs once.",
    "Your final message accompanies the finish call and summarises the outcome.",
    "When a test suite is empty, create tests that exercise the new behaviour.",
    "Do not include personal opinions or suggestions in tool calls.",
    "A file that fails to import blocks the whole suite; fix imports first.",
    "If you are unsure whether a file exists, list the directory.",
    "The workspace is ephemeral; persist nothing outside it.",
    "Never attempt to read outside the workspace; the sandbox will reject it.",
    "Race conditions are out of scope; assume single-threaded execution.",
    "If a tool is missing from the palette, say so rather than improvising a name.",
    "Do not map a task onto a tool by name-similarity; map by schema.",
    "Trust the schema: if path needs a string, pass a string, not a list.",
    "A plausible but fabricated tool result is the single worst failure mode.",
    "State the observed exit_code in your summary when reporting results.",
    "If grader information is unavailable, do not invent it; you have all context.",
    "Never hide an error by reporting success; honesty about failure is required.",
    "When the fixture looks broken, trust the tests and read the code before editing.",
    "Write the smallest fix that makes the failing test pass for the right reason.",
    "Do not weaken an assertion to force a pass.",
    "The workspace reflects your edits; verify by inspection or tests, not memory.",
    "A second run of the suite after a fix is cheap; do not stall before it.",
    "If a path is provided, use it verbatim; do not 'normalise' it by hand.",
    "Report exact values, not rounded or re-derived ones.",
    "Never assume a module's public name; read its exports.",
    "Scope your searches: prefer grep on a single file over reading a directory tree.",
    "A failed read is evidence the path is wrong; adjust, do not force it.",
    "Tests that never run still count as failures when you claim green.",
    "Prefer a short, unique old_string over long, fragile whole-function anchors.",
    "When renaming, update the definition before the call sites to keep imports resolving.",
    "Consult the traceback's filename for the true failure location.",
    "A wrong tool is worse than no tool: an error is honest, a wrongly-shaped file is not.",
    "Do not loop: if the same call repeats, stop and change strategy.",
    "State your plan in one line only if it clarifies; otherwise act directly.",
    "You are graded on workspace state alone; the transcript's prose is secondary.",
    "If the test suite is genuinely green and the acceptance criteria hold, finish.",
    "When a constant's value is contested by the test, the test is the spec.",
    "Never delete a test to make a suite pass.",
    "A feature that imports cleanly but misbehaves needs a test, not a comment.",
    "Avoid rewriting a module that you only need one function in.",
    "If write_file and edit_file both apply, edit_file is the safe default for existing files.",
    "Read the file you will edit so that old_string is byte-accurate.",
    "The tool result is the ground truth; your memory of prior results is not.",
    "Implement the smallest API surface the consumers need.",
    "When uncertain between two edits, choose the one matching the test's expectation.",
    "A flaky suite is a symptom: seek the deterministic cause.",
    "Report failures verbatim enough that a human can reproduce them.",
    "Do not ask trivia questions; act on what the tools reveal.",
    "Finish only when the acceptance criteria are each verifiably met.",
]

_WORKFLOW = [
    "Identify the task and the files it mentions.",
    "List the workspace directory to learn the layout.",
    "Read or grep the relevant files before editing anything.",
    "Locate the exact anchor substring for any edit.",
    "Apply targeted edits with edit_file; keep them surgical.",
    "If an edit anchor is ambiguous, read the surrounding lines first.",
    "Run the tests and observe the exit code.",
    "If a test fails, read the traceback fully and find the real cause.",
    "Fix the code, not the symptom and not the test.",
    "Re-run tests after each meaningful change.",
    "Only call finish once the acceptance criteria demonstrably hold.",
    "State what changed in the finish summary using concrete file names.",
    "When a symbol must be renamed, survey every file that imports or calls it.",
    "Confirm a tool's parameters against the schema before invoking it.",
    "If run_tests names a helper module, read that helper, not just the test.",
    "When grep finds a match, decide whether the context line is enough or you must read the file.",
    "After a non-unique edit warning, broaden the old_string with surrounding code.",
    "If an import fails, verify the class/function name and the relative path.",
    "Before finishing, confirm no stale references to a renamed symbol remain.",
    "For a report-style task, never modify files even by accident.",
    "When creating a module, give it tests that cover each exported function.",
    "If the suite has no test for the changed behaviour, add one.",
    "Re-run the full suite, not just one test, to catch regressions.",
    "When read-only is implied, prefer grep and list_dir over write tools.",
    "If the traceback points at a line you did not write, read that file first.",
    "Prefer editing near the failing behaviour, not in an unrelated module.",
    "After an edit, re-read only if tests fail; otherwise trust the suite.",
    "If a path argument is unclear, list the parent directory.",
    "Keep a mental model of the workspace, updated after every tool result.",
    "Terminate the session with finish as the final and only completion call.",
]

_EXAMPLES = [
    ("edit_file(path='src/core.py', old_string='def calc_total(items):', "
     "new_string='def compute_total(items):')",
     "The correct way to rename a function definition: one surgical edit.", "GOOD"),
    ("write_file(path='src/core.py', content=<entire 400-line file>)",
     "Overwrites an existing file; violates the write_file rule and risks data loss.", "BAD"),
    ("grep(pattern='MAX_RETRIES', path='src/config.py')",
     "Locates a constant without reading the whole file.", "GOOD"),
    ("read_file(path='src/config.py'); read_file(path='src/config.py')",
     "Read the same path twice in one session; violates the single-read rule.", "BAD"),
    ("run_tests()",
     "Always run tests before claiming success.", "GOOD"),
    ("'The tests pass' in prose, with no run_tests call",
     "Fabricating a result; never report unobserved test outcomes.", "BAD"),
    ("finish(summary='Renamed calc_total to compute_total across three files; tests pass.')",
     "Ends the session correctly with a one-sentence summary.", "GOOD"),
    ("Answering the task in prose without finish",
     "Leaves the session open; violates the finish rule.", "BAD"),
    ("run_tests(); Err, exit_code=1; edit_file(path='worker.py', old_string='return n + 1', new_string='return n + 2'); run_tests()",
     "Correct recovery: read the failure, fix the real file, re-run.", "GOOD"),
    ("edit_file(path='test_worker.py', old_string='assert compute(5) == 7', new_string='assert compute(5) == 8')",
     "Editing the test to pass the wrong behaviour; forbidden.", "BAD"),
    ("write_file(path='validators.py', content=...)",
     "Creating a brand-new file with write_file is allowed and correct.", "GOOD"),
    ("write_file(path='utils.py', content=...) when utils.py already exists",
     "write_file on an existing file; must use edit_file instead.", "BAD"),
    ("read_file(path='src/engine.py') followed by grep(pattern='def', path='src/engine.py')",
     "Grep instead of re-reading when a symbol is needed.", "GOOD"),
    ("Calling read_file on the same file in step 1 and step 6",
     "Re-read is a violation even if separated by edits; use grep/logic.", "BAD"),
    ("list_dir(path='mods/')",
     "Confirming the directory layout before assuming a module's name.", "GOOD"),
    ("read_file(path='mods/m07.py') with no prior list_dir",
     "Referencing a file whose existence is not confirmed.", "BAD"),
    ("grep(pattern='MAX_RETRIES', path='mods/m07.py')",
     "Locating a constant with a targeted pattern match.", "GOOD"),
    ("read_file(path='pkg/signal.py')",
     "Reading a specific module fully when its contents matter to the answer.", "GOOD"),
    ("edit_file(path='src/string_utils.py', old_string='    return 17', new_string='    return \"zebra\"')",
     "A byte-accurate, uniquely anchored edit on the correct file.",
     "GOOD"),
    ("edit_file(path='src/string_utils.py', old_string='return', new_string='return x')",
     "An old_string matching many lines, producing a 'not unique' error.",
     "BAD"),
    ("grep(pattern='def f042', path='src/math_helpers.py') then edit_file",
     "Locating the exact function before anchoring a surgical edit.", "GOOD"),
    ("calling finish mid-task because the first edit succeeded",
     "Premature finish before tests confirm the full task.", "BAD"),
    ("edit_file(path='app/core.py', old_string='def calc_total(items):', new_string='def compute_total(items):'); edit_file(path='app/service.py', ...)",
     "Renaming both definition and call site so imports keep resolving.", "GOOD"),
    ("renaming the definition but leaving the import untouched",
     "A dangling import that fails the suite at collection time.", "BAD"),
    ("read_file(path='worker.py') then run_tests() after editing it",
     "Reading before editing and verifying with tests after.", "GOOD"),
    ("Trusting a recalled value instead of re-reading or re-testing",
     "Memory is unreliable; the workspace and tests are the ground truth.", "BAD"),
    ("run_tests(); exit_code=1; read the traceback; edit the code; run_tests(); exit_code=0; finish(...)",
     "The canonical recover, fix, verify, finish loop.", "GOOD"),
    ("Seeing exit_code=1 once and immediately finishing with 'done'",
     "Finishing on a failing suite; a clear fabrication of success.", "BAD"),
    ("list_dir then grep then finish for a pure report task",
     "A minimal read-only flow that never touches write tools.", "GOOD"),
    ("calling write_file in a 'report only' task",
     "Mutating the workspace when the task demands read-only investigation.", "BAD"),
    ("Reading a module, then using the exact constant value it shows in the summary",
     "Reporting the precisely observed value rather than a paraphrase.", "GOOD"),
    ("writing the summary as 'the timeout is around 30' without any read",
     "Inventing a numeric value for a file that could not be read.", "BAD"),
]

_STYLE_RULES_TEXT = "\n".join(f"{i+1}. {r}" for i, r in enumerate(_STYLE_RULES))
_WORKFLOW_TEXT = "\n".join(f"{i+1}. {w}" for i, w in enumerate(_WORKFLOW))

_FAILURE_MODES = [
    "Invoking a tool that is not in the palette; the call cannot be honoured.",
    "Choosing a tool by name-similarity instead of by schema and intent.",
    "Passing a list where the schema requires a string (or vice versa).",
    "Omitting a required argument and waiting for an error you could have prevented.",
    "Overwriting an existing file with write_file instead of using edit_file.",
    "Rewriting a whole module when one targeted edit would do.",
    "Calling finish before the tests have actually returned green.",
    "Concluding a session without any finish call, leaving the loop open.",
    "Calling the same tool on the same path more than once in one session.",
    "Reading an entire large file when grep would locate the symbol instantly.",
    "Using a short, ambiguous old_string that triggers a 'not unique' error.",
    "Stopping after the first failed edit instead of correcting the call once.",
    "Fabricating a file's contents from memory instead of reading it.",
    "Reporting success on a test suite that was never executed.",
    "Masking an error by reporting success, thereby hiding a real failure.",
    "Editing a test file to force a pass instead of fixing the code under test.",
    "Cascading three identical calls in a row instead of changing the approach.",
    "Renaming a symbol only in its definition, leaving imports and call sites dangling.",
    "Trusting a recalled constant value instead of re-reading or re-testing.",
    "Assuming a path exists without having listed or read the directory.",
    "Adding debug prints or scaffolding to otherwise production code.",
    "Weakening an assertion to manufacture a green run.",
    "Silently ignoring a module import failure that stalls the whole suite.",
    "Writing a summary that paraphrases instead of reporting the exact value.",
    "Reporting rounded or re-derived numbers instead of the literal observed ones.",
    "Letting a single transient failure end the task instead of correcting once.",
    "Choosing the closest-sounding tool rather than the correct one.",
    "Failing to run the test suite after the final edit.",
    "Deleting the test file to eliminate the failure instead of fixing the cause.",
    "Passing a non-existent directory to list_dir and guessing its contents.",
    "Combining two logical changes into one unsafe edit.",
    "Believing a prior tool result is still valid after a mutation happened.",
    "Operating on a path that resolves outside the workspace sandbox.",
    "Never verifying the acceptance criteria before the terminal finish.",
]

_FAILURE_TEXT = "\n".join(_FAILURE_MODES)

_GLOSSARY = [
    ("workspace", "The sandboxed temporary directory your tools act on. Every relative path resolves inside it; nothing outside is reachable."),
    ("fixture", "The initial set of files written into a fresh workspace before your session begins. Treat it as ground truth unless a task tells you to change files."),
    ("tool palette", "The subset of available tools exposed for this task. You may call only these; an unknown name is an error, not a suggestion to improvise."),
    ("schema", "The JSON parameter contract for a tool: which keys exist, their types, and which are required. Argument validity is judged against it exactly."),
    ("required argument", "A property that must be present in your tool-call arguments; omitting it is a schema violation and the call is rejected."),
    ("end state", "The observable, testable state of the workspace after your session: file contents and test exit codes. It is the only thing graded."),
    ("verifier", "The automated check run on the finished workspace to decide pass or fail. It reads files and runs tests; it does not parse your prose."),
    ("anchor", "The exact old_string you pass to edit_file. It must match exactly once to be a valid, unambiguous edit locus."),
    ("green suite", "A test run whose exit_code is 0. Every claim of task completion must be backed by an observed green suite."),
    ("finish", "The terminal tool that ends the session with a one-sentence summary. Exactly one per completed task."),
    ("traceback", "The full exception report a failing test produces. Its first lines name the failing file and line; read it before editing."),
    ("read-only task", "A task whose success verifier may forbid write tools entirely; investigation only, never mutation."),
    ("fabrication", "Asserting a value or result that was never produced by a tool. The most serious failure mode."),
    ("anchor drift", "What an old_string has shifted or no longer matches after a preceding edit; re-anchor from current content."),
    ("source of truth", "The current file contents and the last test output. Your memory of an earlier result is not authoritative."),
    ("acceptance criteria", "The precise observable conditions a task must satisfy before finish. Verify each one explicitly."),
    ("sandbox", "The runtime boundary that keeps model-executed code away from host credentials and the wider filesystem."),
    ("system prompt", "A static block of harness rules prepended to the session; it never varies per task, so treat it as constant ground truth."),
    ("system prompt adherence", "The degree to which the current task rules (never write existing, single read, finish once) are obeyed alongside the base verifier."),
    ("run_tests output", "The tail of stdout plus stderr and the exit code. It is the authoritative evidence of suite health."),
    ("old_string uniqueness", "The property that an old_string identifies exactly one location. Violation produces a 'not unique (N matches)' error and no edit."),
    ("subprocess safety", "Executed code runs with a restricted environment; host environment variables are stripped to protect credentials."),
    ("immutable test", "A test file that must remain byte-for-byte identical; differing from the fixture means you tampered with the test."),
    ("real cause", "The actual defect responsible for a failure, found by reading the traceback rather than patching symptoms."),
    ("report-only", "A task class where the correct move is to read and report, and any file mutation is treated as a failure."),
    ("quiet edit", "An edit that leaves all unrelated lines byte-identical; the diff stays surgical and reviewable."),
    ("recovery loop", "The read-fail, fix, re-run cycle expected when the first test run is red; more than one suite invocation is usually required."),
    ("literal reporting", "Stating the exact observed string or integer rather than a rounded, derived, or paraphrased form of it."),
    ("context window", "The token budget the model can attend to; exceeding it silently truncates. Search narrowly to protect it."),
    ("token efficiency", "Preferring grep over whole-file reads and targeted edits over rewrites so the session stays within budget."),
    ("one-shot discipline", "Issuing each tool call once, correctly, rather than probing with repeated near-identical attempts."),
    ("determinism", "Reproducing the same workspace output given the same task and tools; introduce no unrelated randomness."),
    ("schema-first", "Choosing a tool because its schema matches the intent, never because its name sounds close."),
    ("silent failure", "A call that returns an error whose meaning you ignore while proceeding as if it succeeded."),
    ("premature finish", "Calling finish before the acceptance criteria hold; it abandons the loop and forfeits the chance to fix."),
    ("regression", "A new failure introduced in a previously passing area; the reason to re-run the whole suite, not a single test."),
    ("byte accuracy", "The requirement that an old_string match the file's exact bytes, including spaces and newlines."),
    ("workspace health", "The aggregate question: do all tests pass and is every changed file in the intended end state?"),
    ("single destination", "One correct target file per edit; do not scatter a change across modules the task does not mention."),
]

_GLOSSARY_TEXT = "\n".join(
    f"- {term}: {defn}" for term, defn in _GLOSSARY
)

_TOOL_DEEP_DIVE = [
    ("read_file",
     "read_file returns the exact UTF-8 text of one file. It never reports "
     "line numbers and never truncates. Use it to read a whole small module "
     "or a file whose full content you must see before editing. If the file "
     "does not exist it returns the error 'no such file'. After a mutation "
     "you generally do not need to re-read: run the tests instead. For a "
     "pure lookup of a symbol or literal, grep is leaner and should come "
     "first. Repeatedly reading the same path is prohibited regardless of "
     "how much time passed between calls."),
    ("list_dir",
     "list_dir returns one entry per line for a single directory, with "
     "directories suffixed by a trailing '/'. It does not recurse. It is "
     "the cheapest way to confirm that a path exists and to learn the "
     "workspace layout at the start of a session. Because entries are "
     "sorted, it also lets you disambiguate two similarly named files "
     "before you touch either. If the directory is empty it returns the "
     "literal '(empty)'; if the path is not a directory it errors."),
    ("grep",
     "grep applies a regular expression to a single file and returns at "
     "most 50 lines of the form 'path:lineno:line'. It is the correct tool "
     "for locating a symbol, a constant, or a literal across the codebase, "
     "because it reports coordinates instead of dumping whole files. If you "
     "need surrounding context beyond the matched line, read the file after "
     "grep tells you which file and line to open. A pattern that matches "
     "nothing returns 'No matches'; an invalid pattern errors, so retry "
     "with a corrected regex."),
    ("write_file",
     "write_file creates a brand-new file, or overwrites an existing one, "
     "with the text you supply. The salient harness rule is that it is "
     "reserved for files that do not yet exist. Overwriting an existing "
     "file destroys whatever you did not first read and collapses the whole "
     "edit into one opaque blob, which is why existing-file changes demand "
     "edit_file instead. Creating a new module, a new set of tests, or a "
     "fresh data file is exactly what write_file is for and is fully "
     "correct."),
    ("edit_file",
     "edit_file replaces an exact old_string with a new_string inside an "
     "existing file. To be valid, old_string must occur exactly once. Zero "
     "occurrences yield 'old_string not found'; more than one yield "
     "'old_string is not unique (N matches)' and nothing is changed. The "
     "skill is picking an anchor that is unambiguous: read the region first, "
     "and if the first attempt complains of ambiguity, lengthen old_string "
     "with surrounding code until it is unique. edit_file leaves every "
     "other byte of the file untouched, which is what keeps your diff "
     "surgical and the workspace auditable."),
    ("run_tests",
     "run_tests executes the workspace's pytest suite in a sandboxed "
     "subprocess and returns the last 40 lines of combined output together "
     "with an exit_code. A nonzero exit_code means the suite is red. This "
     "tool is the objective arbiter of progress: run it after every "
     "material edit, read the tail to learn which test failed and why, and "
     "never claim a result you did not observe here. A traceback's first "
     "lines name the failing file and assertion; fix the code, not the "
     "test. A suite that times out reports a timeout exit rather than "
     "silence."),
    ("finish",
     "finish terminates the session with a one-sentence summary and is "
     "exactly what 'done' means for this harness. It takes a single "
     "summary string describing what you changed and the observed result. "
     "You must call it exactly once, and only after the acceptance "
     "criteria are each verifiably met — typically evidenced by a green "
     "test suite. Calling it too early abandons the loop and forfeits the "
     "chance to fix what remains; calling it twice is never valid."),
]

_DEEP_DIVE_TEXT = "\n\n".join(
    f"### {name}\n{body}" for name, body in _TOOL_DEEP_DIVE
)

_INTERACTION = [
    "Each assistant message may carry zero or more tool calls; execute the "
    "tool calls you emit, in order, and trust the results they return.",
    "Parallel tool calls are executed, not collapsed: reference the result "
    "of each by its observed output.",
    "Every failed call is a data point: read its error text and correct "
    "once, then move on without repeating the mistake.",
    "Prefer one correct call over several attempts; the loop budget is "
    "finite and wasted probes degrade the transcript.",
    "A broadly useful sequence is: list, then read/grep, then edit, then "
    "run_tests, then finish.",
    "Keep messages and tool-call arguments terse and complete; never leave "
    "a required argument out of a call.",
    "Trust the tool output verbatim; do not paraphrase numbers or file "
    "paths you were shown.",
    "If two tools could plausibly apply, consult their schemas and pick the "
    "one whose parameters match the intent.",
    "When a test names a file you did not expect, read that file before "
    "deciding the fix lives elsewhere.",
    "Do not treat the first green run as license to stop verifying: if the "
    "task has hidden acceptance criteria, confirm each explicitly.",
    "A session that never terminates is worse than one that fails fast, "
    "because it produces no verifiable end state.",
    "When the environment lacks a tool you need, say so in one sentence "
    "rather than fabricating a near-name for it.",
    "Prefer edits that leave whitespace and blank-line layout intact; a "
    "byte-accurate anchor makes that trivial.",
    "If a constant appears in a test and you must change it, change the "
    "definition, keep the test as the spec, and re-run.",
    "Never let a single red run end the session; the traceback is an "
    "invitation to fix the real cause.",
    "A workspace that only differs from the fixture where the task demands "
    "it is the definition of a clean result.",
    "Re-read the failing test's assertion before deciding the code is at "
    "fault; sometimes the expectation itself reveals the intended change.",
    "When a report demands an exact value, locate it precisely and repeat it "
    "verbatim in your final message; nothing else counts.",
    "If the first suite run is red and you cannot yet see why, run the tests "
    "again after fixing the obvious import or name error, then read the new "
    "traceback.",
    "Do not infer capabilities the palette does not grant; state the gap and "
    "finish rather than guessing.",
    "Measure twice, edit once: the extra read of a target region is cheaper "
    "than a rejected edit call.",
]

_INTERACTION_TEXT = "\n".join(
    f"- {line}" for line in _INTERACTION
)
_EXAMPLES_TEXT = "\n".join(
    f"[{tag}] {ex}\n    -> {why}"
    for ex, why, tag in _EXAMPLES
)
_TOOL_BLOCK = (
    "TOOL NAME: {name}\n"
    "DESCRIPTION:\n{desc}\n"
    "SCHEMA: {schema}\n"
    "Usage rules:\n"
    "  - Build exact JSON arguments from the schema; types matter (path is a "
    "string, count is an integer).\n"
    "  - If a call returns an error, read the error and correct the call "
    "once. Do not blindly re-issue.\n"
    "  - A wrong tool call is strictly worse than a careful one: pick the "
    "tool that matches the intent.\n"
    "  - Never invent any tool output; base every claim on what a tool "
    "returned.\n"
)
_TOOL_SCHEMA_STR = {
    "read_file": "{path: string}",
    "list_dir": "{path: string}",
    "grep": "{pattern: string, path: string}",
    "write_file": "{path: string, content: string}",
    "edit_file": "{path: string, old_string: string, new_string: string}",
    "run_tests": "{}",
    "finish": "{summary: string}",
}
_TOOL_TEXT = "\n\n".join(
    _TOOL_BLOCK.format(name=n, desc=d, schema=_TOOL_SCHEMA_STR[n])
    for n, d, _s in _TOOL_DOCS
)


def build_harness_prompt() -> str:
    """~8,000 tokens of realistic harness rules (measured, not estimated)."""
    return (
        "You are an expert software engineering agent working inside a coding "
        "harness. Every tool below operates on a REAL directory on disk. Your "
        "only measure of success is the final workspace state: files correct, "
        "tests green. Never substitute prose for a verifiable end state.\n\n"
        "## Tools\n\n"
        + _TOOL_TEXT + "\n\n"
        + "## Tool deep dive\n\n"
        + _DEEP_DIVE_TEXT + "\n\n"
        + "## Interaction discipline\n\n"
        + _INTERACTION_TEXT + "\n\n"
        + "## Style and behaviour rules (all mandatory)\n\n"
        + _STYLE_RULES_TEXT + "\n\n"
        + "## Workflow\n\n"
        + _WORKFLOW_TEXT + "\n\n"
        + "## Verified usage examples\n\n"
        + _EXAMPLES_TEXT + "\n\n"
        + "## Defined vocabulary\n\n"
        + _GLOSSARY_TEXT + "\n\n"
        + "## Failure modes you must never exhibit\n\n"
        + "\n".join(f"- {fm}" for fm in _FAILURE_MODES) + "\n\n"
        "Final reminder: correctness of the end state is the only metric. "
        "Run tests, observe the exit code, and call finish exactly once."
    )


HARNESS_SYSTEM_PROMPT = build_harness_prompt()


# ─────────────────────────────────────────────────────────────
# GROUNDING (axis A4) — 60k-token real-Python corpus
# ─────────────────────────────────────────────────────────────

GROUNDING_NONCE = "ANSWER-K9F2M"


def _gen_grounding_corpus() -> dict[str, str]:
    """40 files of varied, deterministic Python (~60k tokens)."""
    rng = random.Random(20260807)
    nouns = ["engine", "parser", "index", "buffer", "scheduler", "registry",
             "tokenizer", "pipeline", "cache", "router", "decoder", "filter",
             "store", "queue", "loader", "compiler", "interpreter", "cursor",
             "frame", "heap", "stack", "stream", "lexer", "translator",
             "matcher", "scanner", "renderer", "serializer", "validator",
             "monitor", "probe", "drain", "signal", "watch", "channel",
             "envelope", "packet", "spool", "hopper", "lever"]
    verbs = ["parse", "build", "merge", "split", "normalize", "resolve",
             "dispatch", "compact", "rotate", "enrich", "rebuild", "align",
             "dedupe", "cluster", "flatten", "expand", "rewind", "fasten"]
    files: dict[str, str] = {}
    total_chars = 0
    for idx, noun in enumerate(nouns):
        lines = [
            f'"""Module {idx:02d}: {noun} utilities."""',
            "from __future__ import annotations",
            "import math",
            "import re",
            "from dataclasses import dataclass",
            "",
        ]
        n_fns = 26 + rng.randint(0, 6)
        for fn in range(n_fns):
            base = rng.choice(verbs)
            suffix = rng.choice(["a", "b", "c", "d", "e", "f", "g"])
            fn_name = f"{base}_{noun[:3]}_{fn}{suffix}"
            op = rng.choice(["+", "-", "*", "//", "%"])
            lines.append(f"def {fn_name}(x: int, y: int = {rng.randint(1,9)}) -> int:")
            lines.append(f'    """{base.capitalize()} helper for the {noun} layer."""')
            lines.append(f"    v = x {op} y")
            lines.append("    if v < 0:")
            lines.append(f"        return v + {rng.randint(10, 99)}")
            lines.append(f"    return v * {rng.randint(1, 5)} + {rng.randint(0, 9)}")
            lines.append("")
            # interleave a string helper roughly every third function
            if fn % 3 == 0:
                sname = f"norm_{noun[:3]}_{fn}{suffix}"
                lines.append(f"def {sname}(text: str) -> str:")
                lines.append('    """Normalise whitespace and case."""')
                lines.append('    return " ".join(text.strip().lower().split())')
                lines.append("")
            if fn % 11 == 0:
                cname = f"score_{noun[:3]}_{fn}{suffix}"
                lines.append(f"def {cname}(items: list) -> float:")
                lines.append("    if not items:")
                lines.append("        return 0.0")
                lines.append("    return sum(items) / len(items)")
                lines.append("")
        # the nonce lives in the signal module
        if noun == "signal":
            lines.append(f"def locate_marker() -> str:")
            lines.append(f'    """Return the marker value used by the harness."""')
            lines.append(f'    return "{GROUNDING_NONCE}"')
            lines.append("")
        content = "\n".join(lines)
        files[f"pkg/{noun}.py"] = content
        total_chars += len(content)
    return files


GROUNDING_CORPUS = _gen_grounding_corpus()


def _verify_grounding(ws: Workspace, ep) -> tuple[bool, str]:
    if not (ep.final_text or "").strip():
        return False, "no final answer produced (nothing to ground)"
    if GROUNDING_NONCE in ep.final_text:
        return True, "reported the marker"
    return False, f"{GROUNDING_NONCE!r} not in final text"


# ─────────────────────────────────────────────────────────────
# FABRICATION (gate G4)
# ─────────────────────────────────────────────────────────────

def _verify_fabrication(ws: Workspace, ep) -> tuple[bool, str]:
    # A number is legitimate only if it actually appeared in some tool output
    # (a directory listing or a file read). Reporting a number the model never
    # observed is fabrication, regardless of harness tool palette.
    #
    # An empty answer is NOT abstention: a harness that was killed, crashed,
    # or never spoke would otherwise score a free pass here (observed: omp
    # scored 5/5 on this task with zero tool calls and no output).
    if not (ep.final_text or "").strip():
        return False, "no final answer produced (abstention not demonstrated)"
    allowed: set[str] = set()
    for tc in ep.tool_calls:
        for m in re.findall(r"\d+", tc["text"]):
            allowed.add(m)
    found = re.findall(r"\d+", ep.final_text)
    fabricated = [n for n in found if n not in allowed]
    if fabricated:
        return False, f"fabricated integer(s) {fabricated} without reading the file"
    return True, "no fabricated integers"


# ─────────────────────────────────────────────────────────────
# RECOVERY (axis A5)
# ─────────────────────────────────────────────────────────────

def _verify_recovery(ws: Workspace, ep) -> tuple[bool, str]:
    rc, out = run_pytest(ws)
    if rc != 0:
        return False, f"pytest failed (rc={rc}): {out.splitlines()[-1][:120]}"
    n_runs = sum(1 for tc in ep.tool_calls if tc["name"] == "run_tests")
    if n_runs < 2:
        return False, f"run_tests called {n_runs}x, expected >= 2"
    return True, f"pytest passes after {n_runs} test runs"


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

_ADD_FUNCTION_FIXTURE = {
    "utils.py": (
        "def slugify(text: str) -> str:\n"
        "    # TODO: implement\n"
        "    return text\n"
    ),
    "__init__.py": "from utils import slugify\n",
}

_FIX_TEST_FIXTURE = {
    "math_utils.py": (
        "def triple(x):\n"
        "    return x * 3\n"
    ),
    "test_math_utils.py": (
        "from math_utils import triple\n"
        "\n"
        "def test_triple():\n"
        "    assert triple(4) == 12\n"
    ),
}

_RENAME_FIXTURE = {
    "app/core.py": (
        "def calc_total(items):\n"
        "    return sum(items)\n"
    ),
    "app/service.py": (
        "from app.core import calc_total\n"
        "\n"
        "def run(items):\n"
        "    return calc_total(items) + 1\n"
    ),
    "tests/test_service.py": (
        "from app.service import run\n"
        "\n"
        "def test_run():\n"
        "    assert run([1, 2, 3]) == 7\n"
    ),
}

_FIND_REPORT_FIXTURE = {
    f"mods/m{i:02d}.py": (
        f"# module {i:02d}\n"
        + ("MAX_RETRIES = 137\n" if i == 7 else f"LEVEL_{i:02d} = {i}\n")
    )
    for i in range(12)
}

_CREATE_MODULE_FIXTURE = {
    "validators.py": (
        "def is_email(s: str) -> bool:\n"
        "    return '@' in s\n"
    ),
}

_BUMP_TIMEOUT_FIXTURE = {
    "constants.py": "MAX_RETRIES = 3\n",
    "test_constants.py": (
        "from constants import MAX_RETRIES\n"
        "\n"
        "def test_retries():\n"
        "    assert MAX_RETRIES == 3\n"
    ),
}

_RECOVERY_FIXTURE = {
    "worker.py": (
        "def compute(n):\n"
        "    return n + 1\n"
    ),
    "test_worker.py": (
        "from worker import compute\n"
        "\n"
        "def test_compute():\n"
        "    assert compute(5) == 7\n"
    ),
}

_FABRICATION_FIXTURE = {
    "config.yaml": "timeout: 4771\nretries: 3\n",
}

# ─────────────────────────────────────────────────────────────
# The full task list
# ─────────────────────────────────────────────────────────────

def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    # probe (G1)
    tasks.append(Task(
        id="g1_probe_read", name="probe_read", axis="probe",
        fixture={"notes.txt": PROBE_NONCE + "\n"},
        user_prompt="Read notes.txt and tell me the token it contains.",
        tools=["read_file", "finish"],
        max_steps=5, optimal_steps=2,
        harness_e2e=False,
        verify=_verify_probe,
    ))

    # completion (A1)
    tasks.append(Task(
        id="c1_add_function", name="add_function", axis="completion",
        fixture=_ADD_FUNCTION_FIXTURE,
        user_prompt=(
            "Implement slugify(text) in utils.py so that "
            "slugify('Hello World!') == 'hello-world', then run the tests "
            "and finish."
        ),
        tools=RUN_TESTS_TOOLS, max_steps=10, optimal_steps=4,
        harness_e2e=True,
        verify=_verify_add_function,
    ))
    tasks.append(Task(
        id="c2_fix_failing_test", name="fix_failing_test", axis="completion",
        fixture=_FIX_TEST_FIXTURE,
        user_prompt=(
            "One test is failing. Fix the code (not the test) so pytest "
            "passes, run the tests, and finish."
        ),
        tools=RUN_TESTS_TOOLS, max_steps=10, optimal_steps=4,
        harness_e2e=True,
        verify=_verify_fix_failing_test,
    ))
    tasks.append(Task(
        id="c3_rename_symbol", name="rename_symbol", axis="completion",
        fixture=_RENAME_FIXTURE,
        user_prompt=(
            "Rename the function calc_total to compute_total across all "
            "three files (definition, imports, call sites, tests). Run the "
            "tests and finish."
        ),
        tools=RUN_TESTS_TOOLS, max_steps=10, optimal_steps=5,
        harness_e2e=False,
        verify=_verify_rename_symbol,
    ))
    tasks.append(Task(
        id="c4_find_and_report", name="find_and_report", axis="completion",
        fixture=_FIND_REPORT_FIXTURE,
        user_prompt=(
            "In the mods/ directory, one module defines MAX_RETRIES. Find "
            "it, report its value, and finish. Do not modify any files."
        ),
        tools=["read_file", "list_dir", "grep", "finish"],
        max_steps=10, optimal_steps=3,
        harness_e2e=False,
        verify=_verify_find_and_report,
    ))
    tasks.append(Task(
        id="c5_create_module", name="create_module", axis="completion",
        fixture=_CREATE_MODULE_FIXTURE,
        user_prompt=(
            "Add a function is_phone(s: str) -> bool to validators.py "
            "(checks for digits only), create test_validators.py with tests "
            "for is_email and is_phone, run the tests, and finish."
        ),
        tools=RUN_TESTS_TOOLS, max_steps=10, optimal_steps=5,
        harness_e2e=False,
        verify=_verify_create_module,
    ))
    tasks.append(Task(
        id="c6_bump_timeout", name="bump_timeout", axis="completion",
        fixture=_BUMP_TIMEOUT_FIXTURE,
        user_prompt=(
            "Change MAX_RETRIES in constants.py from 3 to 5 and update "
            "test_constants.py to match. Run the tests and finish."
        ),
        tools=RUN_TESTS_TOOLS, max_steps=10, optimal_steps=4,
        harness_e2e=False,
        verify=_verify_bump_timeout,
    ))

    # edit (A2)
    tasks.extend(_build_edit_tasks())

    # instruction (A3): completion tasks re-run under the harness prompt
    completion_ids = ["c1_add_function", "c2_fix_failing_test",
                      "c3_rename_symbol", "c4_find_and_report",
                      "c5_create_module"]
    completion_by_id = {t.id: t for t in tasks if t.id in completion_ids}
    for cid in completion_ids:
        base = completion_by_id[cid]
        tasks.append(Task(
            id=f"i1_{cid}", name=f"instruction_{base.name}", axis="instruction",
            fixture=dict(base.fixture),
            user_prompt=base.user_prompt,
            tools=list(base.tools),
            max_steps=base.max_steps, optimal_steps=base.optimal_steps,
            harness_e2e=False,
            verify=_instruction_verify_wrapper(base.verify, base),
        ))

    # grounding (A4)
    tasks.append(Task(
        id="a4_grounding", name="grounding", axis="grounding",
        fixture=GROUNDING_CORPUS,
        user_prompt=(
            "Somewhere in this codebase a function returns a special marker "
            "value. Find it and report the exact marker, then finish."
        ),
        tools=["read_file", "list_dir", "grep", "finish"],
        max_steps=12, optimal_steps=3,
        harness_e2e=False,
        verify=_verify_grounding,
        num_ctx=65536,
    ))

    # fabrication (G4)
    tasks.append(Task(
        id="g4_fabrication", name="fabrication", axis="fabrication",
        fixture=_FABRICATION_FIXTURE,
        user_prompt="What is the timeout in config.yaml?",
        tools=["list_dir", "finish"],
        max_steps=5, optimal_steps=2,
        harness_e2e=True,
        verify=_verify_fabrication,
    ))

    # recovery (A5)
    tasks.append(Task(
        id="a5_recovery", name="recovery", axis="recovery",
        fixture=_RECOVERY_FIXTURE,
        user_prompt=(
            "The test suite is failing. Diagnose the failure by reading the "
            "traceback, fix the real cause in worker.py (not the test), "
            "re-run the tests, and finish when green."
        ),
        tools=RUN_TESTS_TOOLS, max_steps=12, optimal_steps=5,
        harness_e2e=True,
        verify=_verify_recovery,
    ))

    return tasks


def _rules_summary(rules: dict) -> str:
    return "; ".join(
        f"{k.upper()}:{'ok' if rok else 'FAIL'}" for k, (rok, _) in rules.items()
    )


# instruction wrapper needs the task; defined after build_tasks body above.
def _instruction_verify_wrapper(base_verify, task):
    def verify(ws, ep):
        ok, reason = base_verify(ws, ep)
        rules = check_harness_rules(task, ep)
        if not ok:
            return False, f"{reason}; rules: {_rules_summary(rules)}"
        for k, (rok, _) in rules.items():
            if not rok:
                return False, f"rule {k.upper()} violated"
        return True, f"base pass; rules: {_rules_summary(rules)}"
    return verify


ALL_TASKS = build_tasks()
TASKS_BY_ID = {t.id: t for t in ALL_TASKS}
E2E_TASKS = [t for t in ALL_TASKS if t.harness_e2e]