"""Resource-limited subprocess evaluator for MBPP candidates.

This is a safety boundary for accidental loops and memory explosions, not a
container-grade security sandbox. Formal evaluation should run it inside an
offline container/user namespace as documented in HANDOFF.md.
"""

from __future__ import annotations

import ast
import os
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


_NAME_ERROR = re.compile(r"NameError: name '([A-Za-z_][A-Za-z0-9_]*)' is not defined")


def _defined_function_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _run_program(program: str, directory: str, timeout_seconds: int) -> dict:
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program], cwd=Path(directory),
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            text=True, capture_output=True, timeout=timeout_seconds,
            preexec_fn=_limits,
        )
        return {
            "passed": completed.returncode == 0, "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:], "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {"passed": False, "returncode": None, "stdout": (exc.stdout or "")[-4000:],
                "stderr": (exc.stderr or "")[-4000:], "timed_out": True}


def evaluate_candidate(code: str, setup: str, tests: list[str], timeout_seconds: int = 10) -> dict:
    program = "\n".join([setup or "", code, *tests])
    # Generated candidates may contain null bytes (decoder artifacts); they are
    # never valid Python source and cannot pass through argv. Strip them so the
    # candidate fails on its own merits (syntax error) instead of crashing the
    # evaluation loop.
    program = program.replace("\x00", "")
    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as directory:
        result = _run_program(program, directory, timeout_seconds)
        result["passed_strict"] = result["passed"]
        if result["passed"] or result["timed_out"]:
            return result
        # Function-name aliasing (conventional MBPP harness reading): the frozen
        # 3-shot prompt does not disclose the canonical name the hidden tests
        # call, so a correct solution under a self-chosen name otherwise fails
        # with NameError. When the missing name is exactly the tested entry
        # point and the candidate defined exactly one top-level function, bind
        # that name to the candidate and retry once. Record the aliasing so the
        # strict (no-alias) outcome stays available alongside.
        match = _NAME_ERROR.search(result["stderr"])
        if not match:
            return result
        expected = match.group(1)
        if not any(re.search(rf"\b{re.escape(expected)}\s*\(", test) for test in tests):
            return result
        definitions = _defined_function_names(code)
        if len(definitions) != 1:
            return result
        result["name_alias"] = {"expected": expected, "aliased_from": definitions[0]}
        aliased = "\n".join([setup or "", code, f"{expected} = {definitions[0]}", *tests])
        retry = _run_program(aliased, directory, timeout_seconds)
        retry["passed_strict"] = result["passed_strict"]
        retry["name_alias"] = result["name_alias"]
        return retry
