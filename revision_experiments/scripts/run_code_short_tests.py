#!/usr/bin/env python3
"""Run CPU/synthetic preflight checks and write one machine-readable report."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from revision_experiments.scripts.audit_artifacts import prometheus_evidence_ready
from revision_experiments.scripts.execution_gates import (
    validate_execution_gates, validate_model_gate_evidence,
)
from revision_experiments.scripts.matrix import expand_matrix, load_matrix
from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "revision_experiments/config"
EXPECTED_MATRIX_COUNTS = {
    "core_500step_matrix.yaml": 48,
    "mechanism_matrix.yaml": 48,
    "downstream_matrix.yaml": 63,
    "scope_matrix.yaml": 12,
    "baseline_search_matrix.yaml": 63,
}


def git_state() -> tuple[str, bool]:
    git_dir = ROOT / ".git-reconstructed"
    command = ["git", f"--git-dir={git_dir}", f"--work-tree={ROOT}"]
    commit = subprocess.run(command + ["rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    status = subprocess.run(command + ["status", "--porcelain"], text=True, capture_output=True, check=True)
    return commit.stdout.strip(), bool(status.stdout.strip())


def run_check(name: str, command: list[str], *, expected_codes: tuple[int, ...] = (0,)) -> tuple[dict, str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True, capture_output=True, check=False,
    )
    output = completed.stdout + completed.stderr
    passed = completed.returncode in expected_codes
    return {
        "name": name, "passed": passed, "exit_code": completed.returncode,
        "expected_exit_codes": list(expected_codes),
        "duration_seconds": time.perf_counter() - started,
        "command": command, "output_tail": output[-2000:],
    }, output


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "revision_experiments/results/audits/code_short_test.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite code short-test report: {args.output}")

    commit, dirty = git_state()
    checks: list[dict] = []
    failures: list[str] = []
    if dirty:
        failures.append("worktree was not clean at short-test start")

    def checked(name: str, command: list[str], expected_codes: tuple[int, ...] = (0,)) -> str:
        record, output = run_check(name, command, expected_codes=expected_codes)
        checks.append(record)
        if not record["passed"]:
            failures.append(f"{name}: exit {record['exit_code']}, expected {expected_codes}")
        return output

    checked("compileall", [sys.executable, "-m", "compileall", "-q", "revision_experiments"])
    checked("download_script_bash_syntax", ["bash", "-n", "revision_experiments/scripts/download_models.sh"])
    unit_output = checked(
        "full_unit_suite",
        [sys.executable, "-m", "unittest", "discover", "-s", "revision_experiments/tests", "-p", "test_*.py", "-v"],
    )
    match = re.search(r"Ran\s+(\d+)\s+tests?", unit_output)
    unit_test_count = int(match.group(1)) if match else None
    if unit_test_count is None:
        failures.append("full_unit_suite: could not parse test count")

    entrypoints = sorted(
        path for path in (ROOT / "revision_experiments/scripts").glob("*.py")
        if "if __name__ == \"__main__\"" in path.read_text(encoding="utf-8")
    )
    help_failures = []
    help_started = time.perf_counter()
    for path in entrypoints:
        completed = subprocess.run(
            [sys.executable, str(path), "--help"], cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)}, text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            help_failures.append({"script": path.name, "exit_code": completed.returncode,
                                  "output_tail": (completed.stdout + completed.stderr)[-500:]})
    checks.append({
        "name": "all_cli_help", "passed": not help_failures,
        "entrypoint_count": len(entrypoints), "failures": help_failures,
        "duration_seconds": time.perf_counter() - help_started,
    })
    if help_failures:
        failures.append(f"all_cli_help: {len(help_failures)} failures")

    matrix_counts = {}
    for filename, expected_count in EXPECTED_MATRIX_COUNTS.items():
        matrix = load_matrix(CONFIG / filename)
        runs = expand_matrix(matrix)
        count = len(runs)
        matrix_counts[filename] = count
        try:
            require(count == expected_count, f"{filename}: {count} != {expected_count}")
            require(len({run["run_id"] for run in runs}) == count, f"{filename}: duplicate run IDs")
            require(matrix.get("enabled") is False, f"{filename}: formal matrix unexpectedly enabled")
            output = checked(
                f"matrix_list_{filename}",
                [sys.executable, "revision_experiments/scripts/run_matrix.py", "--matrix", str(CONFIG / filename), "--list"],
            )
            require(json.loads(output)["count"] == expected_count, f"{filename}: CLI list count mismatch")
        except (RuntimeError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
    smoke_matrix = CONFIG / "smoke/integration_smoke.yaml"
    smoke_runs = expand_matrix(load_matrix(smoke_matrix))
    try:
        require(len(smoke_runs) == 3, "integration smoke matrix must contain exactly three runs")
        output = checked(
            "matrix_list_integration_smoke",
            [sys.executable, "revision_experiments/scripts/run_matrix.py", "--matrix", str(smoke_matrix),
             "--list", "--smoke"],
        )
        require(json.loads(output)["count"] == 3, "integration smoke CLI list count mismatch")
    except (RuntimeError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    with tempfile.TemporaryDirectory(prefix="revision_code_short_test_", dir="/tmp") as temporary:
        temporary_root = Path(temporary)
        initialization_output = temporary_root / "initialization.parquet"
        checked(
            "synthetic_initialization_audit",
            [sys.executable, "revision_experiments/scripts/audit_initialization.py", "--shape", "4", "64",
             "--seeds", "7", "--output", str(initialization_output)],
        )
        try:
            frame = pd.read_parquet(initialization_output)
            require(len(frame) == 8 and frame["method"].nunique() == 8, "synthetic initializer row coverage mismatch")
            require(initialization_output.with_suffix(".csv").is_file(), "synthetic initializer CSV is missing")
            require(initialization_output.with_name("initialization_diagnostic.svg").is_file(), "initializer diagnostic SVG is missing")
            require(initialization_output.with_name("initialization_diagnostic.png").is_file(), "initializer diagnostic PNG is missing")
        except Exception as exc:
            failures.append(f"synthetic_initialization_audit: {exc}")
        checked(
            "completeness_dry_audit",
            [sys.executable, "revision_experiments/scripts/check_completeness.py",
             "--output", str(temporary_root / "completeness.json")],
        )

    disabled_matrix = CONFIG / "core_500step_matrix.yaml"
    first_run = expand_matrix(load_matrix(disabled_matrix))[0]["run_id"]
    output = checked(
        "formal_training_fail_closed",
        [sys.executable, "revision_experiments/scripts/train_revision.py", "--matrix", str(disabled_matrix),
         "--run-id", first_run, "--device", "cuda:0"],
        expected_codes=(1,),
    )
    if "Training matrix is disabled" not in output:
        failures.append("formal_training_fail_closed: did not fail at the disabled-matrix guard")
    output = checked(
        "formal_evaluation_fail_closed",
        [sys.executable, "revision_experiments/scripts/evaluate_checkpoints.py",
         "--run-dir", str(ROOT / "revision_experiments/results/runs/nonexistent-short-test"),
         "--checkpoint", "500", "--task", "cmmlu", "--device", "cuda:0"],
        expected_codes=(1,),
    )
    if "Invalid completed run directory" not in output:
        failures.append("formal_evaluation_fail_closed: invalid run was not rejected before model loading")

    model_errors, model_summary = validate_model_gate_evidence(ROOT)
    judge_ready, judge_errors = prometheus_evidence_ready(ROOT / "revision_experiments/results/audits")
    formal_gates = validate_execution_gates(ROOT)
    checks.append({
        "name": "existing_model_evidence", "passed": not model_errors and judge_ready,
        "model_summary": model_summary, "prometheus_ready": judge_ready,
        "errors": model_errors + judge_errors,
    })
    if model_errors or not judge_ready:
        failures.append("existing_model_evidence: " + "; ".join(model_errors + judge_errors))

    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": commit, "worktree_clean_at_start": not dirty,
        "passed": not failures, "unit_test_count": unit_test_count,
        "entrypoint_count": len(entrypoints), "matrix_counts": matrix_counts,
        "formal_training_ready": formal_gates["ready_for_formal_training"],
        "formal_gate_errors": formal_gates["errors"],
        "scope": "CPU/static/synthetic tests; does not execute 7B training or formal evaluation",
        "checks": checks, "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["report_sha256"] = file_sha256(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
