#!/usr/bin/env python3
"""Expand experiment matrices and launch only after explicit gate acknowledgement."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from revision_experiments.scripts.baseline_selection import verify_selection_manifest
from revision_experiments.scripts.execution_gates import require_execution_gates
from revision_experiments.scripts.matrix import expand_matrix, load_matrix
from revision_experiments.scripts.training_support import MODEL_PATHS
from revision_experiments.scripts.verify_model_architecture import (
    bind_model_identity, resolve_projection_contract, validate_scope_contract,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "revision_experiments" / "results" / "runs"
SMOKE_RESULTS = ROOT / "revision_experiments" / "results" / "smoke" / "runs"
BASELINE_SELECTED = ROOT / "revision_experiments/results/aggregate/baseline_selected_configs.yaml"


def validate_run_limit(limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise RuntimeError("--limit must be a positive integer")


def command_for(run: dict, matrix_path: Path, device: str, *, smoke: bool = False) -> list[str]:
    command = [
        sys.executable, str(ROOT / "revision_experiments/scripts/train_revision.py"),
        "--matrix", str(matrix_path), "--run-id", run["run_id"], "--device", device,
    ]
    if smoke:
        command.append("--smoke")
    return command


def validate_baseline_execution_phase(
    config: dict,
    runs_root: str | Path,
    selected_path: str | Path,
    limit: int | None,
) -> str:
    if config.get("matrix_name") != "baseline_fairness_qwen_table3":
        return "not_baseline"
    validate_run_limit(limit)
    screening_count = len([run for run in expand_matrix(config) if run.get("section") == "screening"])
    selected_path = Path(selected_path)
    if not selected_path.is_file():
        if limit is None or limit > screening_count:
            raise RuntimeError(
                f"Baseline screening phase requires --limit {screening_count}; "
                "final runs are locked until validated selection outputs exist"
            )
        return "screening"
    verify_selection_manifest(config, runs_root, selected_path)
    return "final_ready"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="Print expanded inventory; performs no training")
    parser.add_argument("--execute", action="store_true", help="Launch incomplete runs serially")
    parser.add_argument("--acknowledge-gates", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true", help="Use only a purpose=integration_smoke config and isolated outputs")
    args = parser.parse_args()
    if args.list == args.execute:
        parser.error("choose exactly one of --list or --execute")
    try:
        validate_run_limit(args.limit)
    except RuntimeError as exc:
        parser.error(str(exc))
    config = load_matrix(args.matrix)
    is_smoke_matrix = config.get("purpose") == "integration_smoke"
    if bool(args.smoke) != bool(is_smoke_matrix):
        parser.error("--smoke must be used exactly with a purpose=integration_smoke config")
    runs = expand_matrix(config)
    if args.limit is not None:
        runs = runs[: args.limit]
    if args.list:
        print(json.dumps({"matrix": config["matrix_name"], "count": len(runs), "runs": runs}, indent=2))
        return 0
    if not config.get("enabled", False):
        raise SystemExit("Matrix is disabled. Set enabled: true only after Gate 1-R and Gate 2 pass.")
    if not args.acknowledge_gates:
        raise SystemExit("Refusing to train without --acknowledge-gates")
    try:
        require_execution_gates(ROOT, require_integration_smokes=not args.smoke)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    results_root = SMOKE_RESULTS if args.smoke else RESULTS
    if not args.smoke:
        try:
            validate_baseline_execution_phase(config, RESULTS, BASELINE_SELECTED, args.limit)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    if config.get("matrix_name") == "scope":
        try:
            audit = resolve_projection_contract(
                MODEL_PATHS["openpangu"], rank=int(config["training_defaults"]["lora_rank"])
            )
            audit = bind_model_identity(audit, "openpangu")
            validate_scope_contract(audit, config)
        except RuntimeError as exc:
            raise SystemExit(f"Scope architecture preflight failed: {exc}") from exc
    results_root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run_dir = results_root / run["run_id"]
        if (run_dir / "COMPLETED").is_file():
            print(f"SKIP completed {run['run_id']}")
            continue
        command = command_for(run, args.matrix.resolve(), args.device, smoke=args.smoke)
        print("RUN", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
