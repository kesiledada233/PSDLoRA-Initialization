#!/usr/bin/env python3
"""Fail-closed audit of the isolated pre-matrix GPU integration smokes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from safetensors import safe_open

from revision_experiments.initializers.lora_one import OFFICIAL_COMMIT
from revision_experiments.scripts.aggregate_results import (
    _validate_initialization_audit,
    _validate_mechanism_artifacts,
)
from revision_experiments.scripts.evaluate_checkpoints import smoke_evaluation_paths
from revision_experiments.scripts.matrix import checkpoint_steps_for_run, expand_matrix, load_matrix
from revision_experiments.scripts.schema import canonical_hash, file_sha256, validate_run_directory
from revision_experiments.scripts.training_support import MODEL_IDENTIFIERS, MODEL_PATHS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "revision_experiments/config/smoke/integration_smoke.yaml"
DEFAULT_RUNS = ROOT / "revision_experiments/results/smoke/runs"
DEFAULT_OUTPUT = ROOT / "revision_experiments/results/audits/integration_smoke.json"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSONL {path}: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected non-empty JSON objects in {path}")
    return rows


def _checkpoint_files(run_dir: Path, matrix: dict, run: dict) -> list[Path]:
    files = []
    for step in sorted(checkpoint_steps_for_run(matrix, run)):
        checkpoint = run_dir / "checkpoints" / f"step_{step:06d}"
        config = checkpoint / "adapter_config.json"
        weights = checkpoint / "adapter_model.safetensors"
        if not config.is_file() or not weights.is_file():
            raise RuntimeError(f"incomplete smoke checkpoint: {checkpoint}")
        try:
            with safe_open(weights, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if not keys or not any("lora_A" in key for key in keys) or not any("lora_B" in key for key in keys):
                    raise RuntimeError("checkpoint has no paired LoRA A/B tensors")
        except Exception as exc:
            raise RuntimeError(f"unreadable smoke checkpoint {weights}: {exc}") from exc
        files.extend([config, weights])
    return files


def _mechanism_contract(run: dict) -> dict:
    model_config = _read_json(MODEL_PATHS[run["model"]] / "config.json")
    return {
        "logging": run["logging"],
        "target_modules": tuple(run["target_modules"]),
        "gradient_accumulation_steps": int(run["training"]["gradient_accumulation_steps"]),
        "num_hidden_layers": int(model_config["num_hidden_layers"]),
        "hidden_size": int(model_config["hidden_size"]),
        "num_attention_heads": int(model_config["num_attention_heads"]),
        "num_key_value_heads": int(model_config["num_key_value_heads"]),
        "lora_rank": int(run["training"]["lora_rank"]),
    }


def _audit_mbpp_output(run: dict, checkpoint: int, limit: int = 3) -> tuple[dict, list[Path]]:
    paths = smoke_evaluation_paths(run["run_id"], checkpoint, "mbpp", limit)
    result = _read_json(paths["result"])
    predictions = _read_jsonl(paths["prediction"])
    if result.get("formal_result") is not False or result.get("sample_count") != limit:
        raise RuntimeError("MBPP smoke result is not an isolated exact-size smoke")
    if result.get("run_id") != run["run_id"] or result.get("checkpoint") != checkpoint:
        raise RuntimeError("MBPP smoke result identity mismatch")
    if result.get("prediction_sha256") != file_sha256(paths["prediction"]):
        raise RuntimeError("MBPP smoke prediction hash mismatch")
    if len(predictions) != limit or [row.get("sample_index") for row in predictions] != list(range(limit)):
        raise RuntimeError("MBPP smoke predictions do not cover the ordered first three samples")
    for row in predictions:
        execution = row.get("execution")
        if not isinstance(execution, dict) or not isinstance(execution.get("passed"), bool):
            raise RuntimeError("MBPP smoke prediction lacks an executor result")
        if not isinstance(execution.get("timed_out"), bool) or execution.get("returncode") is not None and not isinstance(execution.get("returncode"), int):
            raise RuntimeError("MBPP smoke executor result is malformed")
    metric = result.get("metrics", {}).get("pass_at_1")
    observed = sum(bool(row["execution"]["passed"]) for row in predictions) / limit
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isclose(float(metric), observed):
        raise RuntimeError("MBPP smoke metric does not match raw executor results")
    return {"sample_count": limit, "passes": int(observed * limit), "pass_at_1": observed}, list(paths.values())


def audit_smokes(matrix_path: Path, runs_root: Path) -> dict:
    matrix = load_matrix(matrix_path)
    if matrix.get("purpose") != "integration_smoke":
        raise RuntimeError("integration smoke audit requires purpose=integration_smoke")
    cases = []
    artifact_paths = [matrix_path]
    for run in expand_matrix(matrix):
        run_dir = runs_root / run["run_id"]
        errors = validate_run_directory(run_dir)
        if errors or not (run_dir / "COMPLETED").is_file() or (run_dir / "FAILED.json").exists():
            raise RuntimeError(f"invalid smoke run {run['run_id']}: {'; '.join(errors) or 'terminal state'}")
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        expected_config = {
            "matrix": matrix["matrix_name"], **run,
            "effective_method": run["method"],
            "effective_learning_rate": float(run["training"]["learning_rate"]),
            "formal_result": False,
        }
        if config != expected_config:
            raise RuntimeError(f"smoke config does not match its matrix declaration: {run['run_id']}")
        metadata = _read_json(run_dir / "metadata.json")
        if metadata.get("config_hash") != canonical_hash(config):
            raise RuntimeError(f"smoke metadata/config hash mismatch: {run['run_id']}")
        if metadata.get("model_checkpoint") != MODEL_IDENTIFIERS[run["model"]]:
            raise RuntimeError(f"smoke model revision mismatch: {run['run_id']}")
        raw = _read_jsonl(run_dir / "raw_loss.jsonl")
        timing = _read_jsonl(run_dir / "timing.jsonl")
        expected_steps = list(range(1, int(run["max_steps"]) + 1))
        if [row.get("step") for row in raw] != expected_steps or [row.get("step") for row in timing] != expected_steps:
            raise RuntimeError(f"smoke raw/timing steps are not contiguous: {run['run_id']}")
        if not all(math.isfinite(float(row["train_loss"])) for row in raw):
            raise RuntimeError(f"smoke loss is non-finite: {run['run_id']}")
        init = _read_json(run_dir / "initialization_stats.json")
        init_errors = _validate_initialization_audit(init, config)
        if init_errors:
            raise RuntimeError(f"invalid smoke initialization audit {run['run_id']}: {'; '.join(init_errors)}")
        checkpoint_files = _checkpoint_files(run_dir, matrix, run)
        case = {"case": run["case"], "run_id": run["run_id"], "passed": True}
        if run["case"] == "lora_one_initialization":
            modules = init.get("initialized_modules")
            if init.get("source_commit") != OFFICIAL_COMMIT or not isinstance(modules, list) or not modules:
                raise RuntimeError("LoRA-One smoke lacks frozen-source initialization evidence")
            if not all(float(row.get("a_norm", 0)) > 0 and float(row.get("b_norm", 0)) > 0 for row in modules):
                raise RuntimeError("LoRA-One smoke did not initialize non-zero A/B factors")
            case["initialized_modules"] = len(modules)
        elif run["case"] == "gradient_logger_20step":
            mechanism_errors = _validate_mechanism_artifacts(
                run_dir, int(run["max_steps"]), _mechanism_contract(run),
            )
            if mechanism_errors:
                raise RuntimeError("invalid 20-step gradient smoke: " + "; ".join(mechanism_errors))
            case["gradient_steps"] = int(run["max_steps"]) + 1
        elif run["case"] == "mbpp_adapter_and_executor":
            checkpoint = next(iter(checkpoint_steps_for_run(matrix, run)))
            case["mbpp"], mbpp_files = _audit_mbpp_output(run, checkpoint)
            artifact_paths.extend(mbpp_files)
        artifact_paths.extend([
            run_dir / name for name in (
                "metadata.json", "config.yaml", "raw_loss.jsonl", "timing.jsonl",
                "initialization_stats.json", "summary.json", "COMPLETED",
            )
        ] + checkpoint_files)
        if (run_dir / "gradients").is_dir():
            artifact_paths.extend(path for path in (run_dir / "gradients").rglob("*") if path.is_file())
            artifact_paths.append(run_dir / "gradient_diagnostics.jsonl")
        cases.append(case)
    records = []
    for path in sorted(set(path.resolve() for path in artifact_paths)):
        records.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)})
    return {
        "schema_version": 1,
        "passed": True,
        "matrix": str(matrix_path.resolve().relative_to(ROOT)),
        "matrix_sha256": file_sha256(matrix_path),
        "cases": cases,
        "artifacts": records,
        "formal_results": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite integration smoke report: {args.output}")
    try:
        report = audit_smokes(args.matrix.resolve(), args.runs_root.resolve())
    except RuntimeError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
