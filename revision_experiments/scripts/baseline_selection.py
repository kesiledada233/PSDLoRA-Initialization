"""Strict validation-only selection for the equal-budget baseline search."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path

import yaml

from revision_experiments.scripts.matrix import expand_matrix
from revision_experiments.scripts.metrics import raw_trapezoid_auc, trapezoid_auc_at_steps
from revision_experiments.scripts.schema import canonical_hash, dataset_split_hash, file_sha256, validate_run_directory


CSV_FIELDS = [
    "model", "task", "search_method", "effective_method", "variant", "search_kind", "search_value",
    "search_space", "seed", "run_id", "learning_rate", "central_learning_rate", "target_modules",
    "trainable_parameter_count", "total_parameter_count", "adapter_initialization_seconds",
    "gradient_estimation_seconds", "initialization_seconds_total", "validation_auc500",
    "validation_final_loss", "validation_guardrail_metric", "validation_guardrail_value",
    "validation_examples_evaluated", "raw_train_auc500", "selection_rank", "selected",
    "git_commit", "config_hash", "dataset_split_hash", "implementation", "official_repository",
    "implementation_version_or_commit", "initialization_data_requirement", "central_learning_rate_reference",
]


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSONL artifact {path}: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Expected non-empty JSON objects in {path}")
    return rows


def _finite_number(value, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise RuntimeError(f"{label} must be >= {minimum}")
    return number


def _expected_validation_steps(matrix: dict) -> list[int]:
    training = matrix["training_defaults"]
    interval = int(training["validation_interval"])
    endpoint = int(training["validation_max_step"])
    screening_steps = int(matrix["screening"]["max_steps"])
    if interval <= 0 or endpoint != screening_steps or endpoint % interval:
        raise RuntimeError("Validation schedule must span fixed steps 0..screening.max_steps at an exact interval")
    return list(range(0, endpoint + 1, interval))


def _search_space(matrix: dict, search_method: str) -> list[float]:
    search = matrix["screening"]["methods"][search_method]
    if "learning_rate_multipliers" in search:
        return [float(value) for value in search["learning_rate_multipliers"]]
    return [float(value) for value in search["alpha_values"]]


def _provenance_row(matrix: dict, search_method: str) -> dict:
    provenance = matrix["provenance"]["methods"][search_method]
    version = provenance.get("implementation_commit") or provenance.get("implementation_version")
    if not version:
        raise RuntimeError(f"Missing implementation version/commit provenance for {search_method}")
    return {
        "implementation": provenance["implementation"],
        "official_repository": provenance.get("official_repository", "project implementation"),
        "implementation_version_or_commit": version,
        "initialization_data_requirement": provenance["initialization_data_requirement"],
        "central_learning_rate_reference": provenance["central_learning_rate_reference"],
    }


def _validate_screening_run(matrix: dict, expected: dict, run_dir: Path, validation_steps: list[int]) -> dict:
    errors = validate_run_directory(run_dir)
    if errors or not (run_dir / "COMPLETED").is_file():
        detail = "; ".join(errors) or "missing COMPLETED marker"
        raise RuntimeError(f"Invalid screening run {expected['run_id']}: {detail}")
    required = ("summary.json", "validation_loss.jsonl")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Invalid screening run {expected['run_id']}: missing {', '.join(missing)}")

    metadata = _read_json(run_dir / "metadata.json")
    summary = _read_json(run_dir / "summary.json")
    init = _read_json(run_dir / "initialization_stats.json")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid config.yaml for {expected['run_id']}")
    expected_config = {
        "matrix": matrix["matrix_name"], **expected,
        "effective_method": expected["method"],
        "effective_learning_rate": float(expected["learning_rate"]),
    }
    if config != expected_config:
        mismatches = sorted(
            key for key in set(config) | set(expected_config)
            if config.get(key) != expected_config.get(key)
        )
        raise RuntimeError(f"Screening config mismatch for {expected['run_id']}: {', '.join(mismatches)}")
    if metadata.get("config_hash") != canonical_hash(config):
        raise RuntimeError(f"metadata config_hash does not bind config.yaml for {expected['run_id']}")
    metadata_expected = {
        "run_id": expected["run_id"], "seed": expected["seed"], "max_steps": expected["max_steps"],
        "method": expected["method"], "target_modules": expected["target_modules"],
        "effective_method": expected["method"], "effective_learning_rate": float(expected["learning_rate"]),
    }
    for key, value in metadata_expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Metadata identity mismatch for {expected['run_id']}: {key}")
    expected_split_hash = dataset_split_hash(expected["task"], uses_validation_split=True)
    if metadata.get("dataset_split_hash") != expected_split_hash:
        raise RuntimeError(f"dataset split hash does not match committed frozen split for {expected['run_id']}")
    declared_peft = str(matrix["provenance"]["peft_version"])
    installed_peft = importlib.metadata.version("peft")
    recorded_peft = metadata.get("environment", {}).get("packages", {}).get("peft")
    if installed_peft != declared_peft or recorded_peft != declared_peft:
        raise RuntimeError(
            f"PEFT version mismatch for {expected['run_id']}: "
            f"declared={declared_peft}, installed={installed_peft}, recorded={recorded_peft}"
        )

    expected_training_steps = list(range(1, int(expected["max_steps"]) + 1))
    raw_rows = _read_jsonl(run_dir / "raw_loss.jsonl")
    timing_rows = _read_jsonl(run_dir / "timing.jsonl")
    raw_steps = [row.get("step") for row in raw_rows]
    timing_steps = [row.get("step") for row in timing_rows]
    if raw_steps != expected_training_steps or timing_steps != expected_training_steps:
        raise RuntimeError(f"Invalid training steps for {expected['run_id']}: expected contiguous 1..{expected['max_steps']}")
    if summary.get("steps_logged") != int(expected["max_steps"]):
        raise RuntimeError(f"summary.steps_logged mismatch for {expected['run_id']}")
    raw_losses = [_finite_number(row.get("train_loss"), "train_loss", minimum=0.0) for row in raw_rows]
    for row in timing_rows:
        _finite_number(row.get("step_time_seconds"), "step_time_seconds", minimum=0.0)
    computed_raw_auc = raw_trapezoid_auc(raw_losses, 0, int(expected["max_steps"]))
    stored_raw_auc = _finite_number(summary.get("raw_auc500"), "raw_auc500", minimum=0.0)
    if not math.isclose(stored_raw_auc, computed_raw_auc, rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError(f"raw_auc500 does not match raw_loss.jsonl for {expected['run_id']}")

    validation = _read_jsonl(run_dir / "validation_loss.jsonl")
    observed_steps = [row.get("step") for row in validation]
    if observed_steps != validation_steps:
        raise RuntimeError(
            f"Invalid validation steps for {expected['run_id']}: expected {validation_steps}, got {observed_steps}"
        )
    values = [_finite_number(row.get("validation_loss"), "validation_loss", minimum=0.0) for row in validation]
    expected_validation_examples = int(matrix["validation_split"]["counts"][expected["task"]])
    observed_validation_examples = [row.get("validation_examples_evaluated") for row in validation]
    if observed_validation_examples != [expected_validation_examples] * len(validation):
        raise RuntimeError(f"Validation did not score the complete frozen holdout for {expected['run_id']}")
    auc = trapezoid_auc_at_steps(observed_steps, values, validation_steps[0], validation_steps[-1])
    stored_auc = _finite_number(summary.get("validation_auc500"), "validation_auc500", minimum=0.0)
    final_loss = _finite_number(summary.get("validation_final_loss"), "validation_final_loss", minimum=0.0)
    if not math.isclose(stored_auc, auc, rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError(f"validation_auc500 does not match validation_loss.jsonl for {expected['run_id']}")
    if not math.isclose(final_loss, values[-1], rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError(f"validation_final_loss does not match validation_loss.jsonl for {expected['run_id']}")
    if (
        summary.get("validation_guardrail_metric") != "validation_loss"
        or summary.get("validation_examples_evaluated") != expected_validation_examples
        or not math.isclose(
            _finite_number(summary.get("validation_guardrail_value"), "validation_guardrail_value", minimum=0.0),
            final_loss, rel_tol=1e-10, abs_tol=1e-8,
        )
    ):
        raise RuntimeError(f"Invalid validation guardrail for {expected['run_id']}")

    trainable = init.get("trainable_parameter_count")
    total = init.get("total_parameter_count")
    if isinstance(trainable, bool) or not isinstance(trainable, int) or trainable <= 0:
        raise RuntimeError(f"Missing positive trainable_parameter_count for {expected['run_id']}")
    if isinstance(total, bool) or not isinstance(total, int) or total < trainable:
        raise RuntimeError(f"Invalid total_parameter_count for {expected['run_id']}")
    adapter_seconds = _finite_number(init.get("adapter_initialization_seconds"), "adapter_initialization_seconds", minimum=0.0)
    gradient_seconds = _finite_number(init.get("gradient_estimation_seconds", 0.0), "gradient_estimation_seconds", minimum=0.0)
    total_seconds = _finite_number(init.get("initialization_seconds_total"), "initialization_seconds_total", minimum=0.0)
    if not math.isclose(total_seconds, adapter_seconds + gradient_seconds, rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError(f"Initialization timing components do not sum for {expected['run_id']}")
    if expected["search_method"] == "lora_one":
        lora_contract = {
            "gradient_batches": expected["gradient_batches"],
            "gradient_batch_size": expected["gradient_batch_size"],
            "gradient_max_length": expected["gradient_max_length"],
        }
        if (
            any(init.get(key) != value for key, value in lora_contract.items())
            or init.get("source_commit") != matrix["provenance"]["lora_one_commit"]
            or float(init.get("stable_gamma", -1)) != float(expected["stable_gamma"])
        ):
            raise RuntimeError(f"LoRA-One provenance/cost contract failed for {expected['run_id']}")

    search_kind = "alpha" if "alpha" in expected else "learning_rate_multiplier"
    search_value = expected.get("alpha", expected.get("learning_rate_multiplier"))
    row = {
        "model": expected["model"], "task": expected["task"], "search_method": expected["search_method"],
        "effective_method": expected["method"], "variant": expected["variant"], "search_kind": search_kind,
        "search_value": float(search_value), "search_space": json.dumps(_search_space(matrix, expected["search_method"])),
        "seed": expected["seed"], "run_id": expected["run_id"], "learning_rate": float(expected["learning_rate"]),
        "central_learning_rate": float(expected["central_learning_rate"]),
        "target_modules": json.dumps(expected["target_modules"]), "trainable_parameter_count": trainable,
        "total_parameter_count": total, "adapter_initialization_seconds": adapter_seconds,
        "gradient_estimation_seconds": gradient_seconds, "initialization_seconds_total": total_seconds,
        "validation_auc500": stored_auc, "validation_final_loss": final_loss,
        "validation_guardrail_metric": "validation_loss", "validation_guardrail_value": final_loss,
        "validation_examples_evaluated": expected_validation_examples,
        "raw_train_auc500": stored_raw_auc, "selection_rank": None, "selected": False,
        "git_commit": metadata["git_commit"], "config_hash": metadata["config_hash"],
        "dataset_split_hash": metadata["dataset_split_hash"],
    }
    row.update(_provenance_row(matrix, expected["search_method"]))
    return row


def collect_screening_results(matrix: dict, runs_root: str | Path) -> tuple[list[dict], dict]:
    if matrix.get("matrix_name") != "baseline_fairness_qwen_table3":
        raise RuntimeError("Selector requires the baseline_fairness_qwen_table3 matrix")
    expected = [run for run in expand_matrix(matrix) if run.get("section") == "screening"]
    runs_root = Path(runs_root)
    missing = [run["run_id"] for run in expected if not (runs_root / run["run_id"]).is_dir()]
    if missing:
        raise RuntimeError(f"Missing screening runs ({len(missing)}/{len(expected)}): {', '.join(missing[:5])}")
    validation_steps = _expected_validation_steps(matrix)
    rows = [_validate_screening_run(matrix, run, runs_root / run["run_id"], validation_steps) for run in expected]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["search_method"]), []).append(row)
    for task in matrix["tasks"]:
        task_hashes = {row["dataset_split_hash"] for row in rows if row["task"] == task}
        if len(task_hashes) != 1:
            raise RuntimeError(f"Inconsistent dataset split hash across screening methods for {task}")
    expected_groups = {(task, method) for task in matrix["tasks"] for method in matrix["screening"]["methods"]}
    if set(grouped) != expected_groups or any(len(group) != 3 for group in grouped.values()):
        raise RuntimeError("Screening budget is not exactly three trials per method/task")

    tasks: dict[str, dict] = {str(task): {} for task in matrix["tasks"]}
    for (task, search_method), group in grouped.items():
        ranked = sorted(group, key=lambda row: (row["validation_auc500"], row["validation_final_loss"], row["run_id"]))
        for rank, row in enumerate(ranked, start=1):
            row["selection_rank"] = rank
            row["selected"] = rank == 1
        winner = ranked[0]
        tasks[task][search_method] = {
            "effective_method": winner["effective_method"],
            "learning_rate": winner["learning_rate"],
            "central_learning_rate": winner["central_learning_rate"],
            "variant": winner["variant"],
            "search_kind": winner["search_kind"],
            "search_value": winner["search_value"],
            "selected_run_id": winner["run_id"],
            "validation_auc500": winner["validation_auc500"],
            "validation_final_loss": winner["validation_final_loss"],
            "validation_guardrail_metric": winner["validation_guardrail_metric"],
            "validation_guardrail_value": winner["validation_guardrail_value"],
            "validation_examples_evaluated": winner["validation_examples_evaluated"],
            "trainable_parameter_count": winner["trainable_parameter_count"],
            "initialization_seconds_total": winner["initialization_seconds_total"],
        }
    manifest_basis = [
        {key: row[key] for key in ("run_id", "config_hash", "dataset_split_hash", "validation_auc500")}
        for row in sorted(rows, key=lambda item: item["run_id"])
    ]
    selected = {
        "schema_version": 1,
        "matrix_name": matrix["matrix_name"],
        "selection_metric": matrix["screening"]["selection_metric"],
        "selection_direction": "minimize",
        "tie_breakers": ["validation_final_loss", "run_id"],
        "selection_uses_official_test": False,
        "screening_split": "frozen holdout removed from each task's original training split",
        "final_training_split": "full original training split",
        "screening_trial_count": len(rows),
        "screening_manifest_hash": canonical_hash(manifest_basis),
        "tasks": tasks,
    }
    return rows, selected


def write_selection_outputs(rows: list[dict], selected: dict, csv_path: str | Path, yaml_path: str | Path) -> None:
    if not rows or selected.get("screening_trial_count") != len(rows):
        raise RuntimeError("Refusing to write incomplete baseline selection outputs")
    if selected.get("selection_uses_official_test") is not False:
        raise RuntimeError("Refusing to write a selection based on official test data")
    if any(set(row) != set(CSV_FIELDS) for row in rows):
        raise RuntimeError("Baseline trial rows do not match the complete output schema")
    csv_path, yaml_path = Path(csv_path), Path(yaml_path)
    existing = [str(path) for path in (csv_path, yaml_path) if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite baseline selection output(s): {', '.join(existing)}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    yaml_tmp = yaml_path.with_name(yaml_path.name + ".tmp")
    try:
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        paired_selected = dict(selected)
        paired_selected["all_trials_file"] = csv_path.name
        paired_selected["all_trials_sha256"] = file_sha256(csv_tmp)
        yaml_tmp.write_text(yaml.safe_dump(paired_selected, sort_keys=False), encoding="utf-8")
        os.replace(csv_tmp, csv_path)
        os.replace(yaml_tmp, yaml_path)
    finally:
        for temporary in (csv_tmp, yaml_tmp):
            if temporary.exists():
                temporary.unlink()


def load_selected_manifest(path: str | Path, *, require_pair: bool = False) -> dict:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Validation-selected config does not exist: {path}")
    try:
        selected = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Invalid validation-selected config {path}: {exc}") from exc
    if not isinstance(selected, dict):
        raise RuntimeError(f"Invalid validation-selected config {path}: expected mapping")
    if require_pair:
        csv_name = selected.get("all_trials_file")
        expected_hash = selected.get("all_trials_sha256")
        if not isinstance(csv_name, str) or Path(csv_name).name != csv_name:
            raise RuntimeError("Selection manifest has invalid all-trials pair path")
        csv_path = path.parent / csv_name
        if not csv_path.is_file() or not isinstance(expected_hash, str) or file_sha256(csv_path) != expected_hash:
            raise RuntimeError("Selection manifest all-trials pair hash mismatch")
    return selected


def verify_selection_manifest(matrix: dict, runs_root: str | Path, selected_path: str | Path) -> dict:
    """Rebuild selection from current runs and require exact manifest agreement."""
    observed = load_selected_manifest(selected_path, require_pair=True)
    _, expected = collect_screening_results(matrix, runs_root)
    observed_core = {
        key: value for key, value in observed.items()
        if key not in {"all_trials_file", "all_trials_sha256"}
    }
    if observed_core != expected:
        raise RuntimeError("Selected configuration does not match current 48 screening artifacts")
    return observed


def resolve_selected_configuration(selected: dict, task: str, final_method: str) -> dict:
    if selected.get("matrix_name") != "baseline_fairness_qwen_table3" or selected.get("selection_metric") != "validation_auc500":
        raise RuntimeError("Selected configuration has incompatible matrix/metric provenance")
    if selected.get("selection_uses_official_test") is not False:
        raise RuntimeError("Refusing selected configuration that used official test data")
    manifest_hash = selected.get("screening_manifest_hash")
    if (
        selected.get("schema_version") != 1
        or selected.get("selection_direction") != "minimize"
        or selected.get("screening_trial_count") != 48
        or not isinstance(manifest_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None
    ):
        raise RuntimeError("Incomplete selection manifest provenance")
    search_method = "proposed" if final_method == "validation_selected_proposed" else final_method
    try:
        resolved = selected["tasks"][task][search_method]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Missing selected configuration for task={task}, method={search_method}") from exc
    required = {"effective_method", "learning_rate", "selected_run_id"}
    if not isinstance(resolved, dict) or required - resolved.keys():
        raise RuntimeError(f"Incomplete selected configuration for task={task}, method={search_method}")
    _finite_number(resolved["learning_rate"], "selected learning_rate", minimum=0.0)
    return dict(resolved, search_method=search_method)
