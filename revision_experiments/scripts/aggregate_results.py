#!/usr/bin/env python3
"""Build a fail-closed run inventory and traceable reviewer artifacts.

The inventory is operational evidence and is always safe to write. Scientific
tables and figures are emitted only when every required raw artifact validates.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from revision_experiments.scripts.baseline_selection import (
    CSV_FIELDS,
    collect_screening_results,
    load_selected_manifest,
    resolve_selected_configuration,
    verify_selection_manifest,
)
from revision_experiments.scripts.matrix import checkpoint_steps_for_run, expand_matrix, load_matrix
from revision_experiments.scripts.metrics import raw_trapezoid_auc, temporal_psd_slope
from revision_experiments.scripts.openpangu_cuda_compat import loader_provenance
from revision_experiments.scripts.schema import (
    canonical_hash, dataset_split_hash, file_sha256, formal_evaluation_config,
    formal_sample_entries, formal_sample_manifest, validate_run_directory,
)
from revision_experiments.scripts.training_support import MODEL_PATHS
from revision_experiments.scripts.verify_model_architecture import (
    bind_model_identity, resolve_projection_contract, target_module_contract, validate_scope_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "revision_experiments/config"
INVENTORY_COLUMNS = [
    "run_id", "model", "task", "method", "seed", "max_steps", "target",
    "matrix_membership", "matrix_membership_count", "analysis_contexts", "status", "terminal_state",
    "metadata_sha256", "config_hash", "git_commit", "expected_checkpoints",
    "expected_config_sha256", "available_checkpoints", "missing_checkpoints", "checkpoint_sha256",
    "expected_evaluations", "expected_evaluation_protocols",
    "available_evaluations", "missing_evaluations", "invalid_evaluations", "raw_auc500_reported",
    "raw_auc500_recomputed", "training_seconds_recomputed", "initialization_seconds_total",
    "failure_reason",
]
PAIR_KEYS = ["analysis_context", "model", "task", "seed", "endpoint", "metric"]
GROUP_KEYS = [
    "analysis_context", "model", "task", "method", "endpoint", "metric", "metric_direction", "unit",
]
METHOD_PALETTE = {
    "peft_default": "#4c78a8", "iid_matched": "#9c755f",
    "powerlaw_global_a03": "#f58518", "powerlaw_global_a06": "#e45756",
    "powerlaw_shuffle_a06": "#72b7b2", "powerlaw_row_a06": "#54a24b",
    "powerlaw_col_a06": "#b279a2", "fft_white_a0": "#bab0ac",
    "dora": "#59a14f", "pissa": "#edc948", "lora_one": "#af7aa1",
    "validation_selected_proposed": "#e15759",
}
CANONICAL_TABLE_NAMES = (
    "reviewer_seed_metrics.csv", "reviewer_summary_metrics.csv",
    "reviewer_paired_differences.csv", "reviewer_paired_difference_summary.csv",
    "initialization_statistics_table.csv", "step_zero_audit_table.csv",
    "downstream_metric_table.csv", "time_to_equivalent_table.csv",
    "all_linear_longer_run_table.csv", "early_psd_gradient_capture_table.csv",
    "baseline_search_supplementary_table.csv",
)
CANONICAL_FIGURE_STEMS = (
    "matched_scale_shuffle_auc500", "early_temporal_psd_gradient_capture",
    "task_metric_vs_steps_time",
)


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


def _finite(value, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise RuntimeError(f"{label} must be >= {minimum}")
    return number


def _evaluation_checkpoints(config: dict, run: dict) -> set[int]:
    """Resolve which saved checkpoints require formal evaluations.

    Default contract: every declared training checkpoint is evaluated. Two
    optional amendment fields restrict this without touching training or the
    per-run configuration hash:

    - ``evaluation_checkpoints`` (matrix level or run-section level): evaluate
      only this subset; it must be a subset of the declared training
      checkpoints.
    - ``step_zero_evaluation: shared_peft_default_first_seed``: require the
      step-0 evaluation only for the representative ``peft_default`` run at
      the section's smallest seed. Gate 2 (schema-v4 step-zero audit) proved
      exact functional equivalence of every B=0 method at step 0, so per-run
      step-0 evaluations would duplicate identical predictions.
    """
    training = checkpoint_steps_for_run(config, run)
    section_config = config.get(run.get("section"), {})
    section = section_config if isinstance(section_config, dict) else {}
    rules = config.get("evaluation_rules")
    if rules is not None:
        if not isinstance(rules, list):
            raise RuntimeError(f"matrix {config['matrix_name']} evaluation_rules must be a list")
        selector_fields = {
            "models": "model", "tasks": "task", "methods": "method", "seeds": "seed",
        }
        allowed_fields = {"name", "checkpoints", *selector_fields}
        declared = set()
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise RuntimeError(
                    f"matrix {config['matrix_name']} evaluation rule {index} must be an object"
                )
            unsupported = sorted(set(rule) - allowed_fields)
            if unsupported:
                raise RuntimeError(
                    f"matrix {config['matrix_name']} evaluation rule {index} has "
                    f"unsupported selectors: {unsupported}"
                )
            if "checkpoints" not in rule or not isinstance(rule["checkpoints"], list):
                raise RuntimeError(
                    f"matrix {config['matrix_name']} evaluation rule {index} needs checkpoints"
                )
            matches = True
            for selector, run_field in selector_fields.items():
                values = rule.get(selector)
                if values is not None:
                    if not isinstance(values, list):
                        raise RuntimeError(
                            f"matrix {config['matrix_name']} evaluation rule {index} "
                            f"selector {selector} must be a list"
                        )
                    if run[run_field] not in values:
                        matches = False
            if matches:
                declared.update(int(step) for step in rule["checkpoints"])
        unknown = declared - training
        if unknown:
            raise RuntimeError(
                f"matrix {config['matrix_name']} evaluation_checkpoints {sorted(unknown)} "
                f"are not declared training checkpoints for {run['run_id']}"
            )
        steps = training & declared
    else:
        declared = {int(step) for step in config.get("evaluation_checkpoints", [])}
        declared |= {int(step) for step in section.get("evaluation_checkpoints", [])}
    if rules is None and declared:
        unknown = declared - training
        if unknown:
            raise RuntimeError(
                f"matrix {config['matrix_name']} evaluation_checkpoints {sorted(unknown)} "
                f"are not declared training checkpoints for {run['run_id']}"
            )
        steps = training & declared
    elif rules is None:
        steps = set(training)
    policy = config.get("step_zero_evaluation") or section.get("step_zero_evaluation")
    if policy == "shared_peft_default_first_seed":
        seed_pool = section.get("seeds") or config.get("seeds") or []
        if not seed_pool:
            raise RuntimeError(
                f"matrix {config['matrix_name']} step_zero_evaluation policy needs a seed pool"
            )
        representative = run["method"] == "peft_default" and int(run["seed"]) == int(
            seed_pool[0]
        )
        if representative:
            if 0 not in training:
                raise RuntimeError(
                    f"matrix {config['matrix_name']} step_zero_evaluation representative "
                    f"{run['run_id']} does not save a step-0 checkpoint"
                )
            steps.add(0)
        else:
            steps.discard(0)
    return steps


def _evaluation_requirements(config: dict, run: dict) -> list[dict]:
    task = run["task"]
    checkpoints = sorted(_evaluation_checkpoints(config, run))
    metrics: list[str] = []
    protocol: dict = {}
    if isinstance(config.get("evaluation"), dict) and task in config["evaluation"]:
        protocol = dict(config["evaluation"][task])
    elif run.get("section") == "final":
        protocol = dict(config.get("final", {}).get("evaluation", {}).get(task, {}))
        if not protocol and task in config.get("final", {}).get("task_metrics", {}):
            protocol = {"metric": config["final"]["task_metrics"][task]}
    metrics = [protocol["metric"]] if "metric" in protocol else list(protocol.get("metrics", []))
    if metrics and (
        not isinstance(protocol.get("sample_count"), int)
        or protocol.get("sample_count", 0) <= 0
        or not isinstance(protocol.get("sample_selection"), str)
    ):
        raise RuntimeError(f"matrix {config['matrix_name']} has incomplete evaluation sampling protocol for {task}")
    requirements = []
    for checkpoint in checkpoints if metrics else []:
        split_hash = dataset_split_hash(task, uses_validation_split=False)
        sample_manifest = formal_sample_manifest(task)
        if sample_manifest["sample_count"] != int(protocol["sample_count"]):
            raise RuntimeError(
                f"matrix {config['matrix_name']} sample_count for {task} does not match frozen sample set: "
                f"{protocol['sample_count']} != {sample_manifest['sample_count']}"
            )
        evaluation_config = formal_evaluation_config(run["run_id"], checkpoint, task, protocol)
        sample_set_hash = canonical_hash(sample_manifest)
        requirements.append({
            "checkpoint": checkpoint, "task": task, "metrics": tuple(metrics), "protocol": protocol,
            "dataset_split_hash": split_hash, "evaluation_config": evaluation_config,
            "sample_set_hash": sample_set_hash, "bind_formal_samples": True,
        })
    return requirements


def _expected_config(matrix_name: str, run: dict, selected: dict | None) -> dict | None:
    effective_method = run["method"]
    effective_learning_rate = float(run.get("learning_rate", run["training"]["learning_rate"]))
    selection = None
    if run.get("selection_method"):
        if selected is None:
            return None
        selection = resolve_selected_configuration(selected, run["task"], run["method"])
        effective_method = selection["effective_method"]
        effective_learning_rate = float(selection["learning_rate"])
    payload = {
        "matrix": matrix_name, **run, "effective_method": effective_method,
        "effective_learning_rate": effective_learning_rate,
    }
    if selection is not None:
        payload["validation_selection"] = selection
    return payload


def _analysis_context(config: dict, run: dict) -> str:
    name = str(config["matrix_name"])
    if name == "baseline_fairness_qwen_table3":
        return "baseline_fairness_screening" if run.get("section") == "screening" else "baseline_fairness_final"
    if name == "scope":
        return f"scope_{run.get('section', 'unknown')}"
    return name


def discover_expected_runs(
    config_dir: str | Path = CONFIG_DIR, *, selected_path: str | Path | None = None,
) -> list[dict]:
    """Expand all matrices and merge duplicate exact run IDs losslessly."""
    merged: dict[str, dict] = {}
    paths = sorted(Path(config_dir).glob("*_matrix.yaml"))
    if not paths:
        raise RuntimeError(f"no experiment matrices found in {config_dir}")
    selected = None
    if selected_path is not None and Path(selected_path).is_file():
        selected = load_selected_manifest(selected_path, require_pair=True)
    for path in paths:
        config = load_matrix(path)
        matrix_name = str(config["matrix_name"])
        for run in expand_matrix(config):
            identity = {key: run[key] for key in ("run_id", "model", "task", "method", "seed", "max_steps", "target")}
            existing = merged.get(run["run_id"])
            if existing is None:
                existing = identity | {
                    "matrix_membership": set(), "expected_checkpoints": set(),
                    "expected_evaluations": {}, "expected_configurations": {},
                    "expected_dataset_split_hashes": set(),
                    "mechanism_required": False, "mechanism_contract": None, "analysis_contexts": set(),
                    "architecture_contract": None,
                }
                merged[run["run_id"]] = existing
            elif any(existing[key] != value for key, value in identity.items()):
                raise RuntimeError(f"conflicting identity for duplicate run ID {run['run_id']}")
            existing["matrix_membership"].add(matrix_name)
            existing["analysis_contexts"].add(_analysis_context(config, run))
            existing["expected_checkpoints"].update(checkpoint_steps_for_run(config, run))
            expected_config = _expected_config(matrix_name, run, selected)
            if expected_config is not None:
                existing["expected_configurations"][canonical_hash(expected_config)] = expected_config
            existing["expected_dataset_split_hashes"].add(dataset_split_hash(
                run["task"], uses_validation_split=(run.get("section") == "screening")
            ))
            for requirement in _evaluation_requirements(config, run):
                key = (requirement["checkpoint"], requirement["task"])
                prior = existing["expected_evaluations"].get(key)
                if prior is not None and prior != requirement:
                    raise RuntimeError(f"conflicting evaluation protocols for duplicate run ID {run['run_id']}")
                existing["expected_evaluations"][key] = requirement
            existing["mechanism_required"] = existing["mechanism_required"] or (
                matrix_name == "mechanism_500step" or run["run_id"].endswith("__gradlog")
            )
            if matrix_name == "mechanism_500step":
                model_config = _read_json(MODEL_PATHS[run["model"]] / "config.json")
                existing["mechanism_contract"] = {
                    "logging": dict(config["logging"]),
                    "target_modules": tuple(run["target_modules"]),
                    "gradient_accumulation_steps": int(run["training"]["gradient_accumulation_steps"]),
                    "num_hidden_layers": int(model_config["num_hidden_layers"]),
                    "hidden_size": int(model_config["hidden_size"]),
                    "num_attention_heads": int(model_config["num_attention_heads"]),
                    "num_key_value_heads": int(model_config.get("num_key_value_heads", model_config["num_attention_heads"])),
                    "lora_rank": int(run["training"]["lora_rank"]),
                }
            if matrix_name == "scope":
                architecture = resolve_projection_contract(
                    MODEL_PATHS[run["model"]], rank=int(run["training"]["lora_rank"])
                )
                architecture = bind_model_identity(architecture, run["model"])
                validate_scope_contract(architecture, config)
                contract = target_module_contract(architecture, list(run["target_modules"]))
                if existing["architecture_contract"] not in (None, contract):
                    raise RuntimeError(f"conflicting architecture contracts for duplicate run ID {run['run_id']}")
                existing["architecture_contract"] = contract
    rows = []
    for run_id in sorted(merged):
        row = merged[run_id]
        row["matrix_membership"] = tuple(sorted(row["matrix_membership"]))
        row["analysis_contexts"] = tuple(sorted(row["analysis_contexts"]))
        row["expected_checkpoints"] = tuple(sorted(int(step) for step in row["expected_checkpoints"]))
        row["expected_evaluations"] = tuple(value for _, value in sorted(row["expected_evaluations"].items()))
        row["expected_configurations"] = tuple(row["expected_configurations"][key] for key in sorted(row["expected_configurations"]))
        row["expected_dataset_split_hashes"] = tuple(sorted(row["expected_dataset_split_hashes"]))
        rows.append(row)
    return rows


def evaluation_filename(run_id: str, checkpoint: int, task: str) -> str:
    return f"{run_id}__step_{int(checkpoint):06d}__{task}.json"


def validate_evaluation_artifact(
    result_path: str | Path, requirement: dict, run_id: str, project_root: str | Path = ROOT,
) -> dict:
    """Validate evaluation identity and its immutable prediction JSONL hash."""
    result_path = Path(result_path)
    expected_name = evaluation_filename(run_id, requirement["checkpoint"], requirement["task"])
    if result_path.name != expected_name:
        raise RuntimeError(f"evaluation filename mismatch: expected {expected_name}")
    payload = _read_json(result_path)
    identity = (payload.get("run_id"), payload.get("checkpoint"), payload.get("task"))
    expected_identity = (run_id, int(requirement["checkpoint"]), requirement["task"])
    if identity != expected_identity:
        raise RuntimeError(f"evaluation identity mismatch in {result_path}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError(f"missing evaluation metrics in {result_path}")
    declared_metrics = set(requirement.get("metrics", ()))
    # Judgeable-sample counts are provenance companions of the ShareGPT score
    # (individual samples can be unjudgeable after the documented retry); they
    # extend, but never replace, the declared metric contract.
    companion_metrics = {"prometheus_judged_samples", "prometheus_unjudgeable_samples"}
    if not set(metrics).issubset(declared_metrics | companion_metrics) or not declared_metrics.issubset(metrics):
        raise RuntimeError(
            f"evaluation metrics do not match declared metric schema in {result_path}: "
            f"expected={sorted(declared_metrics)}, observed={sorted(metrics)}"
        )
    for name, value in metrics.items():
        if name in companion_metrics:
            continue
        _finite(value, f"evaluation metric {name}")
    prediction_name = payload.get("prediction_artifact")
    if not isinstance(prediction_name, str) or not prediction_name.strip():
        raise RuntimeError(f"missing prediction_artifact in {result_path}")
    prediction_path = Path(prediction_name)
    if not prediction_path.is_absolute():
        prediction_path = Path(project_root) / prediction_path
    prediction_path = prediction_path.resolve()
    root = Path(project_root).resolve()
    if not prediction_path.is_relative_to(root) or not prediction_path.is_file():
        raise RuntimeError(f"prediction_artifact is missing or outside project root: {prediction_name}")
    expected_hash = payload.get("prediction_sha256")
    if not isinstance(expected_hash, str) or file_sha256(prediction_path) != expected_hash:
        raise RuntimeError(f"prediction_sha256 mismatch for {result_path}")
    prediction_rows = _read_jsonl(prediction_path)
    sample_count = payload.get("sample_count")
    expected_count = requirement.get("protocol", {}).get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count != expected_count:
        raise RuntimeError(f"sample_count does not match the formal protocol in {result_path}")
    if len(prediction_rows) != sample_count:
        raise RuntimeError(f"sample_count does not match prediction artifact in {result_path}")
    if [row.get("sample_index") for row in prediction_rows] != list(range(sample_count)):
        raise RuntimeError(f"prediction artifact does not contain the exact ordered formal sample set in {result_path}")
    if requirement.get("bind_formal_samples"):
        expected_entries = formal_sample_entries(requirement["task"])
        observed_entries = tuple({
            "sample_id": row.get("sample_id"),
            "sample_input_sha256": row.get("sample_input_sha256"),
        } for row in prediction_rows)
        if observed_entries != expected_entries:
            raise RuntimeError(f"prediction artifact sample IDs/content do not match the frozen formal sample set in {result_path}")
    if payload.get("sample_set_hash") != requirement.get("sample_set_hash"):
        raise RuntimeError(f"sample_set_hash mismatch in {result_path}")
    evaluation_config = payload.get("evaluation_config")
    if evaluation_config != requirement.get("evaluation_config"):
        raise RuntimeError(f"evaluation_config does not match matrix protocol in {result_path}")
    if payload.get("evaluation_config_hash") != canonical_hash(evaluation_config):
        raise RuntimeError(f"evaluation_config_hash does not bind evaluation_config in {result_path}")
    extra_artifacts = []
    if requirement["task"] == "sharegpt":
        resolved = {}
        for label in ("candidate", "candidate_manifest", "judge"):
            artifact_name = payload.get(f"{label}_artifact")
            artifact = Path(artifact_name) if isinstance(artifact_name, str) else Path()
            if not artifact.is_absolute():
                artifact = root / artifact
            artifact = artifact.resolve()
            if (
                not isinstance(artifact_name, str) or not artifact.is_relative_to(root)
                or not artifact.is_file() or file_sha256(artifact) != payload.get(f"{label}_sha256")
            ):
                raise RuntimeError(f"ShareGPT {label}_artifact/hash mismatch in {result_path}")
            resolved[label] = artifact
            extra_artifacts.append(str(artifact))
        if resolved["judge"] != prediction_path:
            raise RuntimeError(f"ShareGPT judge artifact must be the immutable prediction artifact in {result_path}")
        from revision_experiments.scripts.sharegpt_judge import (
            validate_sharegpt_candidate_manifest, validate_sharegpt_raw_artifacts,
        )
        candidate_manifest, _, _ = validate_sharegpt_candidate_manifest(
            resolved["candidate"], run_id=run_id, checkpoint=int(requirement["checkpoint"]),
            project_root=root,
        )
        if not math.isclose(
            float(candidate_manifest["candidate_generation_seconds"]),
            float(payload.get("candidate_generation_seconds", float("nan"))),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise RuntimeError(f"ShareGPT candidate timing mismatch in {result_path}")
        judge_seconds = _finite(payload.get("judge_seconds"), "judge_seconds", minimum=0.0)
        expected_total = float(candidate_manifest["candidate_generation_seconds"]) + judge_seconds
        if not math.isclose(
            expected_total, float(payload.get("evaluation_seconds", float("nan"))),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise RuntimeError(f"ShareGPT evaluation_seconds does not include candidate plus judge time in {result_path}")
        recomputed_metrics, _ = validate_sharegpt_raw_artifacts(
            resolved["candidate"], resolved["judge"], run_id=run_id,
            checkpoint=int(requirement["checkpoint"]),
        )
        if set(recomputed_metrics) != set(metrics) or any(
            not math.isclose(recomputed_metrics[name], float(metrics[name]), rel_tol=1e-12, abs_tol=1e-12)
            for name in recomputed_metrics
        ):
            raise RuntimeError(f"ShareGPT metrics do not match raw candidate/judge artifacts in {result_path}")
    _finite(payload.get("evaluation_seconds"), "evaluation_seconds", minimum=0.0)
    payload["result_sha256"] = file_sha256(result_path)
    payload["resolved_prediction_artifact"] = str(prediction_path)
    payload["resolved_extra_artifacts"] = extra_artifacts
    return payload


def _tree_hash(root: Path) -> str:
    records = [
        {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    if not records:
        raise RuntimeError(f"checkpoint tree contains no files: {root}")
    return canonical_hash(records)


def _load_adapter_weight_tensors(path: Path) -> dict:
    """Parse adapter weights on CPU and reject malformed/non-tensor payloads."""
    try:
        import torch
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            tensors = load_file(str(path), device="cpu")
        else:
            tensors = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"checkpoint weight file is not loadable: {path}: {exc}") from exc
    if not isinstance(tensors, dict) or not tensors or not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise RuntimeError(f"checkpoint weight payload is not a nonempty tensor mapping: {path}")
    if not any("lora_A" in key for key in tensors) or not any("lora_B" in key for key in tensors):
        raise RuntimeError(f"checkpoint weights do not contain LoRA A/B tensors: {path}")
    if any(value.numel() <= 0 or not torch.isfinite(value).all().item() for value in tensors.values()):
        raise RuntimeError(f"checkpoint weights contain empty or non-finite tensors: {path}")
    return tensors


def _checkpoint_state(run_dir: Path, expected: tuple[int, ...]) -> tuple[list[int], list[int], dict[str, str]]:
    available = []
    hashes = {}
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_dir():
        for item in checkpoint_root.iterdir():
            match = re.fullmatch(r"step_(\d{6})", item.name)
            if item.is_dir() and match:
                weights = list(item.glob("adapter_model.safetensors")) + list(item.glob("adapter_model.bin"))
                config_path = item / "adapter_config.json"
                if config_path.is_file() and len(weights) == 1 and weights[0].stat().st_size > 0:
                    try:
                        adapter_config = _read_json(config_path)
                        if (
                            adapter_config.get("peft_type") != "LORA"
                            or isinstance(adapter_config.get("r"), bool)
                            or not isinstance(adapter_config.get("r"), int)
                            or adapter_config["r"] <= 0
                            or not adapter_config.get("target_modules")
                        ):
                            continue
                        _load_adapter_weight_tensors(weights[0])
                    except RuntimeError:
                        continue
                    if adapter_config:
                        step = int(match.group(1))
                        available.append(step)
                        hashes[str(step)] = _tree_hash(item)
    available = sorted(set(available))
    return available, sorted(set(expected) - set(available)), hashes


def _validate_mechanism_artifacts(run_dir: Path, max_steps: int, contract: dict | None = None) -> list[str]:
    errors = []
    manifest_path = run_dir / "gradients/manifest.json"
    diagnostics_path = run_dir / "gradient_diagnostics.jsonl"
    if not manifest_path.is_file():
        errors.append("missing file: gradients/manifest.json")
    else:
        try:
            manifest = _read_json(manifest_path)
            if int(manifest.get("max_steps", -1)) != int(max_steps) or not manifest.get("parameters"):
                errors.append("invalid gradients/manifest.json recording contract")
            logging = (contract or {}).get("logging", {})
            coordinate_count = int(logging.get("coordinates_per_matrix", 0))
            selected_layers = manifest.get("selected_layers")
            layer_count = int((contract or {}).get("num_hidden_layers", 0))
            expected_layers = [0, layer_count // 2, layer_count - 1] if layer_count > 0 else []
            if (
                manifest.get("schema_version") != 1
                or manifest.get("coordinate_seed") != logging.get("coordinate_seed")
                or not isinstance(selected_layers, list) or len(selected_layers) != 3
                or len(set(selected_layers)) != 3
                or selected_layers != sorted(selected_layers)
                or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in selected_layers)
            ):
                errors.append("invalid gradients/manifest.json layer/seed contract")
            if selected_layers != expected_layers:
                errors.append(
                    f"gradient manifest selected_layers must be configured first/middle/last {expected_layers}"
                )
            targets = tuple((contract or {}).get("target_modules", ()))
            expected_parameter_count = len(selected_layers or []) * len(targets) * 2
            parameters = manifest.get("parameters", [])
            if len(parameters) != expected_parameter_count:
                errors.append("gradient manifest does not cover every selected layer/target/adapter parameter")
            observed_keys = set()
            observed_key_order = []
            rng = np.random.default_rng(int(logging.get("coordinate_seed", -1)))
            hidden_size = int((contract or {}).get("hidden_size", 0))
            attention_heads = int((contract or {}).get("num_attention_heads", 0))
            key_value_heads = int((contract or {}).get("num_key_value_heads", 0))
            rank = int((contract or {}).get("lora_rank", 0))
            for parameter in manifest.get("parameters", []):
                name = parameter.get("file")
                if not isinstance(name, str) or Path(name).name != name or not (manifest_path.parent / name).is_file():
                    errors.append(f"missing gradient array declared by manifest: {name}")
                    continue
                parameter_name = parameter.get("name")
                match = re.search(r"layers\.(\d+)\..*?(%s).*?lora_([AB])" % "|".join(map(re.escape, targets)), str(parameter_name)) if targets else None
                if not match:
                    errors.append(f"gradient parameter is outside configured layer/target contract: {parameter_name}")
                    continue
                key = (int(match.group(1)), match.group(2), match.group(3))
                observed_keys.add(key)
                observed_key_order.append(key)
                indices = parameter.get("indices")
                output_size = hidden_size if match.group(2) == "q_proj" else (
                    hidden_size // attention_heads * key_value_heads if attention_heads > 0 else 0
                )
                parameter_numel = rank * (hidden_size if match.group(3) == "A" else output_size)
                selected_count = min(coordinate_count, parameter_numel)
                expected_indices = (
                    np.sort(rng.choice(parameter_numel, size=selected_count, replace=False)).astype(np.int64).tolist()
                    if parameter_numel > 0 and selected_count > 0 else []
                )
                if (
                    not isinstance(indices, list) or len(indices) != selected_count
                    or len(set(indices)) != selected_count or indices != sorted(indices)
                    or any(
                        isinstance(index, bool) or not isinstance(index, int)
                        or index < 0 or index >= parameter_numel for index in indices
                    )
                ):
                    errors.append(f"gradient coordinate-count/index contract failed: {parameter_name}")
                    continue
                if indices != expected_indices:
                    errors.append(f"gradient coordinates are not the configured seed-selected indices: {parameter_name}")
                try:
                    values = np.load(manifest_path.parent / name, mmap_mode="r", allow_pickle=False)
                    if values.shape != (int(max_steps) + 1, selected_count) or not np.isfinite(values).all():
                        errors.append(f"gradient array shape/finite contract failed: {name}")
                except (OSError, ValueError) as exc:
                    errors.append(f"invalid gradient array {name}: {exc}")
            expected_keys = {
                (int(layer), target, adapter)
                for layer in (selected_layers or []) for target in targets for adapter in ("A", "B")
            }
            if observed_keys != expected_keys:
                errors.append("gradient manifest has incomplete or duplicate layer/target/adapter coverage")
            expected_key_order = [
                (layer, target, adapter)
                for layer in expected_layers for target in targets for adapter in ("A", "B")
            ]
            if observed_key_order != expected_key_order:
                errors.append("gradient manifest parameter order does not match deterministic model traversal")
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if not diagnostics_path.is_file():
        errors.append("missing file: gradient_diagnostics.jsonl")
    else:
        try:
            diagnostics = _read_jsonl(diagnostics_path)
            logging = (contract or {}).get("logging", {})
            steps = {int(step) for step in logging.get("full_matrix_steps", [])}
            targets = tuple((contract or {}).get("target_modules", ()))
            accumulation = int((contract or {}).get("gradient_accumulation_steps", 0))
            manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
            layers = {int(layer) for layer in manifest.get("selected_layers", [])}
            expected_effective = {(step, layer, target) for step in steps for layer in layers for target in targets}
            expected_lora = {(step, item.get("name")) for step in steps for item in manifest.get("parameters", [])}
            observed_effective, observed_lora = set(), set()
            for item in diagnostics:
                kind, step, parameter = item.get("kind"), item.get("step"), item.get("parameter")
                if kind == "effective_base_weight_subspace":
                    match = re.search(r"layers\.(\d+)\..*?(%s).*?base_layer\.weight$" % "|".join(map(re.escape, targets)), str(parameter)) if targets else None
                    if not match or step not in steps:
                        errors.append(f"unexpected effective diagnostic row: step={step}, parameter={parameter}")
                        continue
                    observed_effective.add((int(step), int(match.group(1)), match.group(2)))
                    for field in ("gradient_frobenius_norm", "gradient_times_a_transpose_norm"):
                        _finite(item.get(field), field, minimum=0.0)
                    capture = _finite(item.get("gradient_capture_ratio"), "gradient_capture_ratio")
                    if not 0.0 <= capture <= 1.0:
                        errors.append("gradient_capture_ratio must be in [0,1]")
                    if item.get("microbatch_backward_calls") != accumulation:
                        errors.append("effective diagnostic microbatch count mismatch")
                    singular = item.get("a_singular_values")
                    angles = item.get("principal_angles_degrees")
                    if not isinstance(singular, list) or not singular or any(_finite(x, "a_singular_value", minimum=0.0) < 0 for x in singular):
                        errors.append("invalid a_singular_values")
                    if not isinstance(angles, list) or not angles or any(not 0 <= _finite(x, "principal_angle") <= 90 for x in angles):
                        errors.append("invalid principal_angles_degrees")
                    rank = item.get("a_effective_rank")
                    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank > len(singular or []):
                        errors.append("invalid a_effective_rank")
                elif kind == "lora_parameter_gradient":
                    observed_lora.add((int(step), parameter))
                    _finite(item.get("gradient_frobenius_norm"), "gradient_frobenius_norm", minimum=0.0)
                    _finite(item.get("parameter_frobenius_norm"), "parameter_frobenius_norm", minimum=0.0)
                else:
                    errors.append(f"undeclared gradient diagnostic kind: {kind}")
            if observed_effective != expected_effective:
                errors.append("gradient diagnostics do not have exact snapshot/layer/target coverage")
            if observed_lora != expected_lora:
                errors.append("LoRA gradient diagnostics do not have exact snapshot/parameter coverage")
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def _validate_initialization_audit(init: dict, config: dict | None) -> list[str]:
    """Require Task-3 layer statistics and a fixed-batch initial-B gradient audit."""

    errors = []
    if init.get("matrix_statistics_schema_version") != 1:
        errors.append("initialization matrix statistics schema is missing or unsupported")
    rows = init.get("matrix_statistics")
    if not isinstance(rows, list) or not rows:
        return errors + ["initialization matrix statistics must be a non-empty list"]
    parameters = [row.get("parameter") for row in rows if isinstance(row, dict)]
    if len(parameters) != len(rows) or len(set(parameters)) != len(parameters):
        errors.append("initialization matrix statistics have missing or duplicate parameter names")
        return errors
    a_rows = {row["parameter"]: row for row in rows if row.get("factor") == "A"}
    b_rows = {row["parameter"]: row for row in rows if row.get("factor") == "B"}
    normalize = lambda name: name.replace("lora_A", "lora_FACTOR").replace("lora_B", "lora_FACTOR")
    if not a_rows or {normalize(name) for name in a_rows} != {normalize(name) for name in b_rows}:
        errors.append("initialization matrix statistics do not contain exact paired LoRA A/B coverage")
    required_numeric = ("mean", "std", "variance", "frobenius_norm", "spectral_norm", "max_abs")
    for row in rows:
        try:
            shape = row.get("shape")
            if not isinstance(shape, list) or len(shape) != 2 or any(int(value) <= 0 for value in shape):
                raise RuntimeError("shape must contain two positive dimensions")
            if int(row.get("numel")) != int(shape[0]) * int(shape[1]):
                raise RuntimeError("numel does not match shape")
            nonzero = int(row.get("nonzero_count"))
            if nonzero < 0 or nonzero > int(row["numel"]):
                raise RuntimeError("nonzero_count is outside tensor bounds")
            for key in required_numeric:
                _finite(row.get(key), f"initialization {row['parameter']} {key}", minimum=0.0 if key != "mean" else None)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"invalid initialization matrix statistics row: {exc}")
    effective_method = config.get("effective_method") if isinstance(config, dict) else None
    if effective_method not in {"pissa", "lora_one"}:
        for row in b_rows.values():
            if row.get("nonzero_count") != 0 or row.get("frobenius_norm") != 0.0 or row.get("max_abs") != 0.0:
                errors.append(f"zero-B initialization contract failed for {row['parameter']}")
    audit = init.get("initial_gradient_audit")
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        return errors + ["fixed-batch initial-gradient audit is missing or unsupported"]
    expected_microbatches = None
    if isinstance(config, dict) and isinstance(config.get("training"), dict):
        expected_microbatches = config["training"].get("gradient_accumulation_steps")
    if expected_microbatches is None or audit.get("microbatch_count") != int(expected_microbatches):
        errors.append("initial-gradient microbatch count does not match the run configuration")
    digest = audit.get("ordered_batch_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("initial-gradient audit does not bind the ordered fixed batch")
    gradient_rows = audit.get("lora_b")
    if not isinstance(gradient_rows, list):
        errors.append("initial-gradient audit has no LoRA-B rows")
    else:
        gradient_names = [row.get("parameter") for row in gradient_rows if isinstance(row, dict)]
        if len(gradient_names) != len(gradient_rows) or set(gradient_names) != set(b_rows):
            errors.append("initial-gradient audit does not exactly cover every LoRA-B parameter")
        for row in gradient_rows:
            try:
                _finite(row.get("gradient_frobenius_norm"), "initial LoRA-B gradient norm", minimum=0.0)
                _finite(row.get("gradient_max_abs"), "initial LoRA-B gradient maximum", minimum=0.0)
                if int(row.get("gradient_nonzero_count")) < 0:
                    raise RuntimeError("negative gradient_nonzero_count")
            except (RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"invalid initial LoRA-B gradient row: {exc}")
    try:
        _finite(init.get("initialization_audit_seconds"), "initialization_audit_seconds", minimum=0.0)
        _finite(audit.get("audit_seconds"), "initial_gradient_audit.audit_seconds", minimum=0.0)
    except RuntimeError as exc:
        errors.append(str(exc))
    return errors


def _empty_inventory_row(expected: dict) -> dict:
    evaluation_names = [
        evaluation_filename(expected["run_id"], item["checkpoint"], item["task"])
        for item in expected["expected_evaluations"]
    ]
    return {
        "run_id": expected["run_id"], "model": expected["model"], "task": expected["task"],
        "method": expected["method"], "seed": expected["seed"], "max_steps": expected["max_steps"],
        "target": expected["target"], "matrix_membership": json.dumps(expected["matrix_membership"]),
        "matrix_membership_count": len(expected["matrix_membership"]),
        "analysis_contexts": json.dumps(expected.get("analysis_contexts", expected["matrix_membership"])),
        "status": "missing",
        "terminal_state": "missing", "metadata_sha256": None, "config_hash": None, "git_commit": None,
        "expected_checkpoints": json.dumps(expected["expected_checkpoints"]),
        "expected_config_sha256": json.dumps(sorted(canonical_hash(item) for item in expected.get("expected_configurations", ()))),
        "available_checkpoints": json.dumps([]), "missing_checkpoints": json.dumps(expected["expected_checkpoints"]),
        "checkpoint_sha256": json.dumps({}),
        "expected_evaluations": json.dumps(evaluation_names),
        "expected_evaluation_protocols": json.dumps(expected["expected_evaluations"], sort_keys=True),
        "available_evaluations": json.dumps([]),
        "missing_evaluations": json.dumps(evaluation_names), "invalid_evaluations": json.dumps([]),
        "raw_auc500_reported": None,
        "raw_auc500_recomputed": None, "training_seconds_recomputed": None,
        "initialization_seconds_total": None, "failure_reason": "run directory missing",
}


def _evaluation_state(expected: dict, evaluations_root: Path, project_root: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    available, missing, invalid, errors = [], [], [], []
    for requirement in expected["expected_evaluations"]:
        name = evaluation_filename(expected["run_id"], requirement["checkpoint"], requirement["task"])
        path = evaluations_root / name
        if not path.is_file():
            missing.append(name)
            continue
        try:
            validate_evaluation_artifact(path, requirement, expected["run_id"], project_root)
            available.append(name)
        except RuntimeError as exc:
            invalid.append(name)
            errors.append(str(exc))
    return available, missing, invalid, errors


def _inspect_expected_run(expected: dict, run_dir: Path, evaluations_root: Path, project_root: Path) -> dict:
    row = _empty_inventory_row(expected)
    available_evaluations, missing_evaluations, invalid_evaluations, evaluation_errors = _evaluation_state(
        expected, evaluations_root, project_root
    )
    row["available_evaluations"] = json.dumps(available_evaluations)
    row["missing_evaluations"] = json.dumps(missing_evaluations)
    row["invalid_evaluations"] = json.dumps(invalid_evaluations)
    if not run_dir.is_dir():
        if invalid_evaluations:
            row["failure_reason"] += " | invalid orphan evaluations: " + " | ".join(evaluation_errors)
        return row
    completed = (run_dir / "COMPLETED").is_file()
    failed = (run_dir / "FAILED.json").is_file()
    row["terminal_state"] = "conflict" if completed and failed else "complete" if completed else "failed" if failed else "none"
    if (run_dir / "metadata.json").is_file():
        row["metadata_sha256"] = file_sha256(run_dir / "metadata.json")
    available_checkpoints, missing_checkpoints, checkpoint_hashes = _checkpoint_state(run_dir, expected["expected_checkpoints"])
    row["available_checkpoints"] = json.dumps(available_checkpoints)
    row["missing_checkpoints"] = json.dumps(missing_checkpoints)
    row["checkpoint_sha256"] = json.dumps(checkpoint_hashes, sort_keys=True)
    if failed and not completed:
        try:
            failure = _read_json(run_dir / "FAILED.json")
            row["failure_reason"] = str(failure.get("error") or failure.get("traceback") or "FAILED.json present")
        except RuntimeError as exc:
            row["failure_reason"] = str(exc)
        row["status"] = "failed"
        return row
    if not completed:
        row["status"] = "incomplete"
        row["failure_reason"] = "run has no terminal marker"
        return row

    try:
        errors = list(validate_run_directory(run_dir))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        errors = [f"run schema validation failed: {exc}"]
    config = None
    try:
        metadata = _read_json(run_dir / "metadata.json")
        row["config_hash"] = metadata.get("config_hash")
        row["git_commit"] = metadata.get("git_commit")
        for key in ("run_id", "method", "seed", "max_steps"):
            if metadata.get(key) != expected[key]:
                errors.append(f"metadata identity mismatch: {key}")
        if metadata.get("dataset_split_hash") not in expected.get("expected_dataset_split_hashes", ()):
            errors.append("metadata dataset_split_hash is not allowed by the expected matrix context")
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or metadata.get("config_hash") != canonical_hash(config):
            errors.append("metadata config_hash does not bind config.yaml")
        if isinstance(config, dict):
            allowed = tuple(expected.get("expected_configurations", ()))
            fingerprints = {canonical_hash(item) for item in allowed}
            if canonical_hash(config) not in fingerprints:
                errors.append("config matrix/payload does not match an allowed expected configuration fingerprint")
            else:
                accepted = next(item for item in allowed if canonical_hash(item) == canonical_hash(config))
                metadata_expected = {
                    "run_id": accepted["run_id"], "method": accepted["method"],
                    "seed": accepted["seed"], "max_steps": accepted["max_steps"],
                    "target_modules": accepted["target_modules"],
                    "effective_method": accepted["effective_method"],
                    "effective_learning_rate": accepted["effective_learning_rate"],
                    "validation_selection": accepted.get("validation_selection"),
                    "model_loader_provenance": loader_provenance(expected["model"]),
                }
                for key, value in metadata_expected.items():
                    if metadata.get(key) != value:
                        errors.append(f"metadata protocol mismatch: {key}")
                if expected.get("architecture_contract") is not None:
                    if metadata.get("architecture_provenance") != expected["architecture_contract"]:
                        errors.append("metadata architecture_provenance does not match the frozen scope contract")
    except (RuntimeError, OSError, yaml.YAMLError) as exc:
        errors.append(str(exc))
    try:
        summary = _read_json(run_dir / "summary.json")
        raw_rows = _read_jsonl(run_dir / "raw_loss.jsonl")
        timing_rows = _read_jsonl(run_dir / "timing.jsonl")
        expected_steps = list(range(1, int(expected["max_steps"]) + 1))
        if [item.get("step") for item in raw_rows] != expected_steps:
            errors.append(f"raw_loss.jsonl must contain contiguous steps 1..{expected['max_steps']}")
        if [item.get("step") for item in timing_rows] != expected_steps:
            errors.append(f"timing.jsonl must contain contiguous steps 1..{expected['max_steps']}")
        losses = [_finite(item.get("train_loss"), "train_loss", minimum=0.0) for item in raw_rows]
        times = [_finite(item.get("step_time_seconds"), "step_time_seconds", minimum=0.0) for item in timing_rows]
        if summary.get("steps_logged") != int(expected["max_steps"]):
            errors.append("summary.steps_logged does not match expected max_steps")
        row["training_seconds_recomputed"] = float(math.fsum(times))
        if expected["max_steps"] >= 500:
            computed_auc = raw_trapezoid_auc(losses, 0, 500)
            reported_auc = _finite(summary.get("raw_auc500"), "summary.raw_auc500", minimum=0.0)
            row["raw_auc500_reported"] = reported_auc
            row["raw_auc500_recomputed"] = computed_auc
            if not math.isclose(reported_auc, computed_auc, rel_tol=1e-10, abs_tol=1e-8):
                errors.append("summary.raw_auc500 does not match raw_loss.jsonl")
    except RuntimeError as exc:
        errors.append(str(exc))
    except (ValueError, TypeError) as exc:
        errors.append(f"invalid raw AUC input: {exc}")
    try:
        init = _read_json(run_dir / "initialization_stats.json")
        adapter = _finite(init.get("adapter_initialization_seconds"), "adapter_initialization_seconds", minimum=0.0)
        gradient = _finite(init.get("gradient_estimation_seconds", 0.0), "gradient_estimation_seconds", minimum=0.0)
        total = _finite(init.get("initialization_seconds_total"), "initialization_seconds_total", minimum=0.0)
        if not math.isclose(total, adapter + gradient, rel_tol=1e-8, abs_tol=1e-8):
            errors.append("initialization timing components do not sum")
        row["initialization_seconds_total"] = total
        errors.extend(_validate_initialization_audit(init, config if isinstance(config, dict) else None))
    except RuntimeError as exc:
        errors.append(str(exc))
    if missing_checkpoints:
        errors.append(f"missing checkpoints: {missing_checkpoints}")
    errors.extend(evaluation_errors)
    if missing_evaluations:
        errors.append(f"missing evaluations: {missing_evaluations}")
    if expected["mechanism_required"]:
        errors.extend(_validate_mechanism_artifacts(run_dir, expected["max_steps"], expected.get("mechanism_contract")))
    row["status"] = "complete" if not errors else "invalid"
    row["failure_reason"] = " | ".join(dict.fromkeys(errors))
    return row


def _inspect_untracked_run(run_dir: Path) -> dict:
    metadata = {}
    if (run_dir / "metadata.json").is_file():
        try:
            metadata = _read_json(run_dir / "metadata.json")
        except RuntimeError:
            pass
    return {
        "run_id": run_dir.name, "model": metadata.get("model"), "task": metadata.get("task"),
        "method": metadata.get("method"), "seed": metadata.get("seed"), "max_steps": metadata.get("max_steps"),
        "target": None, "matrix_membership": json.dumps([]), "matrix_membership_count": 0,
        "analysis_contexts": json.dumps([]),
        "status": "untracked", "terminal_state": "complete" if (run_dir / "COMPLETED").is_file() else "unknown",
        "metadata_sha256": file_sha256(run_dir / "metadata.json") if (run_dir / "metadata.json").is_file() else None,
        "config_hash": metadata.get("config_hash"), "git_commit": metadata.get("git_commit"),
        "expected_checkpoints": json.dumps([]), "expected_config_sha256": json.dumps([]),
        "available_checkpoints": json.dumps([]),
        "missing_checkpoints": json.dumps([]), "checkpoint_sha256": json.dumps({}),
        "expected_evaluations": json.dumps([]), "expected_evaluation_protocols": json.dumps([]),
        "available_evaluations": json.dumps([]),
        "missing_evaluations": json.dumps([]), "invalid_evaluations": json.dumps([]),
        "raw_auc500_reported": None, "raw_auc500_recomputed": None,
        "training_seconds_recomputed": None, "initialization_seconds_total": None,
        "failure_reason": "run directory is not declared by any current matrix",
    }


def build_run_inventory(
    expected_runs: Iterable[dict], runs_root: str | Path, evaluations_root: str | Path,
    *, project_root: str | Path = ROOT,
) -> pd.DataFrame:
    expected_runs = list(expected_runs)
    runs_root, evaluations_root = Path(runs_root), Path(evaluations_root)
    expected_ids = {item["run_id"] for item in expected_runs}
    rows = [
        _inspect_expected_run(item, runs_root / item["run_id"], evaluations_root, Path(project_root))
        for item in expected_runs
    ]
    if runs_root.is_dir():
        rows.extend(
            _inspect_untracked_run(path)
            for path in sorted(runs_root.iterdir()) if path.is_dir() and path.name not in expected_ids
        )
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS).sort_values("run_id", kind="stable").reset_index(drop=True)


def aggregate_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if "analysis_context" not in seed_metrics:
        seed_metrics = seed_metrics.assign(analysis_context="unspecified")
    required = set(GROUP_KEYS + ["seed", "value", "run_id", "metadata_sha256"])
    missing = sorted(required - set(seed_metrics.columns))
    if missing:
        raise RuntimeError(f"seed metric table missing columns: {missing}")
    duplicate_keys = ["analysis_context", "model", "task", "method", "seed", "endpoint", "metric"]
    if seed_metrics.duplicated(duplicate_keys).any():
        raise RuntimeError("seed metric table contains duplicate run/seed observations")
    rows = []
    for keys, group in seed_metrics.groupby(GROUP_KEYS, dropna=False, sort=True):
        values = pd.to_numeric(group["value"], errors="raise")
        if not np.isfinite(values).all():
            raise RuntimeError("seed metric table contains non-finite values")
        row = dict(zip(GROUP_KEYS, keys))
        row.update({
            "n_runs": int(group["seed"].nunique()), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
            "seeds": json.dumps(sorted(int(seed) for seed in group["seed"])),
            "source_run_ids": json.dumps(sorted(group["run_id"].astype(str).unique())),
            "source_metadata_sha256": json.dumps(sorted(group["metadata_sha256"].astype(str).unique())),
        })
        if row["n_runs"] != len(group):
            raise RuntimeError("independent n must equal distinct run/seed observations")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_differences(
    seed_metrics: pd.DataFrame, references: tuple[str, ...] = ("peft_default", "iid_matched"),
) -> pd.DataFrame:
    if "analysis_context" not in seed_metrics:
        seed_metrics = seed_metrics.assign(analysis_context="unspecified")
    required = set(PAIR_KEYS + ["method", "value", "metric_direction", "run_id", "metadata_sha256"])
    missing = sorted(required - set(seed_metrics.columns))
    if missing:
        raise RuntimeError(f"seed metric table missing columns: {missing}")
    identity = PAIR_KEYS + ["method"]
    if seed_metrics.duplicated(identity).any():
        raise RuntimeError("paired comparison input contains duplicate exact join keys")
    output = []
    grouping = ["analysis_context", "model", "task", "endpoint", "metric"]
    for group_keys, group in seed_metrics.groupby(grouping, dropna=False, sort=True):
        for reference in references:
            reference_rows = group[group["method"] == reference].set_index("seed")
            if reference_rows.empty:
                continue
            for method in sorted(set(group["method"]) - {reference}):
                method_rows = group[group["method"] == method].set_index("seed")
                if set(method_rows.index) != set(reference_rows.index):
                    raise RuntimeError(
                        f"exact paired seed join failed for {group_keys}, {method} versus {reference}: "
                        f"method={sorted(method_rows.index)}, reference={sorted(reference_rows.index)}"
                    )
                n_runs = len(method_rows)
                for seed in sorted(method_rows.index):
                    candidate, baseline = method_rows.loc[seed], reference_rows.loc[seed]
                    if candidate["metric_direction"] != baseline["metric_direction"]:
                        raise RuntimeError("paired metric directions disagree")
                    difference = float(candidate["value"]) - float(baseline["value"])
                    direction = candidate["metric_direction"]
                    if direction not in {"higher", "lower"}:
                        raise RuntimeError(f"unknown metric direction: {direction}")
                    output.append({
                        "analysis_context": group_keys[0], "model": group_keys[1], "task": group_keys[2],
                        "endpoint": group_keys[3], "metric": group_keys[4],
                        "metric_direction": direction, "method": method,
                        "reference_method": reference, "seed": int(seed), "n_runs": n_runs,
                        "value": float(candidate["value"]), "reference_value": float(baseline["value"]),
                        "difference": difference,
                        "favorable_difference": difference if direction == "higher" else -difference,
                        "run_id": candidate["run_id"], "reference_run_id": baseline["run_id"],
                        "metadata_sha256": candidate["metadata_sha256"],
                        "reference_metadata_sha256": baseline["metadata_sha256"],
                    })
    return pd.DataFrame(output)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _matrix_hashes(config_dir: Path) -> dict[str, str]:
    return {path.name: file_sha256(path) for path in sorted(config_dir.glob("*_matrix.yaml"))}


def validate_baseline_search_artifacts(
    matrix: dict, runs_root: str | Path, csv_path: str | Path, selected_path: str | Path,
) -> tuple[list[dict], dict]:
    """Rebuild all 48 trials and require byte-equivalent CSV plus verified selection."""
    rows, _ = collect_screening_results(matrix, runs_root)
    selected = verify_selection_manifest(matrix, runs_root, selected_path)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    try:
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            observed = handle.read()
    except OSError as exc:
        raise RuntimeError(f"cannot read baseline all-trials CSV: {exc}") from exc
    if observed != buffer.getvalue():
        raise RuntimeError("baseline all-trials CSV does not exactly match recomputed screening results")
    return rows, selected


def _publication_prerequisites(
    inventory: pd.DataFrame, audits_root: Path, output_root: Path, config_dir: Path, runs_root: Path,
) -> list[str]:
    blockers = [
        f"{row.run_id}: {row.status}: {row.failure_reason}"
        for row in inventory.itertuples() if row.status != "complete"
    ]
    required = [
        audits_root / "initialization_statistics.csv", audits_root / "step_zero_audit.json",
        audits_root / "step_zero_batch_losses.csv", output_root / "baseline_search_all_trials.csv",
        output_root / "baseline_selected_configs.yaml",
    ]
    blockers.extend(f"missing publication prerequisite: {path}" for path in required if not path.is_file())
    if not any("baseline_search" in str(path) and not path.is_file() for path in required):
        try:
            matrix = load_matrix(config_dir / "baseline_search_matrix.yaml")
            validate_baseline_search_artifacts(
                matrix, runs_root, output_root / "baseline_search_all_trials.csv",
                output_root / "baseline_selected_configs.yaml",
            )
        except (RuntimeError, OSError, ValueError) as exc:
            blockers.append(f"invalid baseline search publication source: {exc}")
    return blockers


def _metric_direction(metric: str) -> str:
    return "lower" if metric in {"raw_auc500", "final_train_loss", "heldout_nll", "validation_loss"} else "higher"


def _collect_metric_rows(
    expected_runs: list[dict], inventory: pd.DataFrame, runs_root: Path,
    evaluations_root: Path, project_root: Path,
) -> pd.DataFrame:
    inventory_by_id = inventory.set_index("run_id")
    rows = []
    for expected in expected_runs:
        contexts = [
            context for context in expected.get("analysis_contexts", expected["matrix_membership"])
            if context != "baseline_fairness_screening"
        ]
        if not contexts:
            continue
        source = inventory_by_id.loc[expected["run_id"]]
        run_dir = runs_root / expected["run_id"]
        metadata = _read_json(run_dir / "metadata.json")
        raw_rows = _read_jsonl(run_dir / "raw_loss.jsonl")
        timing_rows = _read_jsonl(run_dir / "timing.jsonl")
        cumulative = np.cumsum([_finite(item["step_time_seconds"], "step_time_seconds", minimum=0.0) for item in timing_rows])
        init_seconds = float(source["initialization_seconds_total"])
        common = {
            "model": expected["model"], "task": expected["task"], "method": expected["method"],
            "seed": expected["seed"], "run_id": expected["run_id"],
            "metadata_sha256": source["metadata_sha256"], "config_hash": metadata["config_hash"],
            "matrix_membership": source["matrix_membership"],
        }
        observations = [{
            "endpoint": "step_000500", "metric": "raw_auc500", "metric_direction": "lower",
            "unit": "loss_step", "value": float(source["raw_auc500_recomputed"]),
            "training_seconds": float(cumulative[499]), "initialization_seconds": init_seconds,
            "training_plus_initialization_seconds": float(cumulative[499]) + init_seconds,
            "evaluation_seconds": math.nan, "evaluation_result_sha256": None, "prediction_sha256": None,
        }, {
            "endpoint": f"step_{expected['max_steps']:06d}", "metric": "final_train_loss", "metric_direction": "lower",
            "unit": "loss", "value": _finite(raw_rows[-1]["train_loss"], "final train loss", minimum=0.0),
            "training_seconds": float(cumulative[-1]), "initialization_seconds": init_seconds,
            "training_plus_initialization_seconds": float(cumulative[-1]) + init_seconds,
            "evaluation_seconds": math.nan, "evaluation_result_sha256": None, "prediction_sha256": None,
        }]
        for requirement in expected["expected_evaluations"]:
            path = evaluations_root / evaluation_filename(expected["run_id"], requirement["checkpoint"], requirement["task"])
            result = validate_evaluation_artifact(path, requirement, expected["run_id"], project_root)
            checkpoint = int(requirement["checkpoint"])
            checkpoint_seconds = float(cumulative[checkpoint - 1]) if checkpoint else 0.0
            for metric, value in result["metrics"].items():
                observations.append({
                    "endpoint": f"step_{checkpoint:06d}", "metric": metric,
                    "metric_direction": _metric_direction(metric), "unit": "score", "value": float(value),
                    "training_seconds": checkpoint_seconds, "initialization_seconds": init_seconds,
                    "training_plus_initialization_seconds": checkpoint_seconds + init_seconds,
                    "evaluation_seconds": float(result["evaluation_seconds"]),
                    "evaluation_result_sha256": result["result_sha256"],
                    "prediction_sha256": result["prediction_sha256"],
                })
        for context in contexts:
            rows.extend(common | {"analysis_context": context} | observation for observation in observations)
    frame = pd.DataFrame(rows)
    identity = ["analysis_context", "model", "task", "method", "seed", "endpoint", "metric"]
    duplicates = frame[frame.duplicated(identity, keep=False)]
    if not duplicates.empty:
        grouped = duplicates.groupby(identity)["run_id"].nunique()
        if (grouped > 1).any():
            raise RuntimeError("multiple source runs claim one seed-level metric")
        frame = frame.drop_duplicates(identity, keep="first")
    return frame.sort_values(identity, kind="stable").reset_index(drop=True)


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "analysis_context", "model", "task", "endpoint", "metric", "metric_direction", "method",
        "reference_method",
    ]
    rows = []
    for values, group in paired.groupby(keys, dropna=False, sort=True):
        rows.append(dict(zip(keys, values)) | {
            "n_runs": len(group), "mean_difference": float(group["difference"].mean()),
            "sd_difference": float(group["difference"].std(ddof=1)) if len(group) > 1 else math.nan,
            "mean_favorable_difference": float(group["favorable_difference"].mean()),
            "seeds": json.dumps(sorted(int(seed) for seed in group["seed"])),
            "source_run_ids": json.dumps(sorted(group["run_id"].unique())),
            "reference_run_ids": json.dumps(sorted(group["reference_run_id"].unique())),
        })
    return pd.DataFrame(rows)


def _time_to_equivalent(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    scores = seed_metrics[
        (seed_metrics["metric_direction"] == "higher") &
        (~seed_metrics["metric"].isin(["raw_auc500", "final_train_loss"]))
    ]
    rows = []
    for keys, group in scores.groupby(["analysis_context", "model", "task", "seed", "metric"], sort=True):
        baseline = group[(group["method"] == "peft_default") & (group["endpoint"] == "step_002500")]
        if len(baseline) != 1:
            raise RuntimeError(f"time-to-equivalent requires one PEFT-default step-2500 target for {keys}")
        target_row = baseline.iloc[0]
        target = float(target_row["value"])
        for method, candidates in group.groupby("method", sort=True):
            ordered = candidates.assign(step=candidates["endpoint"].str.removeprefix("step_").astype(int)).sort_values("step")
            reached = ordered[ordered["value"] >= target]
            selected = reached.iloc[0] if not reached.empty else None
            final = ordered.iloc[-1]
            evidence = selected if selected is not None else final
            rows.append({
                "analysis_context": keys[0], "model": keys[1], "task": keys[2],
                "seed": int(keys[3]), "metric": keys[4],
                "metric_direction": "higher", "method": method, "target_method": "peft_default",
                "target_endpoint": "step_002500", "target_value": target, "reached": selected is not None,
                "target_run_id": target_row["run_id"],
                "target_metadata_sha256": target_row["metadata_sha256"],
                "candidate_run_ids": json.dumps(ordered["run_id"].astype(str).tolist()),
                "candidate_metadata_sha256": json.dumps(ordered["metadata_sha256"].astype(str).tolist()),
                "candidate_observations": json.dumps([
                    {"endpoint": item["endpoint"], "run_id": item["run_id"],
                     "metadata_sha256": item["metadata_sha256"]}
                    for item in ordered.to_dict("records")
                ]),
                "earliest_endpoint": selected["endpoint"] if selected is not None else None,
                "earliest_value": float(selected["value"]) if selected is not None else math.nan,
                "training_seconds": float(selected["training_seconds"]) if selected is not None else math.nan,
                "initialization_seconds": float(selected["initialization_seconds"]) if selected is not None else math.nan,
                "training_plus_initialization_seconds": float(selected["training_plus_initialization_seconds"]) if selected is not None else math.nan,
                "evaluation_seconds_separate": float(selected["evaluation_seconds"]) if selected is not None else math.nan,
                "run_id": evidence["run_id"], "metadata_sha256": evidence["metadata_sha256"],
                "final_observed_endpoint": final["endpoint"], "final_observed_value": float(final["value"]),
                "final_observed_training_seconds": float(final["training_seconds"]),
                "final_observed_initialization_seconds": float(final["initialization_seconds"]),
                "final_observed_training_plus_initialization_seconds": float(final["training_plus_initialization_seconds"]),
                "final_observed_evaluation_seconds": float(final["evaluation_seconds"]),
                "final_observed_run_id": final["run_id"],
                "final_observed_metadata_sha256": final["metadata_sha256"],
            })
    return pd.DataFrame(rows)


def _audit_tables(audits_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    initialization_path = audits_root / "initialization_statistics.csv"
    initialization = pd.read_csv(initialization_path)
    if initialization.empty:
        raise RuntimeError("initialization statistics are empty")
    initialization["source_artifact"] = str(initialization_path)
    initialization["source_sha256"] = file_sha256(initialization_path)
    audit_path = audits_root / "step_zero_audit.json"
    audit = _read_json(audit_path)
    if audit.get("schema_version") != 4 or audit.get("gate_passed") is not True:
        raise RuntimeError("canonical step-zero audit is incomplete or failed")
    if audit.get("required_cases") != ["openpangu__gsm8k", "qwen__cmmlu"]:
        raise RuntimeError("canonical step-zero audit required-case contract mismatch")
    rows = []
    for case in audit.get("cases", []):
        for method, values in case.get("equivalence", {}).items():
            rows.append({
                "model": case.get("model"), "task": case.get("task"), "seed": case.get("seed"),
                "method": method, **values, "tolerance": case.get("tolerance"),
                "equivalence_passed": case.get("equivalence_passed"),
                "source_artifact": str(audit_path), "source_sha256": file_sha256(audit_path),
            })
    if not rows:
        raise RuntimeError("step-zero audit contains no method equivalence rows")
    return initialization, pd.DataFrame(rows)


def _mechanism_table(expected_runs: list[dict], runs_root: Path) -> pd.DataFrame:
    rows = []
    for expected in expected_runs:
        if not expected["mechanism_required"]:
            continue
        run_dir = runs_root / expected["run_id"]
        metadata_sha256 = file_sha256(run_dir / "metadata.json")
        diagnostics_path = run_dir / "gradient_diagnostics.jsonl"
        diagnostics = pd.DataFrame(_read_jsonl(diagnostics_path))
        effective = diagnostics[diagnostics["kind"] == "effective_base_weight_subspace"]
        if effective.empty:
            raise RuntimeError(f"no effective gradient diagnostics for {expected['run_id']}")
        for step, group in effective.groupby("step", sort=True):
            rows.append({
                "model": expected["model"], "task": expected["task"], "method": expected["method"],
                "seed": expected["seed"], "run_id": expected["run_id"], "step": int(step),
                "metadata_sha256": metadata_sha256,
                "gradient_capture_ratio": float(group["gradient_capture_ratio"].mean()),
                "gradient_frobenius_norm": float(group["gradient_frobenius_norm"].mean()),
                "nested_parameter_count": len(group), "diagnostics_sha256": file_sha256(diagnostics_path),
            })
        manifest_path = run_dir / "gradients/manifest.json"
        manifest = _read_json(manifest_path)
        alpha_values, coordinate_count = [], 0
        for parameter in manifest["parameters"]:
            values = np.load(manifest_path.parent / parameter["file"], mmap_mode="r")
            selected = np.asarray(values[:501], dtype=np.float64)
            if selected.shape[0] != 501 or not np.isfinite(selected).all():
                raise RuntimeError(f"incomplete gradient coordinates for {expected['run_id']}")
            coordinate_count += selected.shape[1]
            for column in range(selected.shape[1]):
                alpha_values.append(float(temporal_psd_slope(selected[:, column], (0.01, 0.08))["alpha"]))
        rows.append({
            "model": expected["model"], "task": expected["task"], "method": expected["method"],
            "seed": expected["seed"], "run_id": expected["run_id"], "step": 500,
            "metadata_sha256": metadata_sha256,
            "temporal_psd_alpha": float(np.mean(alpha_values)),
            "temporal_psd_nested_sd": float(np.std(alpha_values, ddof=1)) if len(alpha_values) > 1 else math.nan,
            "nested_coordinate_count": coordinate_count, "frequency_band_cycles_per_step": "[0.01, 0.08]",
            "gradient_manifest_sha256": file_sha256(manifest_path),
        })
    return pd.DataFrame(rows)


def _save_figure(fig, figures_root: Path, stem: str, provenance: dict) -> list[Path]:
    figures_root.mkdir(parents=True, exist_ok=True)
    paths = [figures_root / f"{stem}.{extension}" for extension in ("svg", "pdf", "png")]
    for path in paths:
        metadata = (
            {"Date": None, "Creator": "revision_experiments.aggregate_results"}
            if path.suffix == ".svg" else
            {"CreationDate": None, "ModDate": None, "Creator": "revision_experiments.aggregate_results"}
            if path.suffix == ".pdf" else
            {"Software": "revision_experiments.aggregate_results"}
        )
        fig.savefig(path, dpi=240 if path.suffix == ".png" else None, bbox_inches="tight", metadata=metadata)
    sidecar = figures_root / f"{stem}.provenance.json"
    sidecar.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths + [sidecar]


def _generate_figures(seed_metrics: pd.DataFrame, mechanism: pd.DataFrame, figures_root: Path) -> list[Path]:
    import matplotlib
    matplotlib.rcParams["svg.hashsalt"] = "revision-experiments-task9"
    import matplotlib.pyplot as plt

    artifacts = []
    auc = seed_metrics[(seed_metrics["analysis_context"] == "core_500step") &
                       (seed_metrics["metric"] == "raw_auc500") & seed_metrics["method"].isin(
        ["peft_default", "iid_matched", "powerlaw_global_a06", "powerlaw_shuffle_a06"]
    )]
    if auc.empty:
        raise RuntimeError("matched-scale/shuffle AUC500 figure has no raw source rows")
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    tasks = sorted(auc["task"].unique())
    for method, group in auc.groupby("method", sort=True):
        stats = group.groupby("task")["value"].agg(["mean", "std"]).reindex(tasks)
        axis.errorbar(tasks, stats["mean"], yerr=stats["std"], marker="o", capsize=3,
                      label=method, color=METHOD_PALETTE.get(method))
    axis.set(title="Matched-scale and shuffle controls: raw training-loss AUC500",
             xlabel="Task", ylabel="Raw loss AUC, steps 1–500 (loss·step)")
    axis.legend(fontsize=8); axis.grid(alpha=0.2)
    artifacts.extend(_save_figure(figure, figures_root, "matched_scale_shuffle_auc500", {
        "title": axis.get_title(),
        "caption": "Points are seed means; error bars are ±1 sample SD across runs/seeds. AUC uses unsmoothed raw loss only.",
        "x_axis": "task", "y_axis": "raw loss AUC500 (loss·step)",
        "smoothing": "none; calculations use raw values",
        "source_runs": auc[["run_id", "metadata_sha256"]].drop_duplicates().to_dict("records"),
        "method_palette": METHOD_PALETTE,
    })); plt.close(figure)

    if mechanism.empty or "temporal_psd_alpha" not in mechanism:
        raise RuntimeError("early temporal PSD/gradient-capture figure has no raw source rows")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    psd = mechanism.dropna(subset=["temporal_psd_alpha"])
    for (model, method), group in psd.groupby(["model", "method"], sort=True):
        label = f"{model}:{method}"
        axes[0].errorbar([label], [group["temporal_psd_alpha"].mean()],
                         yerr=[group["temporal_psd_alpha"].std(ddof=1)], marker="o",
                         color=METHOD_PALETTE.get(method), capsize=3)
    axes[0].set(title="Early gradient temporal PSD", xlabel="Method", ylabel="PSD exponent α (dimensionless)")
    axes[0].tick_params(axis="x", rotation=70)
    capture = mechanism.dropna(subset=["gradient_capture_ratio"])
    for (model, method), group in capture.groupby(["model", "method"], sort=True):
        label = f"{model}:{method}"
        stats = group.groupby("step")["gradient_capture_ratio"].agg(["mean", "std"])
        axes[1].plot(stats.index, stats["mean"], label=label, color=METHOD_PALETTE.get(method))
        axes[1].fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                             color=METHOD_PALETTE.get(method), alpha=0.15)
    axes[1].set(title="Captured base-gradient energy", xlabel="Optimizer step", ylabel="Capture ratio (fraction)")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.2)
    figure.suptitle("Early temporal PSD and gradient-capture diagnostics")
    artifacts.extend(_save_figure(figure, figures_root, "early_temporal_psd_gradient_capture", {
        "title": figure._suptitle.get_text(),
        "caption": "PSD uses Welch fits over 0.01–0.08 cycles/step. Lines/points are run/seed means; bands/error bars are ±1 sample SD across runs/seeds. Coordinates and layers are nested measurements and do not increase n.",
        "x_axes": ["method", "optimizer step"], "y_axes": ["PSD exponent α", "gradient capture ratio"],
        "smoothing": "none",
        "source_runs": mechanism[["run_id", "metadata_sha256"]].drop_duplicates().to_dict("records"),
        "method_palette": METHOD_PALETTE,
    })); plt.close(figure)

    task_scores = seed_metrics[(seed_metrics["analysis_context"] == "downstream_2500step") &
                               (seed_metrics["metric_direction"] == "higher") &
                               (~seed_metrics["metric"].isin(["raw_auc500", "final_train_loss"]))]
    if task_scores.empty:
        raise RuntimeError("task metric versus steps/time figure has no evaluation rows")
    plotted = task_scores.assign(step=task_scores["endpoint"].str.removeprefix("step_").astype(int))
    panels = list(plotted.groupby(["model", "task", "metric"], sort=True))
    figure, axes = plt.subplots(len(panels), 2, figsize=(11.0, max(4.5, 3.5 * len(panels))), squeeze=False)
    for row_index, ((model, task, metric), panel) in enumerate(panels):
        step_axis, time_axis = axes[row_index]
        for method, group in panel.groupby("method", sort=True):
            by_step = group.groupby("step")["value"].agg(["mean", "std"])
            color = METHOD_PALETTE.get(method)
            step_axis.plot(by_step.index, by_step["mean"], marker="o", label=method, color=color)
            step_axis.fill_between(
                by_step.index, by_step["mean"] - by_step["std"], by_step["mean"] + by_step["std"],
                color=color, alpha=0.15,
            )
            for seed_index, (_, seed_rows) in enumerate(group.groupby("seed", sort=True)):
                seed_rows = seed_rows.sort_values("training_plus_initialization_seconds")
                time_axis.plot(
                    seed_rows["training_plus_initialization_seconds"], seed_rows["value"], marker="o",
                    label=method if seed_index == 0 else None, color=color, alpha=0.65,
                )
        label = f"{model} / {task} / {metric}"
        step_axis.set(title=f"{label}: score versus steps", xlabel="Optimizer step",
                      ylabel=f"{metric} (higher is better)")
        time_axis.set(title=f"{label}: score versus time", xlabel="Training + initialization time (seconds)",
                      ylabel=f"{metric} (higher is better)")
        step_axis.legend(fontsize=7); time_axis.legend(fontsize=7)
        step_axis.grid(alpha=0.2); time_axis.grid(alpha=0.2)
    figure.suptitle("Downstream task metric versus steps and cumulative training time")
    artifacts.extend(_save_figure(figure, figures_root, "task_metric_vs_steps_time", {
        "title": figure._suptitle.get_text(),
        "caption": "Step-domain lines are seed means with ±1 sample SD bands across runs/seeds. Time-domain lines show each seed/run without interpolation. Time includes initialization plus cumulative training; evaluation time is reported separately.",
        "x_axes": ["optimizer step", "training + initialization seconds"],
        "y_axis": "task score (higher is better)", "smoothing": "none",
        "source_runs": task_scores[["run_id", "metadata_sha256"]].drop_duplicates().to_dict("records"),
        "method_palette": METHOD_PALETTE,
    })); plt.close(figure)
    return artifacts


def _source_records(paths: Iterable[Path], project_root: Path, run_id: str | None = None) -> list[dict]:
    records = []
    for path in sorted(set(Path(item) for item in paths)):
        if not path.is_file():
            raise RuntimeError(f"missing source artifact during manifest creation: {path}")
        try:
            label = str(path.resolve().relative_to(project_root.resolve()))
        except ValueError as exc:
            raise RuntimeError(f"source/output artifact is outside project root: {path}") from exc
        records.append({"path": label, "sha256": file_sha256(path), "run_id": run_id})
    return records


def _validate_staged_publication(staged_tables: Path, generated_figures: list[Path]) -> None:
    observed_tables = {path.name for path in staged_tables.glob("*.csv")}
    if observed_tables != set(CANONICAL_TABLE_NAMES):
        raise RuntimeError("staging does not contain the exact canonical publication table set")
    expected_figures = {
        f"{stem}.{extension}" for stem in CANONICAL_FIGURE_STEMS
        for extension in ("svg", "pdf", "png", "provenance.json")
    }
    if {path.name for path in generated_figures} != expected_figures:
        raise RuntimeError("staging does not contain every canonical figure format and provenance sidecar")
    seed_table = pd.read_csv(staged_tables / "reviewer_seed_metrics.csv")
    keys = ["analysis_context", "model", "task", "method", "endpoint", "metric"]
    # Intermediate-checkpoint EVALUATION metrics are the deadline-amendment
    # descriptive evidence; training-derived metrics (raw_auc500 etc.) keep
    # their full three-seed contract at every endpoint label.
    evaluation_metrics = {
        "exact_match", "macro_accuracy", "pass_at_1", "pass_at_1_strict",
        "heldout_nll", "prometheus_absolute_score",
    }

    def _is_descriptive_n1(identity: tuple, group) -> bool:
        """Deadline-amendment descriptive evidence is explicitly n=1 (seed 1107).

        Two contract-declared cases: downstream intermediate-curve evaluation
        checkpoints (endpoint < 2500) and the 10k long-run boundary test (all
        endpoints). Everything else must carry the full three paired seeds.
        """
        context, model, task, method, endpoint, metric = identity
        step = str(endpoint).replace("step_", "").replace("update_", "")
        try:
            step_value = int(step)
        except ValueError:
            step_value = None
        if context == "scope_long_run":
            # The 10k boundary test is a one-seed experiment for every metric
            # it reports, including training-derived raw_auc500.
            return True
        if (
            context == "downstream_2500step"
            and step_value is not None and step_value < 2500
            and metric in evaluation_metrics
        ):
            return True
        return False

    for identity, group in seed_table.groupby(keys, dropna=False):
        seeds = set(pd.to_numeric(group["seed"], errors="raise").astype(int))
        if _is_descriptive_n1(identity, group):
            if seeds != {1107} or len(group) != 1:
                raise RuntimeError(
                    f"staged descriptive group must be exactly the single paired seed 1107: {identity}"
                )
            continue
        if seeds != {42, 123, 1107} or len(group) != 3:
            raise RuntimeError(f"staged seed metric group does not contain exact three-run evidence: {identity}")
    for name in CANONICAL_TABLE_NAMES:
        if pd.read_csv(staged_tables / name).empty:
            raise RuntimeError(f"staged canonical publication table is empty: {name}")
    for name in (
        "reviewer_summary_metrics.csv", "reviewer_paired_differences.csv",
        "reviewer_paired_difference_summary.csv",
    ):
        frame = pd.read_csv(staged_tables / name)
        n_runs = pd.to_numeric(frame["n_runs"], errors="coerce")
        if not frame.columns.intersection(["analysis_context", "endpoint"]).empty:
            descriptive = (
                (frame.get("analysis_context", "") == "scope_long_run")
                | (
                    (frame.get("analysis_context", "") == "downstream_2500step")
                    & pd.to_numeric(
                        frame["endpoint"].astype(str).str.replace("step_", "", regex=False)
                        .str.replace("update_", "", regex=False),
                        errors="coerce",
                    ) < 2500
                )
            )
            allowed = n_runs.eq(3) | (descriptive & n_runs.eq(1))
        else:
            allowed = n_runs.eq(3)
        if "n_runs" not in frame or not allowed.all():
            raise RuntimeError(f"staged publication table does not use exact three-run groups: {name}")


def build_publication_tables(
    expected_runs: list[dict], inventory: pd.DataFrame, runs_root: Path, evaluations_root: Path,
    audits_root: Path, output_root: Path, project_root: Path, config_dir: Path = CONFIG_DIR,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    matrix = load_matrix(config_dir / "baseline_search_matrix.yaml")
    baseline_rows, _ = validate_baseline_search_artifacts(
        matrix, runs_root, output_root / "baseline_search_all_trials.csv",
        output_root / "baseline_selected_configs.yaml",
    )
    seed_metrics = _collect_metric_rows(expected_runs, inventory, runs_root, evaluations_root, project_root)
    summaries = aggregate_seed_metrics(seed_metrics)
    paired = compute_paired_differences(seed_metrics)
    initialization, step_zero = _audit_tables(audits_root)
    mechanism = _mechanism_table(expected_runs, runs_root)
    tables = {
        "reviewer_seed_metrics.csv": seed_metrics,
        "reviewer_summary_metrics.csv": summaries,
        "reviewer_paired_differences.csv": paired,
        "reviewer_paired_difference_summary.csv": _paired_summary(paired),
        "initialization_statistics_table.csv": initialization,
        "step_zero_audit_table.csv": step_zero,
        "downstream_metric_table.csv": seed_metrics[seed_metrics["analysis_context"] == "downstream_2500step"],
        "time_to_equivalent_table.csv": _time_to_equivalent(seed_metrics),
        "all_linear_longer_run_table.csv": seed_metrics[seed_metrics["analysis_context"].str.startswith("scope_")],
        "early_psd_gradient_capture_table.csv": mechanism,
        "baseline_search_supplementary_table.csv": pd.DataFrame(baseline_rows, columns=CSV_FIELDS),
    }
    return tables, seed_metrics, mechanism


def _write_publication_artifacts(
    expected_runs: list[dict], inventory: pd.DataFrame, runs_root: Path, evaluations_root: Path,
    audits_root: Path, output_root: Path, figures_root: Path, project_root: Path,
    aggregation_arguments: dict[str, str], config_dir: Path,
) -> dict:
    tables, seed_metrics, mechanism = build_publication_tables(
        expected_runs, inventory, runs_root, evaluations_root, audits_root, output_root, project_root, config_dir
    )
    for name, frame in tables.items():
        if frame.empty:
            raise RuntimeError(f"refusing to publish empty scientific table: {name}")
        if (output_root / name).exists():
            raise RuntimeError(f"refusing to overwrite existing reviewer artifact: {output_root / name}")
    figure_names = [
        f"{stem}.{extension}"
        for stem in ("matched_scale_shuffle_auc500", "early_temporal_psd_gradient_capture", "task_metric_vs_steps_time")
        for extension in ("svg", "pdf", "png", "provenance.json")
    ]
    existing_figures = [str(figures_root / name) for name in figure_names if (figures_root / name).exists()]
    if existing_figures:
        raise RuntimeError(f"refusing to overwrite existing reviewer figure(s): {', '.join(existing_figures)}")
    manifest_path = output_root / "reviewer_tables_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite existing reviewer artifact: {manifest_path}")

    # Build and hash the complete source set before any reviewer output is
    # moved into place. This prevents a late source failure from leaving a
    # partial, apparently publishable evidence set.
    sources = []
    checkpoint_trees = []
    for expected in expected_runs:
        run_dir = runs_root / expected["run_id"]
        paths = [run_dir / name for name in (
            "metadata.json", "config.yaml", "raw_loss.jsonl", "timing.jsonl",
            "initialization_stats.json", "summary.json", "COMPLETED",
        )]
        for checkpoint in expected["expected_checkpoints"]:
            checkpoint_dir = run_dir / "checkpoints" / f"step_{checkpoint:06d}"
            paths.extend(path for path in checkpoint_dir.rglob("*") if path.is_file())
            checkpoint_trees.append({
                "run_id": expected["run_id"], "checkpoint": int(checkpoint),
                "tree_sha256": _tree_hash(checkpoint_dir),
            })
        if "baseline_fairness_screening" in expected.get("analysis_contexts", ()):
            paths.append(run_dir / "validation_loss.jsonl")
        if expected["mechanism_required"]:
            gradient_manifest = _read_json(run_dir / "gradients/manifest.json")
            paths.extend([run_dir / "gradients/manifest.json", run_dir / "gradient_diagnostics.jsonl"])
            paths.extend(run_dir / "gradients" / item["file"] for item in gradient_manifest["parameters"])
        for requirement in expected["expected_evaluations"]:
            result_path = evaluations_root / evaluation_filename(expected["run_id"], requirement["checkpoint"], requirement["task"])
            result = validate_evaluation_artifact(result_path, requirement, expected["run_id"], project_root)
            paths.extend([result_path, Path(result["resolved_prediction_artifact"])])
            paths.extend(Path(path) for path in result.get("resolved_extra_artifacts", []))
        sources.extend(_source_records(paths, project_root, expected["run_id"]))
    sources.extend(_source_records([
        audits_root / "initialization_statistics.csv", audits_root / "step_zero_audit.json",
        audits_root / "step_zero_batch_losses.csv", output_root / "baseline_search_all_trials.csv",
        output_root / "baseline_selected_configs.yaml",
    ], project_root))

    output_paths = []
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task9-publication-", dir=output_root.parent) as temporary:
        staging = Path(temporary)
        staged_tables = staging / "aggregate"
        staged_figures = staging / "figures"
        for name, frame in tables.items():
            _atomic_csv(frame, staged_tables / name)
        generated_figures = _generate_figures(seed_metrics, mechanism, staged_figures)
        _validate_staged_publication(staged_tables, generated_figures)
        output_root.mkdir(parents=True, exist_ok=True)
        figures_root.mkdir(parents=True, exist_ok=True)
        for name in tables:
            destination = output_root / name
            os.replace(staged_tables / name, destination)
            output_paths.append(destination)
        for staged_path in generated_figures:
            destination = figures_root / staged_path.name
            os.replace(staged_path, destination)
            output_paths.append(destination)
    manifest = {
        "schema_version": 1, "aggregation_command": "python revision_experiments/scripts/aggregate_results.py",
        "aggregation_arguments": aggregation_arguments,
        "aggregation_script_sha256": file_sha256(Path(__file__)),
        "experimental_unit": "run/seed",
        "paired_join_keys": ["analysis_context", "model", "task", "seed", "checkpoint/endpoint", "metric"],
        "smoothing_policy": "display-only when explicitly labeled; all calculations use raw values",
        "error_band_definition": "±1 sample standard deviation across independent runs/seeds",
        "inventory_sha256": file_sha256(output_root / "run_inventory.csv"),
        "source_artifacts": sources, "checkpoint_trees": checkpoint_trees,
        "output_artifacts": _source_records(output_paths, project_root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"table_count": len(tables), "figure_file_count": len(output_paths) - len(tables), "manifest": str(manifest_path)}


def run_aggregation(
    *, config_dir: str | Path = CONFIG_DIR,
    runs_root: str | Path = ROOT / "revision_experiments/results/runs",
    evaluations_root: str | Path = ROOT / "revision_experiments/results/evaluations",
    audits_root: str | Path = ROOT / "revision_experiments/results/audits",
    output_root: str | Path = ROOT / "revision_experiments/results/aggregate",
    figures_root: str | Path = ROOT / "revision_experiments/results/figures",
    project_root: str | Path = ROOT,
) -> dict:
    config_dir, runs_root = Path(config_dir), Path(runs_root)
    evaluations_root, audits_root = Path(evaluations_root), Path(audits_root)
    output_root, figures_root, project_root = Path(output_root), Path(figures_root), Path(project_root)
    arguments = {
        "config_dir": str(config_dir.resolve()), "runs_root": str(runs_root.resolve()),
        "evaluations_root": str(evaluations_root.resolve()), "audits_root": str(audits_root.resolve()),
        "output_root": str(output_root.resolve()), "figures_root": str(figures_root.resolve()),
        "project_root": str(project_root.resolve()),
    }
    expected = discover_expected_runs(
        config_dir, selected_path=output_root / "baseline_selected_configs.yaml"
    )
    inventory = build_run_inventory(expected, runs_root, evaluations_root, project_root=project_root)
    inventory_path = output_root / "run_inventory.csv"
    _atomic_csv(inventory, inventory_path)
    blockers = _publication_prerequisites(inventory, audits_root, output_root, config_dir, runs_root)
    result = {
        "schema_version": 1, "expected_unique_runs": len(expected),
        "matrix_memberships": sum(len(item["matrix_membership"]) for item in expected),
        "inventory_rows": len(inventory),
        "status_counts": inventory["status"].value_counts().sort_index().to_dict(),
        "matrix_sha256": _matrix_hashes(config_dir), "inventory_sha256": file_sha256(inventory_path),
        "aggregation_command": "python revision_experiments/scripts/aggregate_results.py",
        "aggregation_arguments": arguments,
        "aggregation_script_sha256": file_sha256(Path(__file__)),
        "reviewer_artifacts_generated": False, "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
    }
    if not blockers:
        result["publication"] = _write_publication_artifacts(
            expected, inventory, runs_root, evaluations_root, audits_root, output_root, figures_root, project_root,
            arguments, config_dir,
        )
        result["reviewer_artifacts_generated"] = True
    status_path = output_root / "aggregation_status.json"
    status_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "revision_experiments/results/runs")
    parser.add_argument("--evaluations-root", type=Path, default=ROOT / "revision_experiments/results/evaluations")
    parser.add_argument("--audits-root", type=Path, default=ROOT / "revision_experiments/results/audits")
    parser.add_argument("--output-root", type=Path, default=ROOT / "revision_experiments/results/aggregate")
    parser.add_argument("--figures-root", type=Path, default=ROOT / "revision_experiments/results/figures")
    args = parser.parse_args()
    result = run_aggregation(
        config_dir=args.config_dir, runs_root=args.runs_root, evaluations_root=args.evaluations_root,
        audits_root=args.audits_root, output_root=args.output_root, figures_root=args.figures_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
