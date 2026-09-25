#!/usr/bin/env python3
"""Fail-closed validation of Gate 1-R and Gate 2 before formal training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import yaml

from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_LEGACY_CATEGORIES = ("configs", "logs", "checkpoints")
REQUIRED_GATE2_CASES = ("openpangu__gsm8k", "qwen__cmmlu")
REQUIRED_GATE2_METHODS = ("base", "peft_default", "powerlaw_global_a06")
REQUIRED_INTEGRATION_SMOKE_CASES = (
    "mbpp_adapter_and_executor", "lora_one_initialization", "gradient_logger_20step",
)
EXPECTED_MODEL_REVISIONS = {
    "openpangu": "0ae1841cbd53f5218f2ce5dc63083d5382cfc9f5",
    "qwen": "e25af2efae60472008fbeaf5fb7c4274a87f78d4",
}


def _read_json(path: Path, label: str, errors: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"missing or invalid {label}: {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"invalid {label}: expected a JSON object")
        return {}
    return value


def _resolve_within(label: object, base: Path, boundary: Path) -> Path | None:
    if not isinstance(label, str) or not label.strip():
        return None
    candidate = Path(label)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    return resolved if resolved.is_relative_to(boundary.resolve()) else None


def _manifest_entries(manifest: Path, import_root: Path, errors: list[str]) -> dict[str, str]:
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"missing legacy SHA256 manifest: {manifest}: {exc}")
        return {}
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            errors.append(f"invalid legacy SHA256 line {line_number}")
            continue
        digest, label = match.groups()
        path = _resolve_within(label, import_root, import_root)
        if path is None or path == manifest.resolve():
            errors.append(f"unsafe legacy manifest path at line {line_number}: {label}")
            continue
        relative = path.relative_to(import_root.resolve()).as_posix()
        if relative in entries:
            errors.append(f"duplicate legacy manifest path: {relative}")
            continue
        entries[relative] = digest
    if not entries:
        errors.append("legacy SHA256 manifest contains no imported artifacts")
    return entries


def validate_legacy_artifacts(project_root: str | Path = ROOT) -> tuple[list[str], dict]:
    """Validate mapped submitted files and their immutable SHA256 manifest."""
    project_root = Path(project_root).resolve()
    errors: list[str] = []
    map_path = project_root / "revision_experiments/config/local_repository_map.yaml"
    try:
        repository_map = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"missing or invalid local repository map: {exc}"], {}
    submitted = repository_map.get("submitted_artifacts") if isinstance(repository_map, dict) else None
    if not isinstance(submitted, dict):
        return ["local repository map has no submitted_artifacts object"], {}
    import_root = _resolve_within(submitted.get("import_root"), project_root, project_root)
    if import_root is None:
        return ["submitted_artifacts.import_root is missing or outside the project"], {}
    manifest = _resolve_within(submitted.get("manifest"), project_root, import_root)
    if manifest is None:
        return ["submitted_artifacts.manifest is missing or outside import_root"], {}
    entries = _manifest_entries(manifest, import_root, errors)
    mapped: dict[str, list[str]] = {}
    mapped_files: set[str] = set()
    for category in REQUIRED_LEGACY_CATEGORIES:
        labels = submitted.get(category)
        if not isinstance(labels, list) or not labels:
            errors.append(f"submitted_artifacts.{category} must contain at least one concrete path")
            mapped[category] = []
            continue
        mapped[category] = []
        for label in labels:
            path = _resolve_within(label, project_root, import_root)
            if path is None or not path.exists():
                errors.append(f"missing or unsafe submitted {category} path: {label}")
                continue
            files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
            if not files:
                errors.append(f"submitted {category} path contains no files: {label}")
                continue
            mapped[category].append(str(path.relative_to(project_root)))
            for item in files:
                relative = item.resolve().relative_to(import_root.resolve()).as_posix()
                mapped_files.add(relative)
                expected = entries.get(relative)
                if expected is None:
                    errors.append(f"mapped submitted artifact is absent from SHA256SUMS: {relative}")
                elif file_sha256(item) != expected:
                    errors.append(f"submitted artifact SHA256 mismatch: {relative}")
    for relative, expected in entries.items():
        path = import_root / relative
        if not path.is_file():
            errors.append(f"legacy manifest file is missing: {relative}")
        elif relative not in mapped_files:
            errors.append(f"legacy manifest artifact is not assigned to configs/logs/checkpoints: {relative}")
        elif not SHA256_RE.fullmatch(expected):
            errors.append(f"invalid legacy artifact SHA256: {relative}")
    summary = {
        "import_root": str(import_root.relative_to(project_root)),
        "manifest": str(manifest.relative_to(project_root)),
        "manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
        "artifact_count": len(entries),
        "mapped": mapped,
    }
    return errors, summary


def validate_gate1_reproduction(project_root: str | Path = ROOT) -> tuple[list[str], dict]:
    """Validate the author-reviewed, hash-bound reconstructed reproduction report."""
    project_root = Path(project_root).resolve()
    errors, legacy = validate_legacy_artifacts(project_root)
    report_path = project_root / "revision_experiments/results/audits/gate1_reproduction.json"
    report = _read_json(report_path, "Gate 1-R report", errors)
    if not report:
        return errors, {"passed": False, "legacy": legacy}
    if report.get("schema_version") != 2 or report.get("gate_id") != "gate1-r":
        errors.append("Gate 1-R report schema/identity mismatch")
    if report.get("gate_passed") is not True:
        errors.append("Gate 1-R report does not record gate_passed=true")
    if report.get("provenance_status") != "reconstructed_without_exact_submitted_config_or_git_history":
        errors.append("Gate 1-R provenance status mismatch")
    if not COMMIT_RE.fullmatch(str(report.get("reconstructed_git_commit", ""))):
        errors.append("Gate 1-R report has no valid reconstructed Git commit")
    if report.get("submitted_manifest_sha256") != legacy.get("manifest_sha256"):
        errors.append("Gate 1-R submitted manifest hash mismatch")
    if report.get("submitted_artifacts") != legacy.get("mapped"):
        errors.append("Gate 1-R submitted artifact mapping mismatch")
    replay = report.get("replay")
    if not isinstance(replay, dict):
        errors.append("Gate 1-R report has no replay object")
        replay = {}
    if replay.get("model") != "openpangu" or replay.get("method") != "peft_default":
        errors.append("Gate 1-R replay must use openpangu PEFT-default")
    if not isinstance(replay.get("task"), str) or not replay.get("task", "").strip():
        errors.append("Gate 1-R replay task must be recorded")
    if isinstance(replay.get("seed"), bool) or not isinstance(replay.get("seed"), int) or replay.get("seed", -1) < 0:
        errors.append("Gate 1-R replay seed must be a non-negative integer")
    if replay.get("exit_code") != 0:
        errors.append("Gate 1-R replay did not exit successfully")
    if not isinstance(replay.get("command"), list) or not replay.get("command") or not all(
        isinstance(token, str) and token for token in replay.get("command", [])
    ):
        errors.append("Gate 1-R replay command must be a non-empty argv list")
    evidence = replay.get("evidence_files")
    required_evidence_kinds = {"config", "raw_loss", "summary", "checkpoint"}
    if not isinstance(evidence, list):
        errors.append("Gate 1-R replay must bind evidence_files")
    else:
        observed_kinds = [record.get("kind") for record in evidence if isinstance(record, dict)]
        if set(observed_kinds) != required_evidence_kinds or len(observed_kinds) != len(required_evidence_kinds):
            errors.append("Gate 1-R replay evidence must contain config/raw_loss/summary/checkpoint exactly once")
        for record in evidence:
            if not isinstance(record, dict):
                errors.append("Gate 1-R replay evidence record is not an object")
                continue
            path = _resolve_within(record.get("path"), project_root, project_root)
            if path is None or not path.is_file():
                errors.append(f"missing or unsafe Gate 1-R replay evidence: {record.get('path')}")
            elif file_sha256(path) != record.get("sha256"):
                errors.append(f"Gate 1-R replay evidence hash mismatch: {record.get('path')}")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        errors.append("Gate 1-R report has no comparison object")
        comparison = {}
    mode = comparison.get("mode")
    if mode not in {"multi_seed_range", "single_seed_shape_only"}:
        errors.append("Gate 1-R comparison mode is invalid")
    if comparison.get("configuration_status") not in {"exact", "reconstructed_from_legacy_evidence"}:
        errors.append("Gate 1-R configuration status is invalid")
    if comparison.get("data_order_status") not in {"exact", "deterministic_reconstruction"}:
        errors.append("Gate 1-R data-order status is invalid")
    if comparison.get("curve_shape_match") is not True:
        errors.append("Gate 1-R comparison requires curve_shape_match=true")
    if comparison.get("configuration_status") == "reconstructed_from_legacy_evidence":
        if any(
            not isinstance(report.get(field), list) or not report.get(field)
            or not all(isinstance(value, str) and value.strip() for value in report[field])
            for field in ("declared_adaptations", "limitations")
        ):
            errors.append("reconstructed Gate 1-R must record adaptations and limitations")
    if comparison.get("reused_as_revision_result") is not False:
        errors.append("Gate 1-R replay must not be reused as a revision result")
    if mode == "multi_seed_range":
        for key in ("auc500_within_submitted_seed_range", "endpoint_loss_within_submitted_seed_range"):
            if comparison.get(key) is not True:
                errors.append(f"multi-seed Gate 1-R requires {key}=true")
    elif mode == "single_seed_shape_only" and any(
        comparison.get(key) is not None
        for key in ("auc500_within_submitted_seed_range", "endpoint_loss_within_submitted_seed_range")
    ):
        errors.append("single-seed Gate 1-R must not claim submitted seed-range checks")
    if not isinstance(report.get("reviewed_by"), str) or not report.get("reviewed_by", "").strip():
        errors.append("Gate 1-R report must identify its reviewer")
    if not isinstance(report.get("notes"), str) or not report.get("notes", "").strip():
        errors.append("Gate 1-R report must document comparison notes")
    return errors, {"passed": not errors, "report": str(report_path.relative_to(project_root)), "legacy": legacy}


def _csv_rows(path: Path, errors: list[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        errors.append(f"missing {label}: {path}: {exc}")
        return []


def validate_gate2_step_zero(project_root: str | Path = ROOT) -> tuple[list[str], dict]:
    """Validate both immutable 100-batch cases and their canonical Gate 2 bundle."""
    project_root = Path(project_root).resolve()
    errors: list[str] = []
    audits = project_root / "revision_experiments/results/audits"
    cases_dir = audits / "step_zero_cases"
    report_path = audits / "step_zero_audit.json"
    combined_path = audits / "step_zero_batch_losses.csv"
    report = _read_json(report_path, "Gate 2 report", errors)
    if not report:
        return errors, {"passed": False}
    if report.get("schema_version") != 4 or report.get("gate_passed") is not True:
        errors.append("Gate 2 report schema is invalid or gate_passed is not true")
    if report.get("required_cases") != list(REQUIRED_GATE2_CASES) or report.get("case_count") != 2:
        errors.append("Gate 2 report must contain the exact two required cases")
    if not combined_path.is_file() or report.get("combined_batch_losses_sha256") != (
        file_sha256(combined_path) if combined_path.is_file() else None
    ):
        errors.append("Gate 2 combined loss CSV is missing or hash-mismatched")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        errors.append("Gate 2 report cases must contain exactly two records")
        cases = []
    observed_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if observed_ids != list(REQUIRED_GATE2_CASES):
        errors.append("Gate 2 case order/identity mismatch")
    expected_combined_rows: list[dict[str, str]] = []
    for expected_id, case in zip(REQUIRED_GATE2_CASES, cases):
        if not isinstance(case, dict):
            continue
        model, task = expected_id.split("__", 1)
        case_json = cases_dir / f"{expected_id}.json"
        case_csv = cases_dir / f"{expected_id}.csv"
        bound = _read_json(case_json, f"Gate 2 case {expected_id}", errors)
        if bound and bound != case:
            errors.append(f"Gate 2 embedded case differs from immutable case file: {expected_id}")
        if case.get("schema_version") != 4 or case.get("case_id") != expected_id:
            errors.append(f"Gate 2 case schema/identity mismatch: {expected_id}")
        if case.get("model") != model or case.get("task") != task or case.get("batches") != 100:
            errors.append(f"Gate 2 case must be {expected_id} with exactly 100 batches")
        if not str(case.get("checkpoint", "")).endswith(f"@{EXPECTED_MODEL_REVISIONS[model]}"):
            errors.append(f"Gate 2 case checkpoint revision mismatch: {expected_id}")
        if case.get("equivalence_passed") is not True or case.get("loss_anomaly") is not False:
            errors.append(f"Gate 2 case failed equivalence/loss-anomaly gate: {expected_id}")
        diagnostics = case.get("batch_target_diagnostics")
        if (
            not isinstance(diagnostics, list) or len(diagnostics) != 100
            or [row.get("batch") for row in diagnostics if isinstance(row, dict)] != list(range(100))
            or any(
                not isinstance(row, dict)
                or not isinstance(row.get("valid_target_tokens"), int)
                or row.get("valid_target_tokens", 0) <= 0
                or row.get("padding_target_tokens") != 0
                for row in diagnostics
            )
        ):
            errors.append(f"Gate 2 per-batch target diagnostics are invalid: {expected_id}")
            diagnostics = []
        valid_total = sum(row["valid_target_tokens"] for row in diagnostics) if diagnostics else 0
        padding_total = sum(row["padding_target_tokens"] for row in diagnostics) if diagnostics else -1
        if (
            case.get("ignore_index") != -100
            or case.get("valid_target_tokens_total") != valid_total
            or case.get("padding_target_tokens_total") != padding_total
            or padding_total != 0
        ):
            errors.append(f"Gate 2 label masking contract failed: {expected_id}")
        if not isinstance(case.get("tokenizer_artifacts_hash"), str) or len(case["tokenizer_artifacts_hash"]) != 64:
            errors.append(f"Gate 2 tokenizer hash is invalid: {expected_id}")
        if not isinstance(case.get("tokenizer_artifacts"), list) or not case.get("tokenizer_artifacts"):
            errors.append(f"Gate 2 tokenizer artifact list is invalid: {expected_id}")
        tolerance = case.get("tolerance")
        equivalence = case.get("equivalence")
        if not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
            errors.append(f"Gate 2 tolerance is invalid: {expected_id}")
            tolerance = -1.0
        if not isinstance(equivalence, dict) or set(equivalence) != set(REQUIRED_GATE2_METHODS[1:]):
            errors.append(f"Gate 2 equivalence method coverage mismatch: {expected_id}")
            equivalence = {}
        for method, values in equivalence.items():
            if not isinstance(values, dict):
                errors.append(f"Gate 2 equivalence row is invalid: {expected_id}/{method}")
                continue
            numeric = [
                values.get("max_abs_logit_difference"), values.get("mean_abs_logit_difference"),
                values.get("max_abs_loss_difference"), values.get("mean_abs_loss_difference"),
            ]
            if isinstance(values.get("b_nonzero"), bool) or values.get("b_nonzero") != 0 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
                for value in numeric
            ) or float(values.get("max_abs_logit_difference", math.inf)) > float(tolerance) or abs(
                float(values.get("mean_abs_logit_difference", math.inf))
            ) > float(tolerance) or abs(
                float(values.get("max_abs_loss_difference", math.inf))
            ) > float(tolerance) or abs(
                float(values.get("mean_abs_loss_difference", math.inf))
            ) > float(tolerance):
                errors.append(f"Gate 2 equivalence values violate tolerance/B=0: {expected_id}/{method}")
        diagnostic = case.get("loss_anomaly_diagnostic")
        expected_diagnostic_status = "resolved_legacy_padding_labels" if model == "openpangu" else "not_triggered"
        if (
            not isinstance(diagnostic, dict) or diagnostic.get("status") != expected_diagnostic_status
            or diagnostic.get("trigger_threshold") != 10.0
            or not isinstance(diagnostic.get("checks"), dict)
        ):
            errors.append(f"Gate 2 loss-anomaly diagnostic is incomplete/unresolved: {expected_id}")
        label_comparison = case.get("legacy_label_comparison")
        if model == "openpangu":
            legacy_audit = audits / "legacy_evidence.json"
            if (
                not isinstance(label_comparison, dict)
                or label_comparison.get("status") != "resolved_by_padding_label_masking"
                or label_comparison.get("same_batches") is not True
                or label_comparison.get("legacy_padding_targets_scored", 0) <= 0
                or label_comparison.get("masked_median_loss") != case.get("median_losses", {}).get("base")
                or label_comparison.get("legacy_unmasked_median_loss", 0) <= 10.0
                or label_comparison.get("median_loss_increase", 0) <= 0
                or not legacy_audit.is_file()
                or label_comparison.get("legacy_evidence_sha256") != file_sha256(legacy_audit)
            ):
                errors.append("Gate 2 openPangu legacy padding-label explanation is missing or invalid")
        elif label_comparison != {"status": "not_applicable"}:
            errors.append(f"Gate 2 non-openPangu legacy label comparison is invalid: {expected_id}")
        medians = case.get("median_losses")
        if not isinstance(medians, dict) or set(medians) != set(REQUIRED_GATE2_METHODS) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
            for value in medians.values()
        ):
            errors.append(f"Gate 2 median-loss coverage is invalid: {expected_id}")
        elif model == "openpangu" and float(medians["base"]) > 10.0:
            errors.append("Gate 2 openPangu base median loss remains above 10")
        if case.get("batch_losses_file") != case_csv.name or not case_csv.is_file() or case.get("batch_losses_sha256") != (
            file_sha256(case_csv) if case_csv.is_file() else None
        ):
            errors.append(f"Gate 2 case loss CSV is missing or hash-mismatched: {expected_id}")
        rows = _csv_rows(case_csv, errors, f"Gate 2 case CSV {expected_id}")
        expected_combined_rows.extend(rows)
        if len(rows) != 300:
            errors.append(f"Gate 2 case must contain 3 methods x 100 loss rows: {expected_id}")
        if any(
            row.get("model") != model or row.get("task") != task or row.get("seed") != str(case.get("seed"))
            for row in rows
        ):
            errors.append(f"Gate 2 case CSV identity/seed mismatch: {expected_id}")
        for method in REQUIRED_GATE2_METHODS:
            method_rows = [row for row in rows if row.get("method") == method]
            try:
                batches = {int(row["batch"]) for row in method_rows}
                finite = all(math.isfinite(float(row["loss"])) for row in method_rows)
            except (KeyError, TypeError, ValueError):
                batches, finite = set(), False
            if len(method_rows) != 100 or batches != set(range(100)) or not finite:
                errors.append(f"Gate 2 loss rows are incomplete/non-finite: {expected_id}/{method}")
        checkpoint_audit = audits / f"{model}_checkpoint_verification.json"
        expected_checkpoint_hash = file_sha256(checkpoint_audit) if checkpoint_audit.is_file() else None
        if case.get("checkpoint_verification_sha256") != expected_checkpoint_hash:
            errors.append(f"Gate 2 checkpoint verification binding mismatch: {expected_id}")
    combined_rows = _csv_rows(combined_path, errors, "Gate 2 combined loss CSV")
    if len(combined_rows) != 600:
        errors.append("Gate 2 combined loss CSV must contain exactly 600 rows")
    if combined_rows != expected_combined_rows:
        errors.append("Gate 2 combined loss CSV differs from the ordered immutable case CSVs")
    return errors, {"passed": not errors, "report": str(report_path.relative_to(project_root))}


def validate_model_gate_evidence(project_root: str | Path = ROOT) -> tuple[list[str], dict]:
    project_root = Path(project_root).resolve()
    audits = project_root / "revision_experiments/results/audits"
    errors: list[str] = []
    summary = {}
    for model, revision in EXPECTED_MODEL_REVISIONS.items():
        verification = _read_json(audits / f"{model}_checkpoint_verification.json", f"{model} checkpoint audit", errors)
        smoke = _read_json(audits / f"{model}_gpu_load_smoke.json", f"{model} GPU load smoke", errors)
        if verification.get("verified") is not True or verification.get("revision") != revision:
            errors.append(f"{model} checkpoint audit is not verified at the pinned revision")
        checkpoint_label = smoke.get("checkpoint")
        if (
            smoke.get("passed") is not True
            or not isinstance(checkpoint_label, str)
            or not checkpoint_label.endswith(f"@{revision}")
        ):
            errors.append(f"{model} GPU load smoke is missing, failed, or revision-mismatched")
        expected_hash = file_sha256(audits / f"{model}_checkpoint_verification.json") if verification else None
        if smoke.get("checkpoint_verification_sha256") != expected_hash:
            errors.append(f"{model} GPU load smoke is not bound to the checkpoint audit")
        summary[model] = {"revision": revision, "verified": not any(error.startswith(model) for error in errors)}
    return errors, summary


def validate_integration_smokes(project_root: str | Path = ROOT) -> tuple[list[str], dict]:
    """Validate the hash-bound, non-formal pre-matrix integration-smoke report."""

    project_root = Path(project_root).resolve()
    errors: list[str] = []
    report_path = project_root / "revision_experiments/results/audits/integration_smoke.json"
    report = _read_json(report_path, "integration smoke report", errors)
    if not report:
        return errors, {"passed": False}
    if report.get("schema_version") != 1 or report.get("passed") is not True:
        errors.append("integration smoke report schema is invalid or passed is not true")
    if report.get("formal_results") is not False:
        errors.append("integration smoke evidence must be explicitly non-formal")
    cases = report.get("cases")
    case_rows = cases if isinstance(cases, list) else []
    observed = [case.get("case") for case in case_rows if isinstance(case, dict)]
    if observed != list(REQUIRED_INTEGRATION_SMOKE_CASES) or any(
        not isinstance(case, dict) or case.get("passed") is not True for case in case_rows
    ):
        errors.append("integration smoke report does not contain the exact three passing cases")
    matrix = _resolve_within(report.get("matrix"), project_root, project_root)
    if matrix is None or not matrix.is_file() or report.get("matrix_sha256") != (
        file_sha256(matrix) if matrix is not None and matrix.is_file() else None
    ):
        errors.append("integration smoke matrix is missing, unsafe, or hash-mismatched")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("integration smoke report has no bound artifacts")
    else:
        labels = []
        for record in artifacts:
            if not isinstance(record, dict):
                errors.append("integration smoke artifact record is not an object")
                continue
            path = _resolve_within(record.get("path"), project_root, project_root)
            labels.append(record.get("path"))
            if path is None or not path.is_file():
                errors.append(f"integration smoke artifact is missing or unsafe: {record.get('path')}")
            elif record.get("sha256") != file_sha256(path):
                errors.append(f"integration smoke artifact hash mismatch: {record.get('path')}")
        if len(labels) != len(set(labels)):
            errors.append("integration smoke report contains duplicate artifact paths")
    return errors, {
        "passed": not errors,
        "report": str(report_path.relative_to(project_root)),
    }


def validate_execution_gates(
    project_root: str | Path = ROOT, *, require_integration_smokes: bool = True,
) -> dict:
    model_errors, models = validate_model_gate_evidence(project_root)
    gate1_errors, gate1 = validate_gate1_reproduction(project_root)
    gate2_errors, gate2 = validate_gate2_step_zero(project_root)
    prerequisite_errors = model_errors + gate1_errors + gate2_errors
    smoke_errors, smokes = validate_integration_smokes(project_root) if require_integration_smokes else ([], {"required": False})
    errors = prerequisite_errors + smoke_errors
    return {
        "schema_version": 1,
        "ready_for_integration_smokes": not prerequisite_errors,
        "ready_for_formal_training": bool(require_integration_smokes) and not errors,
        "model_gate_evidence": models,
        "gate1_reproduction": gate1,
        "gate2_step_zero": gate2,
        "integration_smokes": smokes,
        "errors": errors,
    }


def require_execution_gates(
    project_root: str | Path = ROOT, *, require_integration_smokes: bool = True,
) -> dict:
    payload = validate_execution_gates(
        project_root, require_integration_smokes=require_integration_smokes,
    )
    readiness = "ready_for_formal_training" if require_integration_smokes else "ready_for_integration_smokes"
    if not payload[readiness]:
        label = "Formal training" if require_integration_smokes else "Integration-smoke prerequisite"
        raise RuntimeError(f"{label} gates are not satisfied: " + "; ".join(payload["errors"]))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate_execution_gates(args.project_root)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["ready_for_formal_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
