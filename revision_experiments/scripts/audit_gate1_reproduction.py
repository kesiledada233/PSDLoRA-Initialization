#!/usr/bin/env python3
"""Finalize immutable Gate 1-R evidence after an original-command replay."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from revision_experiments.scripts.execution_gates import ROOT, validate_gate1_reproduction, validate_legacy_artifacts
from revision_experiments.scripts.schema import file_sha256


def _resolve_replay_evidence(label: object, project_root: Path) -> Path:
    if not isinstance(label, str) or not label.strip():
        raise RuntimeError("Gate 1-R evidence path must be a non-empty string")
    path = Path(label)
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    evidence_root = (project_root / "revision_experiments/results/gates/gate1_reproduction").resolve()
    if not resolved.is_relative_to(evidence_root) or not resolved.is_file():
        raise RuntimeError(f"Gate 1-R evidence must be an existing file under {evidence_root}: {label}")
    return resolved


def _git_commit(project_root: Path) -> str:
    reconstructed = project_root / ".git-reconstructed"
    prefix = ["git", f"--git-dir={reconstructed}", f"--work-tree={project_root}"] if reconstructed.is_dir() else ["git"]
    result = subprocess.run(prefix + ["rev-parse", "HEAD"], cwd=project_root, text=True, capture_output=True, check=False)
    status = subprocess.run(prefix + ["status", "--porcelain"], cwd=project_root, text=True, capture_output=True, check=False)
    if result.returncode != 0 or len(result.stdout.strip()) != 40:
        raise RuntimeError("A valid Git commit is required to finalize Gate 1-R")
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("Gate 1-R must be finalized from a clean worktree")
    return result.stdout.strip()


def build_report(spec: dict, project_root: Path) -> dict:
    legacy_errors, legacy = validate_legacy_artifacts(project_root)
    if legacy_errors:
        raise RuntimeError("Legacy artifact validation failed: " + "; ".join(legacy_errors))
    replay = spec.get("replay")
    comparison = spec.get("comparison")
    if not isinstance(replay, dict) or not isinstance(comparison, dict):
        raise RuntimeError("Gate 1-R spec requires replay and comparison objects")
    if replay.get("model") != "openpangu" or replay.get("method") != "peft_default":
        raise RuntimeError("Gate 1-R replay must use openpangu PEFT-default")
    if replay.get("exit_code") != 0:
        raise RuntimeError("Gate 1-R replay exit_code must be zero")
    command = replay.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(token, str) and token for token in command):
        raise RuntimeError("Gate 1-R replay command must be a non-empty argv list")
    evidence_labels = replay.get("evidence_files")
    required_kinds = {"config", "raw_loss", "summary", "checkpoint"}
    if not isinstance(evidence_labels, list) or {
        record.get("kind") for record in evidence_labels if isinstance(record, dict)
    } != required_kinds or len(evidence_labels) != len(required_kinds):
        raise RuntimeError("Gate 1-R replay requires config/raw_loss/summary/checkpoint evidence exactly once")
    evidence = []
    for record in evidence_labels:
        path = _resolve_replay_evidence(record.get("path"), project_root)
        evidence.append({
            "kind": record["kind"], "path": str(path.relative_to(project_root)),
            "sha256": file_sha256(path),
        })
    mode = comparison.get("mode")
    if mode not in {"multi_seed_range", "single_seed_shape_only"}:
        raise RuntimeError("Gate 1-R comparison mode must be multi_seed_range or single_seed_shape_only")
    if comparison.get("configuration_status") not in {"exact", "reconstructed_from_legacy_evidence"}:
        raise RuntimeError("Gate 1-R configuration status is invalid")
    if comparison.get("data_order_status") not in {"exact", "deterministic_reconstruction"}:
        raise RuntimeError("Gate 1-R data-order status is invalid")
    if comparison.get("curve_shape_match") is not True:
        raise RuntimeError("Gate 1-R requires curve_shape_match=true")
    if comparison.get("reused_as_revision_result") is not False:
        raise RuntimeError("Gate 1-R replay cannot be reused as a revision result")
    range_keys = ("auc500_within_submitted_seed_range", "endpoint_loss_within_submitted_seed_range")
    if mode == "multi_seed_range" and any(comparison.get(key) is not True for key in range_keys):
        raise RuntimeError("Multi-seed Gate 1-R requires both submitted seed-range checks")
    if mode == "single_seed_shape_only" and any(comparison.get(key) is not None for key in range_keys):
        raise RuntimeError("Single-seed Gate 1-R must leave seed-range checks null")
    reviewed_by, notes = spec.get("reviewed_by"), spec.get("notes")
    if not isinstance(reviewed_by, str) or not reviewed_by.strip():
        raise RuntimeError("Gate 1-R spec must identify reviewed_by")
    if not isinstance(notes, str) or not notes.strip():
        raise RuntimeError("Gate 1-R spec must contain comparison notes")
    if not isinstance(replay.get("task"), str) or not replay.get("task", "").strip():
        raise RuntimeError("Gate 1-R replay task must be recorded")
    if isinstance(replay.get("seed"), bool) or not isinstance(replay.get("seed"), int) or replay.get("seed", -1) < 0:
        raise RuntimeError("Gate 1-R replay seed must be a non-negative integer")
    for field in ("declared_adaptations", "limitations"):
        values = spec.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value.strip() for value in values):
            raise RuntimeError(f"Gate 1-R spec must contain non-empty {field}")
    return {
        "schema_version": 2,
        "gate_id": "gate1-r",
        "gate_passed": True,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_status": "reconstructed_without_exact_submitted_config_or_git_history",
        "reconstructed_git_commit": _git_commit(project_root),
        "submitted_manifest_sha256": legacy["manifest_sha256"],
        "submitted_artifacts": legacy["mapped"],
        "replay": {
            "model": replay["model"], "method": replay["method"],
            "task": replay.get("task"), "seed": replay.get("seed"),
            "command": command, "exit_code": 0, "evidence_files": evidence,
        },
        "comparison": {key: comparison.get(key) for key in (
            "mode", "configuration_status", "data_order_status", "curve_shape_match",
            "auc500_within_submitted_seed_range", "endpoint_loss_within_submitted_seed_range",
            "reused_as_revision_result",
        )},
        "declared_adaptations": spec.get("declared_adaptations", []),
        "limitations": spec.get("limitations", []),
        "reviewed_by": reviewed_by.strip(),
        "notes": notes.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output or project_root / "revision_experiments/results/audits/gate1_reproduction.json"
    canonical_output = project_root / "revision_experiments/results/audits/gate1_reproduction.json"
    if output.resolve() != canonical_output.resolve():
        raise SystemExit(f"Gate 1-R evidence must use the canonical path: {canonical_output}")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite Gate 1-R evidence: {output}")
    try:
        spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Invalid Gate 1-R spec: {exc}") from exc
    if not isinstance(spec, dict) or spec.get("schema_version") != 2:
        raise SystemExit("Gate 1-R spec must be a schema_version=2 object")
    try:
        report = build_report(spec, project_root)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    errors, _ = validate_gate1_reproduction(project_root)
    if errors:
        raise SystemExit("Generated Gate 1-R report failed validation: " + "; ".join(errors))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
