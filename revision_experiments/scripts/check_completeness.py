#!/usr/bin/env python3
"""Check configs, expected inventories, and terminal run schemas."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from revision_experiments.scripts.aggregate_results import (
    CANONICAL_FIGURE_STEMS,
    CANONICAL_TABLE_NAMES,
    INVENTORY_COLUMNS,
    _generate_figures,
    build_run_inventory,
    build_publication_tables,
    discover_expected_runs,
)
from revision_experiments.scripts.execution_gates import validate_execution_gates
from revision_experiments.scripts.matrix import expand_matrix, load_matrix
from revision_experiments.scripts.schema import file_sha256, validate_run_directory


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "revision_experiments" / "config"
RUNS_DIR = ROOT / "revision_experiments" / "results" / "runs"
EVALUATIONS_DIR = ROOT / "revision_experiments" / "results" / "evaluations"
AGGREGATE_DIR = ROOT / "revision_experiments" / "results" / "aggregate"
FIGURES_DIR = ROOT / "revision_experiments" / "results" / "figures"
PLACEHOLDER_PREFIXES = ("detected_", "exact_", "concrete/")


def find_placeholders(value, path="root") -> list[str]:
    errors = []
    if isinstance(value, dict):
        for key, child in value.items():
            errors.extend(find_placeholders(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(find_placeholders(child, f"{path}[{index}]"))
    elif isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIXES):
        errors.append(f"placeholder at {path}: {value}")
    return errors


def _resolve_manifest_path(label: object, project_root: Path) -> Path | None:
    if not isinstance(label, str) or not label.strip():
        return None
    path = Path(label)
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    return resolved if resolved.is_relative_to(project_root.resolve()) else None


def validate_publication_manifest(
    manifest_path: str | Path,
    inventory: pd.DataFrame,
    project_root: str | Path = ROOT,
    *,
    expected_inventory_sha: str,
) -> list[str]:
    """Validate that every publication input/output is tracked and immutable."""
    manifest_path, project_root = Path(manifest_path), Path(project_root)
    errors = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid publication manifest {manifest_path}: {exc}"]
    if not isinstance(manifest, dict):
        return [f"invalid publication manifest {manifest_path}: expected JSON object"]
    if manifest.get("experimental_unit") != "run/seed":
        errors.append("publication manifest experimental_unit must be run/seed")
    join_keys = manifest.get("paired_join_keys")
    required_join_keys = {"analysis_context", "model", "task", "seed", "checkpoint/endpoint", "metric"}
    if not isinstance(join_keys, list) or set(join_keys) != required_join_keys:
        errors.append("publication manifest has incomplete paired join keys")
    if manifest.get("inventory_sha256") != expected_inventory_sha:
        errors.append("publication manifest inventory hash mismatch")
    aggregation_script = Path(__file__).with_name("aggregate_results.py")
    if manifest.get("aggregation_script_sha256") != file_sha256(aggregation_script):
        errors.append("publication manifest aggregation script hash mismatch")
    complete_ids = set(inventory.loc[inventory["status"] == "complete", "run_id"].astype(str))
    for section in ("source_artifacts", "output_artifacts"):
        records = manifest.get(section)
        if not isinstance(records, list) or not records:
            errors.append(f"publication manifest {section} must be a non-empty list")
            continue
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"publication manifest {section}[{index}] is not an object")
                continue
            path = _resolve_manifest_path(record.get("path"), project_root)
            if path is None or not path.is_file():
                errors.append(f"publication artifact missing: {record.get('path')}")
            elif file_sha256(path) != record.get("sha256"):
                errors.append(f"publication artifact hash mismatch: {record.get('path')}")
            run_id = record.get("run_id")
            if section == "source_artifacts" and run_id is not None and str(run_id) not in complete_ids:
                errors.append(f"publication source references untracked run: {run_id}")
    return errors


def _scientific_outputs(aggregate_dir: Path, figures_dir: Path) -> list[Path]:
    tables = [
        path for path in aggregate_dir.glob("*.csv")
        if path.name.startswith("reviewer_")
        or path.name.endswith("_table.csv")
        or path.name == "baseline_search_supplementary_table.csv"
    ]
    figures = [
        path for path in figures_dir.glob("*")
        if path.suffix in {".svg", ".pdf", ".png", ".json"}
    ]
    return sorted(tables + figures)


def validate_reviewer_tables(aggregate_dir: str | Path, inventory: pd.DataFrame) -> list[str]:
    """Enforce seed/run as n and exact three-seed reviewer table groups."""
    aggregate_dir = Path(aggregate_dir)
    errors = []
    missing_canonical = sorted(name for name in CANONICAL_TABLE_NAMES if not (aggregate_dir / name).is_file())
    if missing_canonical:
        errors.append(f"missing canonical publication tables: {missing_canonical}")
    seed_path = aggregate_dir / "reviewer_seed_metrics.csv"
    if not seed_path.is_file():
        return [f"missing reviewer seed table: {seed_path}"]
    try:
        seed_table = pd.read_csv(seed_path)
    except Exception as exc:
        return [f"invalid reviewer seed table: {exc}"]
    identity = ["analysis_context", "model", "task", "method", "seed", "endpoint", "metric"]
    required = set(identity + ["run_id", "metadata_sha256"])
    missing = sorted(required - set(seed_table.columns))
    if missing:
        return [f"reviewer seed table missing columns: {missing}"]
    if seed_table.duplicated(identity).any():
        errors.append("reviewer seed table duplicates an independent run/seed observation")
    group_keys = ["analysis_context", "model", "task", "method", "endpoint", "metric"]
    for keys, group in seed_table.groupby(group_keys, dropna=False):
        observed_seeds = set(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int))
        if observed_seeds != {42, 123, 1107}:
            errors.append(f"reviewer seed group must contain exactly three seeds for {keys}: {sorted(observed_seeds)}")
    complete_ids = set(inventory.loc[inventory["status"] == "complete", "run_id"].astype(str))
    untracked = sorted(set(seed_table["run_id"].astype(str)) - complete_ids)
    if untracked:
        errors.append(f"reviewer seed table references untracked run IDs: {untracked[:5]}")
    for name, count_column in (
        ("reviewer_summary_metrics.csv", "n_runs"),
        ("reviewer_paired_differences.csv", "n_runs"),
        ("reviewer_paired_difference_summary.csv", "n_runs"),
    ):
        path = aggregate_dir / name
        if not path.is_file():
            errors.append(f"missing reviewer table: {path}")
            continue
        try:
            frame = pd.read_csv(path)
            counts = pd.to_numeric(frame[count_column], errors="coerce")
            if frame.empty or not counts.eq(3).all():
                errors.append(f"{name} must report n_runs=3 from independent seeds")
        except Exception as exc:
            errors.append(f"invalid reviewer table {path}: {exc}")
    return errors


def validate_aggregation_artifacts(
    config_dir: str | Path,
    runs_dir: str | Path,
    evaluations_dir: str | Path,
    aggregate_dir: str | Path,
    figures_dir: str | Path,
    project_root: str | Path = ROOT,
    *,
    require_publication: bool,
    audits_dir: str | Path | None = None,
) -> list[str]:
    """Recompute inventory and validate publication provenance, if present."""
    config_dir, runs_dir, evaluations_dir = Path(config_dir), Path(runs_dir), Path(evaluations_dir)
    aggregate_dir, figures_dir, project_root = Path(aggregate_dir), Path(figures_dir), Path(project_root)
    errors = []
    try:
        expected = discover_expected_runs(
            config_dir, selected_path=aggregate_dir / "baseline_selected_configs.yaml"
        )
        recomputed = build_run_inventory(expected, runs_dir, evaluations_dir, project_root=project_root)
    except Exception as exc:
        return [f"could not recompute aggregation inventory: {exc}"]
    inventory_path = aggregate_dir / "run_inventory.csv"
    status_path = aggregate_dir / "aggregation_status.json"
    if not inventory_path.is_file():
        errors.append(f"missing aggregation inventory: {inventory_path}")
        return errors
    if inventory_path.read_text(encoding="utf-8") != recomputed.to_csv(index=False):
        errors.append("run_inventory.csv is not reproducible from current matrices and raw artifacts")
    inventory_sha = file_sha256(inventory_path)
    try:
        observed = pd.read_csv(inventory_path)
        missing_columns = sorted(set(INVENTORY_COLUMNS) - set(observed.columns))
        if missing_columns:
            errors.append(f"run inventory missing columns: {missing_columns}")
        if observed["run_id"].duplicated().any():
            errors.append("run inventory contains duplicate run IDs")
    except Exception as exc:
        errors.append(f"invalid run_inventory.csv: {exc}")
        observed = recomputed
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid aggregation status {status_path}: {exc}")
        status = {}
    if status.get("inventory_sha256") != inventory_sha:
        errors.append("aggregation status inventory hash mismatch")
    current_matrix_hashes = {
        path.name: file_sha256(path) for path in sorted(config_dir.glob("*_matrix.yaml"))
    }
    if status.get("matrix_sha256") != current_matrix_hashes:
        errors.append("aggregation status matrix hashes do not match current matrices")
    aggregation_script = Path(__file__).with_name("aggregate_results.py")
    if status.get("aggregation_script_sha256") != file_sha256(aggregation_script):
        errors.append("aggregation status script hash does not match current aggregation command")
    if status.get("aggregation_command") != "python revision_experiments/scripts/aggregate_results.py":
        errors.append("aggregation status does not record the reproducible repository command")
    expected_arguments = {
        "config_dir": str(config_dir.resolve()), "runs_root": str(runs_dir.resolve()),
        "evaluations_root": str(evaluations_dir.resolve()), "output_root": str(aggregate_dir.resolve()),
        "figures_root": str(figures_dir.resolve()), "project_root": str(project_root.resolve()),
    }
    recorded_arguments = status.get("aggregation_arguments")
    if not isinstance(recorded_arguments, dict) or any(
        recorded_arguments.get(key) != value for key, value in expected_arguments.items()
    ):
        errors.append("aggregation status arguments do not match the checked CLI paths")

    scientific_outputs = _scientific_outputs(aggregate_dir, figures_dir)
    manifest_path = aggregate_dir / "reviewer_tables_manifest.json"
    if scientific_outputs and not manifest_path.is_file():
        errors.append("scientific tables/figures exist without reviewer_tables_manifest.json")
    if manifest_path.is_file():
        errors.extend(validate_publication_manifest(
            manifest_path, observed, project_root, expected_inventory_sha=inventory_sha
        ))
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("aggregation_arguments") != recorded_arguments:
                errors.append("publication manifest aggregation arguments differ from aggregation status")
            tracked = {
                str((_resolve_manifest_path(record.get("path"), project_root) or Path()).resolve())
                for record in manifest.get("output_artifacts", []) if isinstance(record, dict)
            }
            for path in scientific_outputs:
                if str(path.resolve()) not in tracked:
                    errors.append(f"scientific output is not tracked by publication manifest: {path}")
        except (OSError, json.JSONDecodeError):
            pass
    generated = status.get("reviewer_artifacts_generated") is True
    if scientific_outputs and not generated:
        errors.append("aggregation status denies reviewer artifacts although scientific outputs exist")
    if generated and not manifest_path.is_file():
        errors.append("aggregation status claims reviewer artifacts without a publication manifest")
    if generated:
        errors.extend(validate_reviewer_tables(aggregate_dir, observed))
        expected_science_names = set(CANONICAL_TABLE_NAMES) | {
            f"{stem}.{extension}" for stem in CANONICAL_FIGURE_STEMS
            for extension in ("svg", "pdf", "png", "provenance.json")
        }
        observed_science_names = {path.name for path in scientific_outputs}
        if observed_science_names != expected_science_names:
            errors.append(
                "publication does not contain the exact canonical table/figure set: "
                f"missing={sorted(expected_science_names - observed_science_names)}, "
                f"extra={sorted(observed_science_names - expected_science_names)}"
            )
        try:
            resolved_audits = Path(audits_dir) if audits_dir is not None else project_root / "revision_experiments/results/audits"
            tables, seed_metrics, mechanism = build_publication_tables(
                expected, recomputed, runs_dir, evaluations_dir, resolved_audits,
                aggregate_dir, project_root, config_dir,
            )
            for name, frame in tables.items():
                path = aggregate_dir / name
                if path.is_file() and path.read_text(encoding="utf-8") != frame.to_csv(index=False):
                    errors.append(f"scientific table is not reproducible from raw artifacts: {name}")
            with tempfile.TemporaryDirectory(prefix="task9-recompute-") as temporary:
                regenerated = _generate_figures(seed_metrics, mechanism, Path(temporary))
                for path in regenerated:
                    observed_path = figures_dir / path.name
                    if observed_path.is_file() and file_sha256(path) != file_sha256(observed_path):
                        errors.append(f"scientific figure is not reproducible from raw artifacts: {path.name}")
        except Exception as exc:
            errors.append(f"could not independently regenerate publication outputs: {exc}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_labels = {record.get("path") for record in manifest.get("source_artifacts", [])}
            expected_checkpoint_trees = []
            for item in expected:
                run_base = runs_dir / item["run_id"]
                required_sources = []
                if "baseline_fairness_screening" in item.get("analysis_contexts", ()):
                    required_sources.append(run_base / "validation_loss.jsonl")
                for checkpoint in item["expected_checkpoints"]:
                    checkpoint_dir = run_base / "checkpoints" / f"step_{checkpoint:06d}"
                    required_sources.extend(
                        path for path in checkpoint_dir.rglob("*")
                        if path.is_file()
                    )
                    tree_hashes = json.loads(recomputed.set_index("run_id").loc[item["run_id"], "checkpoint_sha256"])
                    expected_checkpoint_trees.append({
                        "run_id": item["run_id"], "checkpoint": int(checkpoint),
                        "tree_sha256": tree_hashes[str(checkpoint)],
                    })
                for path in required_sources:
                    label = str(path.resolve().relative_to(project_root.resolve()))
                    if label not in source_labels:
                        errors.append(f"publication manifest omits required raw/checkpoint source: {label}")
            if manifest.get("checkpoint_trees") != expected_checkpoint_trees:
                errors.append("publication manifest checkpoint-tree hashes do not match recomputed checkpoints")
        except Exception as exc:
            errors.append(f"could not validate complete publication source set: {exc}")
    if require_publication:
        incomplete = recomputed[recomputed["status"] != "complete"]
        if not incomplete.empty:
            errors.append(f"publication requires all expected runs complete; {len(incomplete)} are not complete")
        if not generated:
            errors.append("required reviewer tables/figures were not generated")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete-runs", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "revision_experiments/results/aggregate/completeness.json")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--evaluations-dir", type=Path, default=EVALUATIONS_DIR)
    parser.add_argument("--aggregate-dir", type=Path, default=AGGREGATE_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    errors = []
    repository_map = yaml.safe_load((args.config_dir / "local_repository_map.yaml").read_text(encoding="utf-8"))
    errors.extend(find_placeholders(repository_map))
    matrix_summary = {}
    for path in sorted(args.config_dir.glob("*_matrix.yaml")):
        config = load_matrix(path)
        try:
            runs = expand_matrix(config)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        complete = 0
        failed = 0
        invalid = 0
        for run in runs:
            run_dir = args.runs_dir / run["run_id"]
            if (run_dir / "COMPLETED").is_file():
                complete += 1
                run_errors = validate_run_directory(run_dir)
                if run_errors:
                    invalid += 1
                    errors.extend(f"{run['run_id']}: {error}" for error in run_errors)
            elif (run_dir / "FAILED.json").is_file():
                failed += 1
        matrix_summary[config["matrix_name"]] = {
            "expected": len(runs), "complete": complete, "failed": failed,
            "missing": len(runs) - complete - failed, "invalid": invalid,
            "enabled": bool(config.get("enabled", False)),
        }
        if args.require_complete_runs and complete != len(runs):
            errors.append(f"{config['matrix_name']}: only {complete}/{len(runs)} complete")
    enabled_matrices = [name for name, summary in matrix_summary.items() if summary["enabled"]]
    if enabled_matrices or args.require_complete_runs:
        gate_status = validate_execution_gates(ROOT)
        if not gate_status["ready_for_formal_training"]:
            errors.append(
                "formal matrices require verified Gate 1-R and Gate 2 evidence: "
                + "; ".join(gate_status["errors"])
            )
    errors.extend(validate_aggregation_artifacts(
        args.config_dir, args.runs_dir, args.evaluations_dir, args.aggregate_dir,
        args.figures_dir, ROOT, require_publication=args.require_complete_runs,
        audits_dir=ROOT / "revision_experiments/results/audits",
    ))
    payload = {"ok": not errors, "matrices": matrix_summary, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
