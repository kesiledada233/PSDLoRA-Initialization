#!/usr/bin/env python3
"""Inventory residual submitted-era artifacts without treating derived plots as raw evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]
DERIVED_CANDIDATES = {
    "fusion_result.json": "empty_or_derived_fusion_summary",
    "time_to_threshold_results.json": "derived_threshold_summary",
    "time_to_threshold_speedup_by_threshold.csv": "derived_threshold_table",
    "time_to_threshold_line_comparison.png": "derived_threshold_figure",
    "fda_enhanced_overview.png": "derived_overview_figure",
    "method_overview_assets.pdf": "method_figure_asset",
    "method_overview_assets.png": "method_figure_asset",
}
SUBMITTED_ENTRYPOINTS = (
    "train_openpangu_fda_lora_final.py",
    "train_qwen2.5_fda_lora_final.py",
    "train_sharegpt_final.py",
    "evaluate_downstream.py",
    "fdt_init.py",
)
CONFIG_NAMES = {"config.json", "config.yaml", "config.yml", "training_args.json", "run_config.json"}
CHECKPOINT_NAMES = {"adapter_model.safetensors", "adapter_model.bin"}


def _record(path: Path, root: Path, role: str) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _output_dirs(root: Path) -> list[Path]:
    direct = [path for path in root.glob("outputs_*") if path.is_dir()]
    imported = root / "legacy_artifacts/submission"
    nested = [path for path in imported.rglob("outputs_*") if path.is_dir()] if imported.is_dir() else []
    return sorted({*direct, *nested})


def _raw_files(root: Path, output_dirs: list[Path]) -> dict[str, list[dict]]:
    result = {"configs": [], "raw_logs": [], "checkpoints": []}
    for directory in output_dirs:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.name in CONFIG_NAMES:
                result["configs"].append(_record(path, root, "runtime_config"))
            if path.name == "training_log.csv" or "raw_loss" in path.name:
                result["raw_logs"].append(_record(path, root, "raw_training_log"))
            if path.name in CHECKPOINT_NAMES or path.suffix in {".ckpt", ".pt", ".pth"}:
                result["checkpoints"].append(_record(path, root, "adapter_or_training_checkpoint"))
    return result


def _coherent_bundles(root: Path, output_dirs: list[Path]) -> list[dict]:
    bundles = []
    for output_dir in output_dirs:
        configs = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name in CONFIG_NAMES)
        for config in configs:
            run_dir = config.parent
            logs = sorted(path for path in run_dir.rglob("*") if path.is_file() and (
                path.name == "training_log.csv" or "raw_loss" in path.name
            ))
            checkpoints = sorted(path for path in run_dir.rglob("*") if path.is_file() and (
                path.name in CHECKPOINT_NAMES or path.suffix in {".ckpt", ".pt", ".pth"}
            ))
            bundles.append({
                "run_directory": str(run_dir.relative_to(root)),
                "config": _record(config, root, "runtime_config"),
                "raw_logs": [_record(path, root, "raw_training_log") for path in logs],
                "checkpoints": [_record(path, root, "adapter_or_training_checkpoint") for path in checkpoints],
                "complete": bool(logs and checkpoints),
            })
    return bundles


def build_discovery(root: str | Path = ROOT) -> dict:
    root = Path(root).resolve()
    output_dirs = _output_dirs(root)
    raw = _raw_files(root, output_dirs)
    bundles = _coherent_bundles(root, output_dirs)
    derived = [
        _record(root / name, root, role)
        for name, role in DERIVED_CANDIDATES.items() if (root / name).is_file()
    ]
    entrypoints = [
        _record(root / name, root, "submitted_source_entrypoint")
        for name in SUBMITTED_ENTRYPOINTS if (root / name).is_file()
    ]
    required = {
        "exact_runtime_config": bool(raw["configs"]),
        "raw_per_step_loss_log": bool(raw["raw_logs"]),
        "corresponding_adapter_checkpoint": bool(raw["checkpoints"]),
    }
    sufficient = any(bundle["complete"] for bundle in bundles)
    reasons = []
    if derived and not raw["raw_logs"]:
        reasons.append("derived summaries/figures exist, but their source per-step training logs are absent")
    if not output_dirs:
        reasons.append("no outputs_* submitted-run directory exists in the workspace")
    if not raw["configs"]:
        reasons.append("no exact runtime config was found in a submitted-run output directory")
    if not raw["checkpoints"]:
        reasons.append("no corresponding adapter/training checkpoint was found outside pretrained models")
    if all(required.values()) and not sufficient:
        reasons.append("the three raw artifact categories were not found under one coherent run directory")
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_status": "reconstructed_without_submitted_git_history",
        "submitted_source_entrypoints": entrypoints,
        "derived_candidates": derived,
        "discovered_output_directories": [str(path.relative_to(root)) for path in output_dirs],
        "raw_run_artifacts": raw,
        "coherent_gate1_candidates": bundles,
        "gate1_required_categories_found": required,
        "sufficient_for_gate1_reproduction": sufficient,
        "reasons": reasons,
        "ruling": (
            "Candidate files may be preserved as historical derived outputs but cannot be used as "
            "raw Gate 1-R evidence without the source config, per-step log, and corresponding checkpoint."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "revision_experiments/results/audits/legacy_artifact_discovery.json",
    )
    args = parser.parse_args()
    payload = build_discovery(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["sufficient_for_gate1_reproduction"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
