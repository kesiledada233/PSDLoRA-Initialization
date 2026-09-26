#!/usr/bin/env python3
"""Compare a completed Gate 1-R replay and write the author-review spec."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from revision_experiments.scripts.execution_gates import ROOT
from revision_experiments.scripts.schema import file_sha256


DEFAULT_REPLAY = ROOT / "revision_experiments/results/gates/gate1_reproduction"
DEFAULT_LEGACY_AUDIT = ROOT / "revision_experiments/results/audits/legacy_evidence.json"
DEFAULT_CONFIG = ROOT / "revision_experiments/config/gate1_replay.yaml"


def build_spec(replay_dir: Path, legacy_audit_path: Path, config_path: Path, reviewed_by: str) -> dict:
    if not reviewed_by.strip():
        raise RuntimeError("--reviewed-by must identify the operator")
    if not (replay_dir / "COMPLETED").is_file() or (replay_dir / "FAILED.json").exists():
        raise RuntimeError("Gate 1-R replay is not cleanly completed")
    summary = json.loads((replay_dir / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads(legacy_audit_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_reconstructed_replay" or summary.get("reusable_as_revision_result") is not False:
        raise RuntimeError("invalid Gate 1-R replay summary")
    bindings = {
        "config.yaml": "config_sha256", "raw_loss.jsonl": "raw_loss_sha256",
        "data_order.jsonl": "data_order_sha256",
    }
    for filename, key in bindings.items():
        if summary.get(key) != file_sha256(replay_dir / filename):
            raise RuntimeError(f"Gate 1-R replay hash mismatch: {filename}")
    checkpoint = replay_dir / "checkpoint/adapter_model.safetensors"
    if summary.get("checkpoint_adapter_weights_sha256") != file_sha256(checkpoint):
        raise RuntimeError("Gate 1-R checkpoint hash mismatch")

    ranges = audit["reference_ranges"]
    auc = float(summary["trapezoid_auc_steps_1_to_500"])
    endpoint = float(summary["loss_at_microstep_500"])
    initial = float(summary["first_100_median_loss"])
    auc_pass = ranges["trapezoid_auc_steps_1_to_500"][0] <= auc <= ranges["trapezoid_auc_steps_1_to_500"][1]
    endpoint_pass = ranges["loss_at_microstep_500"][0] <= endpoint <= ranges["loss_at_microstep_500"][1]
    curve_pass = all(math.isfinite(value) for value in (auc, endpoint, initial)) and initial > 10 and endpoint < 1
    if not (auc_pass and endpoint_pass and curve_pass):
        raise RuntimeError(
            "Gate 1-R comparison failed: "
            f"auc_in_range={auc_pass}, endpoint_in_range={endpoint_pass}, curve_shape={curve_pass}"
        )
    command = [
        "python", "revision_experiments/scripts/run_gate1_replay.py",
        "--config", str(config_path.relative_to(ROOT)), "--device", "cuda:0",
    ]
    return {
        "schema_version": 2,
        "reviewed_by": reviewed_by.strip(),
        "notes": (
            "Bounded CUDA reconstruction matched the three-seed legacy AUC500 and step-500 loss ranges; "
            "it is a provenance sanity check and is not reused as a revision result."
        ),
        "replay": {
            "model": "openpangu", "method": "peft_default", "task": config["task"],
            "seed": config["replay_seed"], "command": command, "exit_code": 0,
            "evidence_files": [
                {"kind": "config", "path": str((replay_dir / "config.yaml").relative_to(ROOT))},
                {"kind": "raw_loss", "path": str((replay_dir / "raw_loss.jsonl").relative_to(ROOT))},
                {"kind": "summary", "path": str((replay_dir / "summary.json").relative_to(ROOT))},
                {"kind": "checkpoint", "path": str(checkpoint.relative_to(ROOT))},
            ],
        },
        "comparison": {
            "mode": "multi_seed_range",
            "configuration_status": "reconstructed_from_legacy_evidence",
            "data_order_status": "deterministic_reconstruction",
            "curve_shape_match": curve_pass,
            "auc500_within_submitted_seed_range": auc_pass,
            "endpoint_loss_within_submitted_seed_range": endpoint_pass,
            "reused_as_revision_result": False,
        },
        "declared_adaptations": config["declared_adaptations"],
        "limitations": config["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--legacy-audit", type=Path, default=DEFAULT_LEGACY_AUDIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite Gate 1-R spec: {args.output}")
    try:
        spec = build_spec(
            args.replay_dir.resolve(), args.legacy_audit.resolve(), args.config.resolve(), args.reviewed_by,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
