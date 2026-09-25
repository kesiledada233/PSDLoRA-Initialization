#!/usr/bin/env python3
"""Verify and copy the minimal coherent legacy baseline bundle.

The recovery directory is treated as immutable input.  This command copies only
the three-seed openPangu/GSM8K PEFT-default evidence needed by Gate 1-R and
builds a fresh canonical manifest under ``legacy_artifacts/submission``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECOVERY = ROOT / "FDA_INIT_legacy_recovery_20260904_1816"
DESTINATION = ROOT / "legacy_artifacts/submission"
AUDIT_PATH = ROOT / "revision_experiments/results/audits/legacy_import.json"


def selected_files() -> list[dict[str, str]]:
    """Return the deliberately small, coherent import profile."""
    records = [{
        "category": "configs",
        "source": "legacy_recovery/candidates/source_code/FDT_Init/train_openpangu_fda_lora_final.py",
        "destination": "FDT_Init/train_openpangu_fda_lora_final.py",
        "role": "legacy_training_entrypoint_with_runtime_defaults",
    }]
    bundles = {
        1107: ("baseline", "FDT_Init__outputs_gsm8k__baseline--fc733e5d3f"),
        123: ("baseline_seed123", "FDT_Init__outputs_gsm8k__baseline_seed123--92d16c983a"),
        42: ("baseline_seed42", "FDT_Init__outputs_gsm8k__baseline_seed42--36aa4ec4bd"),
    }
    for seed, (run_name, submission_bundle) in bundles.items():
        run_root = f"FDT_Init/outputs_gsm8k/{run_name}"
        records.extend([
            {
                "category": "configs",
                "source": f"legacy_recovery/candidates/derived_outputs/{run_root}/results.json",
                "destination": f"{run_root}/results.json",
                "role": f"legacy_run_summary_and_partial_runtime_configuration_seed_{seed}",
            },
            {
                "category": "logs",
                "source": f"legacy_artifacts/submission/{submission_bundle}/logs/training_log.csv",
                "destination": f"{run_root}/training_log.csv",
                "role": f"raw_per_microbatch_loss_seed_{seed}",
            },
            {
                "category": "checkpoints",
                "source": f"legacy_artifacts/submission/{submission_bundle}/checkpoints/final_model/adapter_config.json",
                "destination": f"{run_root}/final_model/adapter_config.json",
                "role": f"peft_adapter_configuration_seed_{seed}",
            },
            {
                "category": "checkpoints",
                "source": f"legacy_artifacts/submission/{submission_bundle}/checkpoints/final_model/adapter_model.safetensors",
                "destination": f"{run_root}/final_model/adapter_model.safetensors",
                "role": f"peft_adapter_weights_seed_{seed}",
            },
        ])
    return records


def recovery_hashes(recovery_root: Path) -> dict[str, str]:
    """Load hashes independently recorded for submission and candidate files."""
    hashes: dict[str, str] = {}
    manifest = recovery_root / "legacy_artifacts/submission/SHA256SUMS"
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"invalid recovery SHA256SUMS line {line_number}")
        hashes[f"legacy_artifacts/submission/{parts[1]}"] = parts[0]
    inventory = json.loads((recovery_root / "legacy_recovery/inventory.json").read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise RuntimeError("recovery inventory must be a JSON list")
    for row in inventory:
        if isinstance(row, dict) and isinstance(row.get("package_relative_path"), str):
            hashes[row["package_relative_path"].replace("\\", "/")] = row.get("sha256")
    return hashes


def inspect_recovery(recovery_root: Path) -> list[dict]:
    verification = json.loads((recovery_root / "legacy_recovery/COPY_VERIFICATION.json").read_text(encoding="utf-8"))
    if verification.get("checks_passed") is not True:
        raise RuntimeError("recovery package does not declare checks_passed=true")
    expected = recovery_hashes(recovery_root)
    inspected = []
    for record in selected_files():
        source = (recovery_root / record["source"]).resolve()
        if not source.is_relative_to(recovery_root.resolve()) or not source.is_file():
            raise RuntimeError(f"missing or unsafe recovery source: {record['source']}")
        digest = file_sha256(source)
        if expected.get(record["source"]) != digest:
            raise RuntimeError(f"recovery source hash mismatch or absent from inventory: {record['source']}")
        inspected.append({**record, "size_bytes": source.stat().st_size, "sha256": digest})
    return inspected


def write_manifest(destination: Path) -> Path:
    manifest = destination / "SHA256SUMS"
    files = sorted(path for path in destination.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(destination).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return manifest


def ingest(recovery_root: Path, apply: bool) -> dict:
    inspected = inspect_recovery(recovery_root)
    if apply:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        unexpected = [path for path in DESTINATION.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
        expected_destinations = {record["destination"] for record in inspected}
        unexpected = [path for path in unexpected if path.relative_to(DESTINATION).as_posix() not in expected_destinations]
        if unexpected:
            raise RuntimeError(f"canonical import contains unexpected files; refusing to merge: {unexpected[0]}")
        for record in inspected:
            source = recovery_root / record["source"]
            target = DESTINATION / record["destination"]
            if target.exists() and file_sha256(target) != record["sha256"]:
                raise RuntimeError(f"refusing to overwrite different canonical artifact: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            if file_sha256(target) != record["sha256"]:
                raise RuntimeError(f"post-copy hash mismatch: {target}")
        manifest = write_manifest(DESTINATION)
    else:
        manifest = DESTINATION / "SHA256SUMS"
    payload = {
        "schema_version": 1,
        "profile": "openpangu_gsm8k_peft_default_three_seed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_root": str(recovery_root.relative_to(ROOT)),
        "recovery_report_sha256": file_sha256(recovery_root / "legacy_recovery/RECOVERY_REPORT.md"),
        "copy_verification_sha256": file_sha256(recovery_root / "legacy_recovery/COPY_VERIFICATION.json"),
        "source_package_manifest_sha256": file_sha256(recovery_root / "legacy_artifacts/submission/SHA256SUMS"),
        "selected_file_count": len(inspected),
        "selected_files": inspected,
        "canonical_manifest": str(manifest.relative_to(ROOT)),
        "canonical_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
        "source_files_unchanged": True,
        "configuration_status": "reconstructed_from_legacy_entrypoint_defaults_run_summaries_logs_and_adapter_configs",
        "limitations": [
            "The original per-run config.json files were not copied by the local recovery tool.",
            "The original command line and submitted Git/environment snapshots remain unavailable.",
            "Imported legacy runs are provenance/comparison evidence only and cannot be reused as revision results.",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--apply", action="store_true", help="copy the verified profile into the canonical import root")
    parser.add_argument("--output", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    recovery_root = args.recovery_root.resolve()
    if not recovery_root.is_relative_to(ROOT):
        raise SystemExit("recovery root must be inside the project")
    try:
        payload = ingest(recovery_root, args.apply)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.apply:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            stable_existing = {key: value for key, value in existing.items() if key != "captured_at_utc"}
            stable_payload = {key: value for key, value in payload.items() if key != "captured_at_utc"}
            if stable_existing != stable_payload:
                raise SystemExit(f"refusing to overwrite different legacy import audit: {args.output}")
            payload = existing
        else:
            args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
