#!/usr/bin/env python3
"""Audit local model/data prerequisites and optionally hash large model shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from revision_experiments.scripts.execution_gates import (
    validate_execution_gates,
    validate_model_gate_evidence,
)
from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]


def inspect_file(path: Path, hash_large: bool) -> dict:
    result = {"path": str(path.relative_to(ROOT)), "exists": path.is_file()}
    if path.is_file():
        result["size_bytes"] = path.stat().st_size
        if hash_large or path.stat().st_size < 64 * 1024 * 1024:
            result["sha256"] = file_sha256(path)
    return result


def inspect_hf_checkpoint(path: Path) -> dict:
    index_path = path / "model.safetensors.index.json"
    shard_names = []
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index.get("weight_map", {}).values()))
        except (OSError, json.JSONDecodeError):
            shard_names = []
    elif (path / "model.safetensors").is_file():
        shard_names = ["model.safetensors"]
    required_metadata = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required_metadata + shard_names if not (path / name).is_file()]
    incomplete = sorted(item.name for item in path.rglob("*.incomplete")) if path.is_dir() else []
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.is_dir(),
        "expected_weight_files": len(shard_names),
        "missing_files": missing,
        "incomplete_files": incomplete,
        "ready": path.is_dir() and bool(shard_names) and not missing and not incomplete,
    }


def prometheus_evidence_ready(audits: Path) -> tuple[bool, list[str]]:
    errors = []
    revision = "66ffb1fc20beebfb60a3964a957d9011723116c5"
    paths = {
        "verification": audits / "prometheus_checkpoint_verification.json",
        "load": audits / "prometheus_gpu_load_smoke.json",
        "scoring": audits / "prometheus_scoring_smoke.json",
    }
    payloads = {}
    for label, path in paths.items():
        try:
            payloads[label] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"missing or invalid Prometheus {label} evidence: {exc}")
            payloads[label] = {}
    verification, load, scoring = payloads["verification"], payloads["load"], payloads["scoring"]
    if verification.get("verified") is not True or verification.get("revision") != revision:
        errors.append("Prometheus checkpoint evidence is not verified at the pinned revision")
    expected_hash = file_sha256(paths["verification"]) if paths["verification"].is_file() else None
    if (
        load.get("passed") is not True
        or not str(load.get("checkpoint", "")).endswith(f"@{revision}")
        or load.get("checkpoint_verification_sha256") != expected_hash
    ):
        errors.append("Prometheus GPU load evidence is failed or not checkpoint-bound")
    if (
        scoring.get("passed") is not True
        or scoring.get("judge_revision") != revision
        or scoring.get("checkpoint_verification_sha256") != expected_hash
        or scoring.get("parseable_count") != 3
        or scoring.get("correct_above_incorrect") is not True
    ):
        errors.append("Prometheus scoring evidence is failed, incomplete, or not checkpoint-bound")
    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-large", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "revision_experiments/results/audits/artifact_inventory.json")
    args = parser.parse_args()
    pangu = ROOT / "pretrained_models/openPangu-Embedded-7B-V1.1"
    qwen = ROOT / "pretrained_models/Qwen2.5-7B"
    judge = ROOT / "pretrained_models/prometheus-7b-v2.0"
    required = [
        pangu / "config.json", pangu / "tokenizer_config.json", pangu / "tokenizer.model",
        pangu / "model.safetensors.index.json",
        *[pangu / f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
        qwen / "config.json", qwen / "tokenizer_config.json", qwen / "model.safetensors.index.json",
        *[qwen / f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
        judge / "config.json", judge / "tokenizer_config.json", judge / "model.safetensors.index.json",
        *[judge / f"model-0000{i}-of-00008.safetensors" for i in range(1, 9)],
        ROOT / "pretrained_models/gsm8k/dataset_dict.json",
        ROOT / "pretrained_models/mbpp/mbpp.jsonl",
        ROOT / "pretrained_models/sharegpt_datasets/computer_en_26k.jsonl",
        ROOT / "revision_experiments/data/processed/manifest.json",
        ROOT / "revision_experiments/data/processed/cmmlu/dataset_dict.json",
        ROOT / "revision_experiments/data/processed/mbpp/dataset_dict.json",
        ROOT / "revision_experiments/data/processed/sharegpt_split.json",
        ROOT / "revision_experiments/data/processed/validation_split.json",
        ROOT / "revision_experiments/data/processed/sharegpt_judge_prompts.jsonl",
        ROOT / "revision_experiments/data/processed/sharegpt_judge_prompts.manifest.json",
        ROOT / "revision_experiments/results/audits/model_architecture.json",
        ROOT / "revision_experiments/results/audits/environment_revision.json",
        ROOT / "revision_experiments/results/audits/code_short_test.json",
        ROOT / "revision_experiments/results/audits/initialization_statistics.csv",
        ROOT / "revision_experiments/results/audits/initialization_statistics.parquet",
        ROOT / "revision_experiments/results/audits/initialization_statistics_diagnostic.svg",
        ROOT / "revision_experiments/results/audits/initialization_statistics_diagnostic.png",
        ROOT / "revision_experiments/results/audits/legacy_artifact_discovery.json",
        ROOT / "revision_experiments/results/audits/legacy_import.json",
        ROOT / "revision_experiments/results/audits/legacy_evidence.json",
        ROOT / "revision_experiments/config/smoke/integration_smoke.yaml",
        ROOT / "revision_experiments/results/audits/openpangu_checkpoint_verification.json",
        ROOT / "revision_experiments/results/audits/qwen_checkpoint_verification.json",
        ROOT / "revision_experiments/results/audits/openpangu_gpu_load_smoke.json",
        ROOT / "revision_experiments/results/audits/qwen_gpu_load_smoke.json",
        ROOT / "revision_experiments/results/audits/prometheus_checkpoint_verification.json",
        ROOT / "revision_experiments/results/audits/prometheus_gpu_load_smoke.json",
        ROOT / "revision_experiments/results/audits/prometheus_scoring_smoke.json",
    ]
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_large": args.hash_large,
        "files": [inspect_file(path, args.hash_large) for path in required],
        "directories": {
            "qwen": qwen.is_dir(),
            "prometheus_judge": judge.is_dir(),
            "cmmlu_dev_subjects": len(list((ROOT / "pretrained_models/cmmlu/dev").glob("*.csv"))),
            "cmmlu_test_subjects": len(list((ROOT / "pretrained_models/cmmlu/test").glob("*.csv"))),
        },
        "model_checkpoints": {
            "openpangu": inspect_hf_checkpoint(pangu),
            "qwen": inspect_hf_checkpoint(qwen),
            "prometheus_judge": inspect_hf_checkpoint(judge),
        },
    }
    payload["ready_without_optional_downloads"] = all(item["exists"] for item in payload["files"])
    model_gate_errors, _ = validate_model_gate_evidence(ROOT)
    payload["ready_for_training_gates"] = (
        payload["ready_without_optional_downloads"]
        and payload["model_checkpoints"]["openpangu"]["ready"]
        and payload["model_checkpoints"]["qwen"]["ready"]
        and not model_gate_errors
    )
    judge_ready, judge_errors = prometheus_evidence_ready(ROOT / "revision_experiments/results/audits")
    payload["ready_for_all_evaluations"] = (
        payload["ready_for_training_gates"]
        and payload["model_checkpoints"]["prometheus_judge"]["ready"]
        and judge_ready
    )
    payload["evidence_validation_errors"] = model_gate_errors + judge_errors
    payload["execution_gates"] = validate_execution_gates(ROOT)
    payload["ready_for_formal_training"] = payload["execution_gates"]["ready_for_formal_training"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if payload["ready_for_training_gates"] and payload["ready_for_all_evaluations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
