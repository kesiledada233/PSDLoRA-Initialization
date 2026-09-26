#!/usr/bin/env python3
"""Audit the recovered Gate 1-R baseline and explain the legacy high loss."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml
from safetensors import safe_open

from revision_experiments.scripts.execution_gates import ROOT, validate_legacy_artifacts
from revision_experiments.scripts.schema import file_sha256


DEFAULT_CONFIG = ROOT / "revision_experiments/config/gate1_replay.yaml"
DEFAULT_OUTPUT = ROOT / "revision_experiments/results/audits/legacy_evidence.json"


def argparse_defaults(source: str) -> dict[str, object]:
    """Extract literal argparse defaults without importing legacy code."""
    tree = ast.parse(source)
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            flag = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        keyword = next((item for item in node.keywords if item.arg == "default"), None)
        if keyword is not None:
            try:
                defaults[flag[2:]] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
        elif any(isinstance(item, ast.keyword) and item.arg == "action" for item in node.keywords):
            action = next(item for item in node.keywords if item.arg == "action")
            try:
                if ast.literal_eval(action.value) == "store_true":
                    defaults[flag[2:]] = False
            except (ValueError, TypeError):
                pass
    return defaults


def legacy_source_contract(source: str) -> dict[str, bool]:
    compact = "".join(source.split())
    attach_index = compact.find("model=get_peft_model(model,lora_config)")
    move_index = compact.find("model=model.to(device)", attach_index)
    return {
        "labels_clone_input_ids": "'labels':item['input_ids'].clone()" in compact,
        "padding_labels_masked": "labels[attention_mask==0]=-100" in compact,
        "pad_token_set_to_eos_when_missing": "tokenizer.pad_token=tokenizer.eos_token" in compact,
        "gsm8k_chinese_text_template": (
            'returnf"问题：{example[\'question\']}\\n解答：{example[\'answer\']}"' in compact
        ),
        "peft_attached_on_cpu_before_accelerator_move": (
            attach_index >= 0 and move_index > attach_index
        ),
        "reported_loss_restores_accumulation_divisor": "current_loss=loss.item()*args.grad_accum_steps" in compact,
        "optimizer_updates_on_accumulation_boundary": "ifstep%args.grad_accum_steps==0:" in compact,
    }


def loss_summary(path: Path, expected_steps: int = 2500) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["train_loss"]) for row in rows]
    if steps != list(range(1, expected_steps + 1)):
        raise RuntimeError(f"legacy log is not contiguous 1..{expected_steps}: {path}")
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"legacy log contains non-finite losses: {path}")
    first500 = losses[:500]
    return {
        "rows": len(rows),
        "step_min": steps[0],
        "step_max": steps[-1],
        "first_100_median_loss": statistics.median(losses[:100]),
        "legacy_sum_auc500": sum(first500),
        "trapezoid_auc_steps_1_to_500": sum(
            (first500[index - 1] + first500[index]) / 2 for index in range(1, len(first500))
        ),
        "loss_at_microstep_500": first500[-1],
    }


def adapter_tensor_contract(
    path: Path, rank: int, hidden_size: int = 4096, value_projection_size: int = 1024, layers: int = 34,
) -> dict:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    a_keys = [key for key in keys if key.endswith("lora_A.weight")]
    b_keys = [key for key in keys if key.endswith("lora_B.weight")]
    if len(keys) != layers * 4 or len(a_keys) != layers * 2 or len(b_keys) != layers * 2:
        raise RuntimeError(f"legacy adapter tensor coverage is invalid: {path}")
    if any(shape != [rank, hidden_size] for key, shape in shapes.items() if key in a_keys):
        raise RuntimeError(f"legacy adapter A tensor shape is invalid: {path}")
    for key in b_keys:
        expected = [value_projection_size, rank] if ".v_proj." in key else [hidden_size, rank]
        if shapes[key] != expected:
            raise RuntimeError(f"legacy adapter B tensor shape is invalid: {path}: {key}")
    modules = sorted({"q_proj" if ".q_proj." in key else "v_proj" for key in keys})
    return {
        "tensor_count": len(keys), "lora_a_tensor_count": len(a_keys),
        "lora_b_tensor_count": len(b_keys), "transformer_layer_count": layers,
        "target_modules": modules, "rank": rank, "hidden_size": hidden_size,
        "value_projection_size": value_projection_size,
    }


def audit(config_path: Path, root: Path = ROOT) -> dict:
    errors, legacy = validate_legacy_artifacts(root)
    if errors:
        raise RuntimeError("legacy artifact validation failed: " + "; ".join(errors))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entrypoint = root / config["references"]["entrypoint"]
    source = entrypoint.read_text(encoding="utf-8")
    defaults = argparse_defaults(source)
    expected_defaults = {
        "model_path": config["model"]["submitted_path"],
        "lora_r": config["adapter"]["rank"],
        "lora_alpha": config["adapter"]["alpha"],
        "lora_dropout": config["adapter"]["dropout"],
        "lora_target_modules": config["adapter"]["target_modules"],
        "batch_size": config["training"]["batch_size"],
        "max_length": config["training"]["max_length"],
        "max_iters": config["training"]["submitted_microsteps"],
        "grad_accum_steps": config["training"]["gradient_accumulation_microsteps"],
        "lr": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_steps": config["training"]["warmup_optimizer_steps"],
        "max_grad_norm": config["training"]["max_grad_norm"],
        "seed": config["replay_seed"],
    }
    mismatches = {
        key: {"expected": value, "observed": defaults.get(key)}
        for key, value in expected_defaults.items() if defaults.get(key) != value
    }
    source_contract = legacy_source_contract(source)
    if mismatches:
        raise RuntimeError(f"reconstructed config differs from legacy source defaults: {mismatches}")
    if not source_contract["labels_clone_input_ids"] or source_contract["padding_labels_masked"]:
        raise RuntimeError("legacy source no longer demonstrates the unmasked-padding label policy")
    if not source_contract["gsm8k_chinese_text_template"]:
        raise RuntimeError("legacy source no longer matches the reconstructed GSM8K text template")
    if not source_contract["peft_attached_on_cpu_before_accelerator_move"]:
        raise RuntimeError("legacy source no longer matches the reconstructed PEFT attachment order")
    if config["training"].get("text_format") != "submitted_chinese_task_template":
        raise RuntimeError("Gate 1-R config does not declare the submitted text template")
    if config["training"].get("peft_attachment_order") != (
        "base_model_on_cpu_then_move_augmented_model_to_accelerator"
    ):
        raise RuntimeError("Gate 1-R config does not declare the submitted PEFT attachment order")

    runs = []
    for reference in config["references"]["runs"]:
        summary_path = root / reference["summary"]
        log_path = root / reference["raw_log"]
        checkpoint_dir = root / reference["checkpoint"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        adapter = json.loads((checkpoint_dir / "adapter_config.json").read_text(encoding="utf-8"))
        tensor_contract = adapter_tensor_contract(
            checkpoint_dir / "adapter_model.safetensors", int(config["adapter"]["rank"]),
        )
        metrics = loss_summary(log_path, config["training"]["submitted_microsteps"])
        expected_summary = {
            "dataset": config["task"], "model_path": config["model"]["submitted_path"],
            "lora_r": config["adapter"]["rank"], "lora_alpha": config["adapter"]["alpha"],
            "init_method": "PEFT_Default", "init_preset": "baseline",
        }
        if any(summary.get(key) != value for key, value in expected_summary.items()):
            raise RuntimeError(f"legacy result summary configuration mismatch: {summary_path}")
        expected_adapter = {
            "r": config["adapter"]["rank"], "lora_alpha": config["adapter"]["alpha"],
            "lora_dropout": config["adapter"]["dropout"], "peft_type": "LORA",
            "init_lora_weights": True, "use_dora": False,
        }
        if any(adapter.get(key) != value for key, value in expected_adapter.items()):
            raise RuntimeError(f"legacy adapter configuration mismatch: {checkpoint_dir}")
        if set(adapter.get("target_modules", [])) != set(config["adapter"]["target_modules"]):
            raise RuntimeError(f"legacy target modules mismatch: {checkpoint_dir}")
        recorded_auc = summary.get("auc_metrics", {}).get("auc_0_500")
        if not math.isclose(float(recorded_auc), metrics["legacy_sum_auc500"], rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"legacy summary AUC does not match raw loss: {summary_path}")
        runs.append({
            "seed": reference["seed"],
            "summary": reference["summary"], "summary_sha256": file_sha256(summary_path),
            "raw_log": reference["raw_log"], "raw_log_sha256": file_sha256(log_path),
            "checkpoint": reference["checkpoint"],
            "adapter_config_sha256": file_sha256(checkpoint_dir / "adapter_config.json"),
            "adapter_weights_sha256": file_sha256(checkpoint_dir / "adapter_model.safetensors"),
            "adapter_tensor_contract": tensor_contract,
            **metrics,
        })
    auc_values = [row["trapezoid_auc_steps_1_to_500"] for row in runs]
    endpoints = [row["loss_at_microstep_500"] for row in runs]
    initial_medians = [row["first_100_median_loss"] for row in runs]
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_gate1_reconstructed_replay",
        "provenance_status": config["legacy_provenance"]["status"],
        "canonical_manifest_sha256": legacy["manifest_sha256"],
        "gate1_config": str(config_path.relative_to(root)),
        "gate1_config_sha256": file_sha256(config_path),
        "entrypoint": str(entrypoint.relative_to(root)),
        "entrypoint_sha256": file_sha256(entrypoint),
        "source_defaults": defaults,
        "source_contract": source_contract,
        "runs": runs,
        "reference_ranges": {
            "trapezoid_auc_steps_1_to_500": [min(auc_values), max(auc_values)],
            "loss_at_microstep_500": [min(endpoints), max(endpoints)],
            "first_100_median_loss": [min(initial_medians), max(initial_medians)],
        },
        "legacy_high_loss_explanation": {
            "observed": all(value > 10.0 for value in initial_medians),
            "observed_first_100_median_range": [min(initial_medians), max(initial_medians)],
            "code_mechanism": "Legacy labels cloned every padded input token and never replaced padding labels with -100.",
            "gradient_accumulation_not_the_multiplier": (
                "The legacy code divides loss before backward and multiplies it back only for logging; "
                "the CSV therefore stores the model loss, not a fourfold sum."
            ),
            "causal_confirmation_required": (
                "Gate 2 must compare the same openPangu batches with masked and legacy-unmasked labels."
            ),
        },
        "limitations": config["limitations"],
        "reusable_as_revision_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        payload = audit(args.config.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        stable_existing = {key: value for key, value in existing.items() if key != "captured_at_utc"}
        stable_payload = {key: value for key, value in payload.items() if key != "captured_at_utc"}
        if stable_existing != stable_payload:
            raise SystemExit(f"refusing to overwrite different legacy evidence audit: {args.output}")
        payload = existing
    else:
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
