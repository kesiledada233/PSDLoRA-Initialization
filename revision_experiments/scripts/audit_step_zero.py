#!/usr/bin/env python3
"""Gate 2 audit: functional equivalence and deterministic no-update losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from revision_experiments.initializers.variants import apply_registered_initialization
from revision_experiments.scripts.rng import derive_init_seed, seed_initialization, seed_training
from revision_experiments.scripts.openpangu_cuda_compat import loader_provenance
from revision_experiments.scripts.schema import canonical_hash, file_sha256
from revision_experiments.scripts.training_support import (
    MODEL_IDENTIFIERS, MODEL_PATHS, build_datasets, load_model, load_tokenizer,
)


ROOT = Path(__file__).resolve().parents[2]


def first_batches(dataset, count: int):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for index, batch in enumerate(loader):
        if index >= count:
            break
        yield batch


def model_losses(model, batches, device):
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**moved).loss))
    return losses


def target_diagnostics(batches) -> list[dict[str, int]]:
    """Record the scored-token and padding-label contract for every audited batch."""
    rows = []
    for index, batch in enumerate(batches):
        scored = batch["labels"] != -100
        rows.append({
            "batch": index,
            "valid_target_tokens": int(scored.sum()),
            "padding_target_tokens": int((scored & (batch["attention_mask"] == 0)).sum()),
        })
    return rows


def with_legacy_unmasked_labels(batches):
    """Clone batches using the submitted bug: padding IDs remain scored labels."""
    return [
        {**batch, "labels": batch["input_ids"].clone()}
        for batch in batches
    ]


def adapter_equivalence(model, batches, device) -> tuple[list[float], dict[str, float]]:
    """Compare an enabled adapter with its disabled base on every identical batch."""
    losses: list[float] = []
    max_logit_difference = 0.0
    sum_logit_difference = 0.0
    logit_elements = 0
    loss_differences: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved = {key: value.to(device) for key, value in batch.items()}
            with model.disable_adapter():
                base_output = model(**moved)
            adapter_output = model(**moved)
            difference = (adapter_output.logits.detach().float() - base_output.logits.detach().float()).abs()
            max_logit_difference = max(max_logit_difference, float(difference.max()))
            sum_logit_difference += float(difference.sum())
            logit_elements += difference.numel()
            adapter_loss = float(adapter_output.loss)
            losses.append(adapter_loss)
            loss_differences.append(abs(adapter_loss - float(base_output.loss)))
    if not losses or logit_elements <= 0:
        raise RuntimeError("Step-zero equivalence requires at least one nonempty batch")
    return losses, {
        "max_abs_logit_difference": max_logit_difference,
        "mean_abs_logit_difference": sum_logit_difference / logit_elements,
        "max_abs_loss_difference": max(loss_differences),
        "mean_abs_loss_difference": sum(loss_differences) / len(loss_differences),
    }


def tokenizer_artifact_hash(model_key: str) -> tuple[str, list[str]]:
    """Hash the local tokenizer assets without rehashing multi-gigabyte model shards."""
    names = sorted(
        item.name for item in MODEL_PATHS[model_key].iterdir()
        if item.is_file() and (
            item.name.startswith(("tokenizer", "vocab", "merges", "added_tokens", "special_tokens"))
            or item.suffix == ".model"
        )
    )
    if not names:
        raise RuntimeError(f"No tokenizer artifacts found for {model_key}")
    return canonical_hash([{"file": name, "sha256": file_sha256(MODEL_PATHS[model_key] / name)} for name in names]), names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["openpangu", "qwen"], required=True)
    parser.add_argument("--task", choices=["gsm8k", "cmmlu", "sharegpt", "mbpp"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "revision_experiments/results/audits/step_zero_cases",
    )
    args = parser.parse_args()
    if args.batches <= 0:
        raise SystemExit("--batches must be positive")
    case_id = f"{args.model}__{args.task}"
    json_path = args.output_dir / f"{case_id}.json"
    csv_path = args.output_dir / f"{case_id}.csv"
    if json_path.exists() or csv_path.exists():
        raise SystemExit(f"Refusing to overwrite step-zero case evidence: {case_id}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; Gate 2 audit was not run")
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model)
    train_dataset, _ = build_datasets(args.task, tokenizer, 512)
    cached_batches = list(first_batches(train_dataset, args.batches))
    if len(cached_batches) != args.batches:
        raise SystemExit(f"Requested {args.batches} batches but dataset supplied {len(cached_batches)}")
    diagnostics = target_diagnostics(cached_batches)
    methods = ["base", "peft_default", "powerlaw_global_a06"]
    all_losses = {}
    equivalence = {}
    legacy_unmasked_losses = None
    for method in methods:
        seed_initialization(derive_init_seed(args.seed, method))
        model = load_model(args.model, device, args.precision)
        if method == "base":
            seed_training(args.seed)
            all_losses[method] = model_losses(model, cached_batches, device)
            if args.model == "openpangu":
                legacy_unmasked_losses = model_losses(
                    model, with_legacy_unmasked_labels(cached_batches), device,
                )
        else:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none",
            ))
            if method != "peft_default":
                apply_registered_initialization(model, method, derive_init_seed(args.seed, method))
            b_nonzero = sum(torch.count_nonzero(parameter).item() for name, parameter in model.named_parameters()
                            if "lora_B" in name)
            seed_training(args.seed)
            all_losses[method], differences = adapter_equivalence(model, cached_batches, device)
            equivalence[method] = {"b_nonzero": int(b_nonzero), **differences}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for method, losses in all_losses.items():
        rows.extend({
            "model": args.model, "task": args.task, "seed": args.seed,
            "batch": index, "method": method, "loss": loss,
        } for index, loss in enumerate(losses))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    tolerance = 1e-6 if args.precision == "fp32" else 1e-3
    tokenizer_hash, tokenizer_files = tokenizer_artifact_hash(args.model)
    base_median = float(np.median(all_losses["base"]))
    payload = {
        "schema_version": 4, "case_id": case_id,
        "model": args.model, "task": args.task, "seed": args.seed, "batches": len(cached_batches),
        "checkpoint": MODEL_IDENTIFIERS[args.model],
        "checkpoint_verification_sha256": file_sha256(
            ROOT / "revision_experiments/results/audits" / f"{args.model}_checkpoint_verification.json"
        ),
        "model_loader_provenance": loader_provenance(args.model),
        "precision": args.precision, "tolerance": tolerance, "equivalence": equivalence,
        "median_losses": {method: float(np.median(losses)) for method, losses in all_losses.items()},
        "batch_target_diagnostics": diagnostics,
        "valid_target_tokens_total": sum(row["valid_target_tokens"] for row in diagnostics),
        "padding_target_tokens_total": sum(row["padding_target_tokens"] for row in diagnostics),
        "label_shift": "Transformers causal-LM internal one-token shift",
        "ignore_index": -100, "loss_reduction": "mean over non-ignored shifted tokens",
        "tokenizer_artifacts_hash": tokenizer_hash, "tokenizer_artifacts": tokenizer_files,
        "chat_template_hash": canonical_hash(getattr(tokenizer, "chat_template", None) or ""),
    }
    payload["equivalence_passed"] = all(
        values["b_nonzero"] == 0 and values["max_abs_logit_difference"] <= tolerance
        and values["mean_abs_logit_difference"] <= tolerance
        and values["max_abs_loss_difference"] <= tolerance
        and values["mean_abs_loss_difference"] <= tolerance for values in equivalence.values()
    )
    payload["loss_anomaly"] = args.model == "openpangu" and base_median > 10.0
    legacy_audit_path = ROOT / "revision_experiments/results/audits/legacy_evidence.json"
    if args.model == "openpangu":
        if legacy_unmasked_losses is None or not legacy_audit_path.is_file():
            raise SystemExit("openPangu Gate 2 requires legacy unmasked losses and legacy_evidence.json")
        legacy_unmasked_median = float(np.median(legacy_unmasked_losses))
        legacy_padding_targets = sum(
            int(((batch["attention_mask"] == 0) & (batch["input_ids"] != -100)).sum())
            for batch in cached_batches
        )
        mechanism_reproduced = (
            base_median <= 10.0 and legacy_unmasked_median > 10.0
            and legacy_unmasked_median > base_median and legacy_padding_targets > 0
        )
        payload["legacy_label_comparison"] = {
            "status": "resolved_by_padding_label_masking" if mechanism_reproduced else "unresolved",
            "masked_median_loss": base_median,
            "legacy_unmasked_median_loss": legacy_unmasked_median,
            "median_loss_increase": legacy_unmasked_median - base_median,
            "legacy_padding_targets_scored": legacy_padding_targets,
            "same_batches": True,
            "legacy_evidence_sha256": file_sha256(legacy_audit_path),
        }
    else:
        mechanism_reproduced = True
        payload["legacy_label_comparison"] = {"status": "not_applicable"}
    payload["loss_anomaly_diagnostic"] = {
        "trigger_threshold": 10.0, "observed_base_median_loss": base_median,
        "status": (
            "unresolved" if payload["loss_anomaly"] or not mechanism_reproduced
            else "resolved_legacy_padding_labels" if args.model == "openpangu"
            else "not_triggered"
        ),
        "checks": {
            "checkpoint_tokenizer_binding": {
                "checkpoint": MODEL_IDENTIFIERS[args.model], "tokenizer_artifacts_hash": tokenizer_hash,
            },
            "chat_template_hash": payload["chat_template_hash"],
            "label_shift": payload["label_shift"], "ignore_index": payload["ignore_index"],
            "padding_target_tokens_total": payload["padding_target_tokens_total"],
            "sequence_packing": False, "optimizer_updates": 0,
            "gradient_accumulation_log_summation": False, "external_loss_scaling": False,
        },
    }
    payload["batch_losses_file"] = csv_path.name
    payload["batch_losses_sha256"] = file_sha256(csv_path)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["equivalence_passed"] and not payload["loss_anomaly"] and mechanism_reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
