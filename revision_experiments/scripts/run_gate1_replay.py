#!/usr/bin/env python3
"""Run the bounded CUDA reconstruction of the submitted PEFT-default baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from revision_experiments.scripts.execution_gates import ROOT, validate_legacy_artifacts, validate_model_gate_evidence
from revision_experiments.scripts.metrics import raw_trapezoid_auc
from revision_experiments.scripts.openpangu_cuda_compat import loader_provenance
from revision_experiments.scripts.schema import canonical_hash, dataset_split_hash, file_sha256
from revision_experiments.scripts.training_support import (
    MODEL_IDENTIFIERS, load_model, load_task_records, load_tokenizer,
)


DEFAULT_CONFIG = ROOT / "revision_experiments/config/gate1_replay.yaml"
DEFAULT_OUTPUT = ROOT / "revision_experiments/results/gates/gate1_reproduction"
LEGACY_AUDIT = ROOT / "revision_experiments/results/audits/legacy_evidence.json"


class LegacyUnmaskedDataset(Dataset):
    """Reproduce the submitted label bug while retaining source indices."""

    def __init__(self, tokenizer, records: list[dict], task: str, max_length: int):
        self.examples = []
        for index, record in enumerate(records):
            text = format_legacy_record(task, record)
            if len(text.strip()) < 10:
                continue
            encoded = tokenizer(
                text, truncation=True, max_length=int(max_length), padding="max_length", return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": input_ids.clone(),
                "source_index": torch.tensor(index, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def format_legacy_record(task: str, record: dict) -> str:
    """Reproduce the submitted training entrypoint's text templates exactly."""
    if task == "gsm8k":
        return f"问题：{record['question']}\n解答：{record['answer']}"
    if task == "cmmlu":
        choices = f"A. {record['A']}  B. {record['B']}  C. {record['C']}  D. {record['D']}"
        return f"问题：{record['Question']}\n选项：{choices}\n答案：{record['Answer']}"
    if task == "mbpp":
        return f"# Problem\n{record['text']}\n\n# Solution\n{record['code']}"
    if task == "sharegpt":
        return "\n".join(
            f"{turn.get('from', 'unknown')}: {turn.get('value', '')}"
            for turn in record.get("conversations", [])
        ).strip()
    raise ValueError(f"Unsupported legacy task: {task}")


def git_state(root: Path) -> tuple[str, bool]:
    git_dir = root / ".git-reconstructed"
    command = ["git", f"--git-dir={git_dir}", f"--work-tree={root}"]
    commit = subprocess.run(command + ["rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    status = subprocess.run(command + ["status", "--porcelain"], text=True, capture_output=True, check=False)
    if commit.returncode or len(commit.stdout.strip()) != 40:
        raise RuntimeError("Gate 1-R requires the reconstructed Git repository")
    return commit.stdout.strip(), bool(status.stdout.strip())


def seed_legacy(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_prerequisites(config: dict, device: torch.device, minimum_free_gib: float) -> dict:
    if not config.get("enabled") or config.get("purpose") != "gate1_reconstruction_only_not_a_revision_result":
        raise RuntimeError("Gate 1-R replay config is disabled or has the wrong purpose")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Gate 1-R replay requires CUDA")
    training = config.get("training", {})
    if training.get("text_format") != "submitted_chinese_task_template":
        raise RuntimeError("Gate 1-R requires the submitted task text template")
    if training.get("peft_attachment_order") != (
        "base_model_on_cpu_then_move_augmented_model_to_accelerator"
    ):
        raise RuntimeError("Gate 1-R requires the submitted PEFT attachment order")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"Gate 1-R requires at least {minimum_free_gib:.1f} GiB free before model load; "
            f"observed {free_bytes / 1024**3:.2f} GiB"
        )
    legacy_errors, legacy = validate_legacy_artifacts(ROOT)
    model_errors, models = validate_model_gate_evidence(ROOT)
    if legacy_errors or model_errors:
        raise RuntimeError("prerequisite audit failed: " + "; ".join(legacy_errors + model_errors))
    audit = json.loads(LEGACY_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "ready_for_gate1_reconstructed_replay":
        raise RuntimeError("legacy evidence audit is not ready")
    config_path = ROOT / "revision_experiments/config/gate1_replay.yaml"
    if audit.get("gate1_config_sha256") != file_sha256(config_path):
        raise RuntimeError("legacy evidence audit is stale relative to gate1_replay.yaml")
    if audit.get("canonical_manifest_sha256") != legacy.get("manifest_sha256"):
        raise RuntimeError("legacy evidence audit is stale relative to the canonical manifest")
    return {"legacy": legacy, "models": models, "audit": audit, "free_bytes": free_bytes}


def run(config_path: Path, output: Path, device_label: str, minimum_free_gib: float) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(device_label)
    prerequisite = ensure_prerequisites(config, device, minimum_free_gib)
    commit, dirty = git_state(ROOT)
    if dirty:
        raise RuntimeError("Gate 1-R replay requires a clean worktree")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite Gate 1-R replay directory: {output}")

    training = config["training"]
    adapter = config["adapter"]
    seed = int(config["replay_seed"])
    replay_steps = int(training["replay_microsteps"])
    seed_legacy(seed)
    tokenizer = load_tokenizer(config["model"]["key"])
    # The submitted entrypoint attached PEFT while the base model was still on
    # CPU, then moved the augmented model to the accelerator.  This matters for
    # the RNG stream used by PEFT's random LoRA-A initialization.
    model = load_model(
        config["model"]["key"], device, config["model"]["precision"], move_to_device=False,
    )
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=int(adapter["rank"]), lora_alpha=int(adapter["alpha"]),
        lora_dropout=float(adapter["dropout"]), target_modules=list(adapter["target_modules"]),
        bias=adapter["bias"], task_type=adapter["task_type"], init_lora_weights=True,
    ))
    model.to(device)
    records, _ = load_task_records(config["task"])
    dataset = LegacyUnmaskedDataset(tokenizer, records, config["task"], training["max_length"])
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
    )
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(training["warmup_optimizer_steps"]), int(training["submitted_microsteps"]),
    )

    output.mkdir(parents=True)
    config_output = output / "config.yaml"
    raw_output = output / "raw_loss.jsonl"
    order_output = output / "data_order.jsonl"
    summary_output = output / "summary.json"
    checkpoint_output = output / "checkpoint"
    actual_config = {
        **config,
        "execution": {
            "git_commit": commit,
            "device": device_label,
            "device_name": torch.cuda.get_device_name(device),
            "free_bytes_before_load": prerequisite["free_bytes"],
            "model_identifier": MODEL_IDENTIFIERS[config["model"]["key"]],
            "model_loader_provenance": loader_provenance(config["model"]["key"]),
            "dataset_split_hash": dataset_split_hash(config["task"]),
            "legacy_evidence_sha256": file_sha256(LEGACY_AUDIT),
        },
    }
    config_output.write_text(yaml.safe_dump(actual_config, sort_keys=True), encoding="utf-8")
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    started = time.perf_counter()
    try:
        with raw_output.open("w", encoding="utf-8") as raw_handle, order_output.open("w", encoding="utf-8") as order_handle:
            model.train()
            for microstep in range(1, replay_steps + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                source_indices = [int(value) for value in batch.pop("source_index")]
                moved = {key: value.to(device) for key, value in batch.items()}
                loss = model(**moved).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at legacy microstep {microstep}")
                (loss / int(training["gradient_accumulation_microsteps"])).backward()
                if microstep % int(training["gradient_accumulation_microsteps"]) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                value = float(loss.detach())
                losses.append(value)
                raw_handle.write(json.dumps({
                    "microstep": microstep, "train_loss": value,
                    "optimizer_step": microstep // int(training["gradient_accumulation_microsteps"]),
                    "learning_rate": scheduler.get_last_lr()[0],
                }) + "\n")
                order_handle.write(json.dumps({"microstep": microstep, "source_indices": source_indices}) + "\n")
                raw_handle.flush(); order_handle.flush()
        model.save_pretrained(checkpoint_output)
        elapsed = time.perf_counter() - started
        if len(losses) != replay_steps or not all(math.isfinite(value) for value in losses):
            raise RuntimeError("Gate 1-R replay loss sequence is incomplete")
        summary = {
            "schema_version": 1,
            "status": "completed_reconstructed_replay",
            "reusable_as_revision_result": False,
            "legacy_microsteps": len(losses),
            "optimizer_steps": replay_steps // int(training["gradient_accumulation_microsteps"]),
            "trapezoid_auc_steps_1_to_500": raw_trapezoid_auc(losses, 0, replay_steps),
            "legacy_sum_auc500": sum(losses),
            "loss_at_microstep_500": losses[-1],
            "first_100_median_loss": float(np.median(losses[:100])),
            "wall_time_seconds": elapsed,
            "config_sha256": file_sha256(config_output),
            "raw_loss_sha256": file_sha256(raw_output),
            "data_order_sha256": file_sha256(order_output),
            "checkpoint_adapter_config_sha256": file_sha256(checkpoint_output / "adapter_config.json"),
            "checkpoint_adapter_weights_sha256": file_sha256(checkpoint_output / "adapter_model.safetensors"),
        }
        summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output / "COMPLETED").write_text("complete\n", encoding="utf-8")
        return summary
    except Exception as exc:
        (output / "FAILED.json").write_text(json.dumps({
            "error": repr(exc), "traceback": traceback.format_exc(),
        }, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--preflight", action="store_true", help="validate inputs/GPU without creating outputs")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.preflight:
            payload = ensure_prerequisites(config, torch.device(args.device), args.minimum_free_gib)
            print(json.dumps({
                "ready": True, "free_gib": payload["free_bytes"] / 1024**3,
                "legacy_artifact_count": payload["legacy"]["artifact_count"],
            }, indent=2))
            return 0
        summary = run(args.config.resolve(), args.output.resolve(), args.device, args.minimum_free_gib)
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
