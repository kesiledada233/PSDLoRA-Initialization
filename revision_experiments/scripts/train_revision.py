#!/usr/bin/env python3
"""Audited single-GPU LoRA training entrypoint used by run_matrix.py."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from revision_experiments.scripts.baseline_selection import resolve_selected_configuration, verify_selection_manifest
from revision_experiments.scripts.execution_gates import require_execution_gates
from revision_experiments.initializers.audit import (
    collect_lora_b_gradient_statistics,
    collect_lora_parameter_statistics,
)
from revision_experiments.initializers.variants import apply_registered_initialization
from revision_experiments.initializers.lora_one import (
    apply_lora_one_initialization,
    estimate_target_gradients,
)
from revision_experiments.scripts.gradient_logger import GradientLogger, FullMatrixDiagnosticLogger
from revision_experiments.scripts.matrix import checkpoint_steps_for_run, expand_matrix, load_matrix
from revision_experiments.scripts.metrics import raw_trapezoid_auc, trapezoid_auc_at_steps
from revision_experiments.scripts.openpangu_cuda_compat import loader_provenance
from revision_experiments.scripts.rng import derive_init_seed, seed_initialization, seed_training
from revision_experiments.scripts.schema import canonical_hash, dataset_split_hash, validate_metadata
from revision_experiments.scripts.training_support import (
    MODEL_IDENTIFIERS, MODEL_PATHS, build_datasets, build_datasets_with_validation, load_model, load_tokenizer,
)
from revision_experiments.scripts.verify_model_architecture import (
    bind_model_identity, resolve_projection_contract, target_module_contract, validate_scope_contract,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_CONFIG = ROOT / "revision_experiments/results/aggregate/baseline_selected_configs.yaml"
DEFAULT_BASELINE_MATRIX = ROOT / "revision_experiments/config/baseline_search_matrix.yaml"
DEFAULT_RUNS_ROOT = ROOT / "revision_experiments/results/runs"
DEFAULT_SMOKE_RUNS_ROOT = ROOT / "revision_experiments/results/smoke/runs"


def git_state() -> tuple[str, bool]:
    reconstructed = ROOT / ".git-reconstructed"
    prefix = ["git", f"--git-dir={reconstructed}", f"--work-tree={ROOT}"] if reconstructed.is_dir() else ["git"]
    commit = subprocess.run(prefix + ["rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    status = subprocess.run(prefix + ["status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False)
    if commit.returncode != 0:
        raise RuntimeError("A valid reconstructed Git baseline is required before training")
    return commit.stdout.strip(), bool(status.stdout.strip())


def environment_versions() -> dict:
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "conda": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
        "python": sys.version.split()[0],
        "packages": packages,
    }


def resolve_method(
    run: dict,
    selected_path: str | Path = DEFAULT_SELECTED_CONFIG,
    *,
    matrix: dict | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> tuple[dict, str, float | None, dict | None]:
    """Resolve method and LR, requiring validation selection for selected final runs."""
    method = run["method"]
    learning_rate = run.get("learning_rate")
    selection = None
    if run.get("selection_method"):
        selection_matrix = matrix or load_matrix(DEFAULT_BASELINE_MATRIX)
        selected_manifest = verify_selection_manifest(selection_matrix, runs_root, selected_path)
        selection = resolve_selected_configuration(
            selected_manifest, run["task"], method
        )
        method = selection["effective_method"]
        learning_rate = float(selection["learning_rate"])
    lora_kwargs = {}
    if method == "dora":
        lora_kwargs["use_dora"] = True
    elif method == "pissa":
        lora_kwargs["init_lora_weights"] = "pissa"
    return lora_kwargs, method, learning_rate, selection


def trainable_parameter_counts(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(trainable), int(total)


def write_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def fixed_batch_hash(batches: list[dict[str, torch.Tensor]]) -> str:
    """Bind the exact ordered tensors used by the initial-gradient audit."""

    payload = []
    for batch in batches:
        payload.append({
            key: value.detach().cpu().tolist()
            for key, value in sorted(batch.items())
        })
    return canonical_hash(payload)


@torch.no_grad()
def validation_loss(model, dataset, device, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    examples = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_examples = int(next(iter(batch.values())).shape[0])
        weighted_loss += float(model(**batch).loss.detach()) * batch_examples
        examples += batch_examples
    model.train(was_training)
    if examples != len(dataset) or examples <= 0:
        raise RuntimeError(f"Expected to score all {len(dataset)} validation examples, observed {examples}")
    return weighted_loss / examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run only a purpose=integration_smoke config")
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    is_smoke_matrix = matrix.get("purpose") == "integration_smoke"
    if bool(args.smoke) != bool(is_smoke_matrix):
        raise SystemExit("--smoke must be used exactly with a purpose=integration_smoke config")
    if not matrix.get("enabled", False):
        raise SystemExit("Training matrix is disabled")
    try:
        require_execution_gates(ROOT, require_integration_smokes=not args.smoke)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    candidates = {run["run_id"]: run for run in expand_matrix(matrix)}
    if args.run_id not in candidates:
        raise SystemExit(f"Unknown run ID for matrix: {args.run_id}")
    run = candidates[args.run_id]
    run_root = DEFAULT_SMOKE_RUNS_ROOT if args.smoke else DEFAULT_RUNS_ROOT
    run_dir = run_root / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        commit, dirty = git_state()
        if dirty:
            raise RuntimeError("Formal runs require a clean worktree")
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("Formal training requires an available CUDA device")
        training = run["training"]
        architecture_provenance = None
        if matrix["matrix_name"] == "scope":
            architecture_audit = resolve_projection_contract(
                MODEL_PATHS[run["model"]], rank=int(training["lora_rank"])
            )
            architecture_audit = bind_model_identity(architecture_audit, run["model"])
            validate_scope_contract(architecture_audit, matrix)
            architecture_provenance = target_module_contract(
                architecture_audit, list(run["target_modules"])
            )
        lora_kwargs, effective_method, selected_learning_rate, selection = resolve_method(run, matrix=matrix)
        effective_learning_rate = float(
            selected_learning_rate if selected_learning_rate is not None
            else run.get("learning_rate", training["learning_rate"])
        )
        device = torch.device(args.device)
        tokenizer = load_tokenizer(run["model"])
        uses_validation = run.get("section") == "screening" and "validation_interval" in training
        if uses_validation:
            train_dataset, validation_dataset, test_dataset = build_datasets_with_validation(
                run["task"], tokenizer, int(training["max_length"])
            )
        else:
            train_dataset, test_dataset = build_datasets(run["task"], tokenizer, int(training["max_length"]))
            validation_dataset = None
        config_payload = {
            "matrix": matrix["matrix_name"], **run,
            "effective_method": effective_method,
            "effective_learning_rate": effective_learning_rate,
        }
        if args.smoke:
            config_payload["formal_result"] = False
        if selection is not None:
            config_payload["validation_selection"] = selection
        config_hash = canonical_hash(config_payload)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")
        init_seed = derive_init_seed(run["seed"], effective_method)
        model_identifier = MODEL_IDENTIFIERS[run["model"]]
        template = getattr(tokenizer, "chat_template", None) or ""
        metadata = {
            "run_id": run["run_id"], "git_commit": commit, "dirty_worktree": False,
            "model_checkpoint": model_identifier, "tokenizer": model_identifier,
            "chat_template_hash": canonical_hash(template),
            "dataset_split_hash": dataset_split_hash(run["task"], uses_validation),
            "method": run["method"] if run["method"] in {"dora", "pissa", "lora_one", "validation_selected_proposed"} else run["method"],
            "seed": run["seed"], "init_seed": init_seed, "data_order_seed": run["seed"],
            "training_seed": run["seed"], "max_steps": run["max_steps"],
            "target_modules": run["target_modules"], "environment": environment_versions(),
            "hardware": {"device": torch.cuda.get_device_name(device), "count": 1}, "config_hash": config_hash,
            "effective_method": effective_method, "effective_learning_rate": effective_learning_rate,
            "validation_selection": selection,
            "model_loader_provenance": loader_provenance(run["model"]),
        }
        if architecture_provenance is not None:
            metadata["architecture_provenance"] = architecture_provenance
        errors = validate_metadata(metadata)
        if errors:
            raise RuntimeError("Invalid metadata: " + "; ".join(errors))
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        seed_initialization(init_seed)
        model = load_model(run["model"], device, training["precision"])
        from peft import LoraConfig, get_peft_model
        lora_one_gradients = None
        lora_one_gradient_seconds = None
        if effective_method == "lora_one":
            if uses_validation:
                gradient_train_dataset, _, _ = build_datasets_with_validation(
                    run["task"], tokenizer, int(run["gradient_max_length"])
                )
            else:
                gradient_train_dataset, _ = build_datasets(
                    run["task"], tokenizer, int(run["gradient_max_length"])
                )
            gradient_started = time.perf_counter()
            lora_one_gradients = estimate_target_gradients(
                model, gradient_train_dataset, run["target_modules"], device,
                batches=int(run["gradient_batches"]), batch_size=int(run["gradient_batch_size"]),
            )
            lora_one_gradient_seconds = time.perf_counter() - gradient_started
        adapter_started = time.perf_counter()
        lora_config = LoraConfig(
            r=int(training["lora_rank"]), lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]), target_modules=run["target_modules"],
            task_type="CAUSAL_LM", bias="none", use_rslora=bool(training.get("use_rslora", False)), **lora_kwargs,
        )
        model = get_peft_model(model, lora_config)
        if effective_method == "lora_one":
            init_info = apply_lora_one_initialization(
                model, lora_one_gradients, stable_gamma=float(run["stable_gamma"])
            )
            init_info["gradient_estimation_seconds"] = lora_one_gradient_seconds
            init_info["gradient_batches"] = int(run["gradient_batches"])
            init_info["gradient_batch_size"] = int(run["gradient_batch_size"])
            init_info["gradient_max_length"] = int(run["gradient_max_length"])
        elif effective_method not in {"peft_default", "dora", "pissa"}:
            init_info = apply_registered_initialization(model, effective_method, init_seed)
        else:
            init_info = {"method": effective_method, "source": "PEFT LoraConfig/reset path"}
        adapter_initialization_seconds = time.perf_counter() - adapter_started
        gradient_estimation_seconds = float(lora_one_gradient_seconds or 0.0)
        trainable_count, total_count = trainable_parameter_counts(model)
        if (
            architecture_provenance is not None
            and trainable_count != architecture_provenance["expected_lora_trainable_parameters"]
        ):
            raise RuntimeError(
                "PEFT trainable parameter count does not match the frozen architecture contract: "
                f"{trainable_count} != {architecture_provenance['expected_lora_trainable_parameters']}"
            )
        init_info["adapter_initialization_seconds"] = adapter_initialization_seconds
        init_info["gradient_estimation_seconds"] = gradient_estimation_seconds
        init_info["initialization_seconds_total"] = adapter_initialization_seconds + gradient_estimation_seconds
        init_info["trainable_parameter_count"] = trainable_count
        init_info["total_parameter_count"] = total_count
        initialization_audit_started = time.perf_counter()
        init_info["matrix_statistics_schema_version"] = 1
        init_info["matrix_statistics"] = collect_lora_parameter_statistics(model)
        init_info["initialization_audit_seconds"] = time.perf_counter() - initialization_audit_started

        checkpoints = checkpoint_steps_for_run(matrix, run)
        if 0 in checkpoints:
            model.save_pretrained(run_dir / "checkpoints" / "step_000000")

        seed_training(run["seed"])
        generator = torch.Generator().manual_seed(run["seed"])
        loader = DataLoader(train_dataset, batch_size=int(training["batch_size"]), shuffle=True, generator=generator)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=effective_learning_rate,
                                      weight_decay=float(training["weight_decay"]))
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(training["warmup_steps"]), run["max_steps"])
        logging = run.get("logging") or matrix.get("logging", {})
        gradient_logger = None
        full_matrix_logger = None
        if logging.get("gradient_coordinates"):
            gradient_logger = GradientLogger(model, run_dir / "gradients", run["max_steps"],
                                             logging["coordinates_per_matrix"], logging["coordinate_seed"])
            full_matrix_logger = FullMatrixDiagnosticLogger(
                model, logging["full_matrix_steps"], logging["coordinate_seed"]
            )
        raw_handle = (run_dir / "raw_loss.jsonl").open("w", encoding="utf-8")
        timing_handle = (run_dir / "timing.jsonl").open("w", encoding="utf-8")
        validation_handle = (run_dir / "validation_loss.jsonl").open("w", encoding="utf-8") if uses_validation else None
        diagnostic_handle = (run_dir / "gradient_diagnostics.jsonl").open("w", encoding="utf-8") if gradient_logger else None
        model.train()
        iterator = iter(loader)
        losses = []
        optimizer.zero_grad(set_to_none=True)

        accumulation_steps = int(training["gradient_accumulation_steps"])
        step_zero_batches = []
        validation_steps = []
        validation_losses = []
        if uses_validation:
            loss = validation_loss(
                model, validation_dataset, device,
                int(training["validation_batch_size"]),
            )
            validation_steps.append(0); validation_losses.append(loss)
            write_jsonl(validation_handle, {
                "step": 0, "validation_loss": loss,
                "validation_examples_evaluated": len(validation_dataset),
            })
        # Every method receives the same fixed step-zero global batch. It is
        # reused by optimizer step 1, and the training RNG is reset afterwards,
        # so this required initial-gradient audit does not alter paired data or
        # dropout streams. Diagnostic time is intentionally excluded from the
        # comparable adapter/training timing fields.
        initial_gradient_started = time.perf_counter()
        if gradient_logger:
            full_matrix_logger.start(0)
        for _ in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            step_zero_batches.append(batch)
            (model(**batch).loss / accumulation_steps).backward()
        init_info["initial_gradient_audit"] = {
            "schema_version": 1,
            "batch_scope": "one_complete_global_batch_before_optimizer_step_1",
            "microbatch_count": accumulation_steps,
            "ordered_batch_sha256": fixed_batch_hash(step_zero_batches),
            "lora_b": collect_lora_b_gradient_statistics(model),
            "audit_seconds": time.perf_counter() - initial_gradient_started,
        }
        if gradient_logger:
            gradient_logger.record(0)
            for row in gradient_logger.full_diagnostics(0):
                write_jsonl(diagnostic_handle, row)
            for row in full_matrix_logger.finish(0):
                write_jsonl(diagnostic_handle, row)
        optimizer.zero_grad(set_to_none=True)
        seed_training(run["seed"])
        (run_dir / "initialization_stats.json").write_text(
            json.dumps(init_info, indent=2) + "\n", encoding="utf-8",
        )

        for step in range(1, run["max_steps"] + 1):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            raw_loss = 0.0
            if full_matrix_logger:
                full_matrix_logger.start(step)
            for micro_step in range(accumulation_steps):
                if step == 1 and step_zero_batches:
                    batch = step_zero_batches[micro_step]
                else:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)
                    batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                raw_loss += float(output.loss.detach()) / accumulation_steps
                (output.loss / accumulation_steps).backward()
            if gradient_logger:
                gradient_logger.record(step)
                if step in set(logging["full_matrix_steps"]):
                    for row in gradient_logger.full_diagnostics(step):
                        write_jsonl(diagnostic_handle, row)
                    for row in full_matrix_logger.finish(step):
                        write_jsonl(diagnostic_handle, row)
            losses.append(raw_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            write_jsonl(raw_handle, {"step": step, "train_loss": raw_loss})
            write_jsonl(timing_handle, {"step": step, "step_time_seconds": elapsed})
            if uses_validation and step <= int(training["validation_max_step"]) and step % int(training["validation_interval"]) == 0:
                loss = validation_loss(
                    model, validation_dataset, device,
                    int(training["validation_batch_size"]),
                )
                validation_steps.append(step); validation_losses.append(loss)
                write_jsonl(validation_handle, {
                    "step": step, "validation_loss": loss,
                    "validation_examples_evaluated": len(validation_dataset),
                })
            if step in checkpoints:
                model.save_pretrained(run_dir / "checkpoints" / f"step_{step:06d}")
        raw_handle.close(); timing_handle.close()
        if validation_handle:
            validation_handle.close()
        if diagnostic_handle:
            diagnostic_handle.close()
        if full_matrix_logger:
            full_matrix_logger.close()
        summary = {
            "raw_auc500": raw_trapezoid_auc(losses, 0, 500) if run["max_steps"] >= 500 else None,
            "validation_auc500": trapezoid_auc_at_steps(validation_steps, validation_losses)
            if uses_validation else None,
            "validation_final_loss": validation_losses[-1] if uses_validation else None,
            "validation_guardrail_metric": "validation_loss" if uses_validation else None,
            "validation_guardrail_value": validation_losses[-1] if uses_validation else None,
            "validation_examples_evaluated": len(validation_dataset) if validation_dataset is not None else 0,
            "steps_logged": len(losses), "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset) if validation_dataset is not None else 0,
            "test_examples": len(test_dataset),
            "effective_method": effective_method, "effective_learning_rate": effective_learning_rate,
            "trainable_parameter_count": trainable_count,
            "initialization_seconds_total": init_info["initialization_seconds_total"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
        return 0
    except Exception as exc:
        (run_dir / "FAILED.json").write_text(json.dumps({"error": repr(exc), "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
