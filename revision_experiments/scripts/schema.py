"""Validation for immutable run metadata and result directory contracts."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from revision_experiments.initializers.registry import METHOD_REGISTRY

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ROOT = Path(__file__).resolve().parents[2]

REQUIRED_METADATA = {
    "run_id", "git_commit", "dirty_worktree", "model_checkpoint", "tokenizer",
    "chat_template_hash", "dataset_split_hash", "method", "seed",
    "init_seed", "data_order_seed", "training_seed", "max_steps",
    "target_modules", "environment", "hardware", "config_hash",
}
BASELINE_METHODS = {"dora", "pissa", "lora_one", "validation_selected_proposed"}
RESULT_METHODS = set(METHOD_REGISTRY) | BASELINE_METHODS


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    """Hash file names and contents for a deterministic dataset directory."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file() and not any(
        part.startswith(".") for part in item.relative_to(root).parts
    ))
    if not files:
        raise ValueError(f"Dataset directory has no hashable files: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


@lru_cache(maxsize=None)
def dataset_split_hash(task: str, uses_validation_split: bool = False) -> str:
    paths = {
        "gsm8k": ROOT / "pretrained_models/gsm8k/dataset_dict.json",
        "cmmlu": ROOT / "revision_experiments/data/processed/cmmlu/dataset_dict.json",
        "mbpp": ROOT / "revision_experiments/data/processed/mbpp/dataset_dict.json",
        "sharegpt": ROOT / "revision_experiments/data/processed/sharegpt_split.json",
    }
    path = paths[task]
    source_hash = tree_sha256(path.parent) if path.name == "dataset_dict.json" else file_sha256(path)
    if not uses_validation_split:
        return source_hash
    validation_hash = file_sha256(ROOT / "revision_experiments/data/processed/validation_split.json")
    return canonical_hash({"source": source_hash, "validation_split": validation_hash})


@lru_cache(maxsize=None)
def formal_sharegpt_examples() -> tuple[dict, ...]:
    """Return frozen prompts together with their source-bound reference replies."""
    prompt_path = ROOT / "revision_experiments/data/processed/sharegpt_judge_prompts.jsonl"
    source_path = ROOT / "pretrained_models/sharegpt_datasets/computer_en_26k.jsonl"
    prompts = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = []
    for prompt in prompts:
        source_index = prompt.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or not 0 <= source_index < len(source_rows):
            raise RuntimeError("frozen ShareGPT prompt has an invalid source index")
        turns = source_rows[source_index].get("conversation")
        if not isinstance(turns, list) or not turns or not isinstance(turns[0], dict):
            raise RuntimeError("frozen ShareGPT source has no paired first turn")
        human = turns[0].get("human")
        assistant = turns[0].get("assistant")
        if human != prompt.get("prompt") or not isinstance(assistant, str) or not assistant.strip():
            raise RuntimeError("frozen ShareGPT prompt/reference does not match its source turn")
        examples.append({**prompt, "reference_response": assistant})
    return tuple(examples)


@lru_cache(maxsize=None)
def formal_sample_entries(task: str) -> tuple[dict, ...]:
    """Return ordered sample identifiers and per-input content hashes."""
    if task == "sharegpt":
        rows = list(formal_sharegpt_examples())
        identifiers = [row.get("prompt_id") for row in rows]
    else:
        from revision_experiments.scripts.training_support import load_task_records
        _, rows = load_task_records(task)
        key = {"gsm8k": "id", "cmmlu": "id", "mbpp": "task_id"}[task]
        identifiers = [row.get(key, index) for index, row in enumerate(rows)]
    if not rows or len(set(map(str, identifiers))) != len(rows):
        raise RuntimeError(f"formal {task} sample identifiers are empty or not unique")
    return tuple(
        {"sample_id": identifier, "sample_input_sha256": canonical_hash(row)}
        for identifier, row in zip(identifiers, rows)
    )


@lru_cache(maxsize=None)
def formal_sample_manifest(task: str) -> dict:
    """Return the ordered, content-bound sample set for formal evaluation."""
    entries = formal_sample_entries(task)
    return {
        "sample_count": len(entries),
        "ordered_sample_ids_sha256": canonical_hash([item["sample_id"] for item in entries]),
        "ordered_sample_content_sha256": canonical_hash([item["sample_input_sha256"] for item in entries]),
    }


def formal_evaluation_config(run_id: str, checkpoint: int, task: str, protocol: dict) -> dict:
    """Build the canonical matrix-bound configuration for one formal evaluation."""
    return {
        "schema_version": 1, "run_id": run_id, "checkpoint": int(checkpoint), "task": task,
        "protocol": dict(protocol),
        "dataset_split_hash": dataset_split_hash(task, uses_validation_split=False),
        "sample_manifest": formal_sample_manifest(task),
    }


def validate_metadata(metadata: dict) -> list[str]:
    errors = [f"missing field: {key}" for key in sorted(REQUIRED_METADATA - metadata.keys())]
    if "git_commit" in metadata and not COMMIT_RE.fullmatch(str(metadata["git_commit"])):
        errors.append("git_commit must be a 40-character lowercase SHA1")
    if metadata.get("dirty_worktree") is not False:
        errors.append("dirty_worktree must be false")
    for key in ("chat_template_hash", "dataset_split_hash", "config_hash"):
        if key in metadata and not SHA256_RE.fullmatch(str(metadata[key])):
            errors.append(f"{key} must be a 64-character lowercase SHA256")
    if metadata.get("method") not in RESULT_METHODS:
        errors.append(f"method must be registered: {metadata.get('method')!r}")
    for key in ("seed", "init_seed", "data_order_seed", "training_seed", "max_steps"):
        if key in metadata and (not isinstance(metadata[key], int) or metadata[key] < 0):
            errors.append(f"{key} must be a non-negative integer")
    if not isinstance(metadata.get("target_modules"), list) or not metadata.get("target_modules"):
        errors.append("target_modules must be a non-empty list")
    for key in ("model_checkpoint", "tokenizer"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip() or value in {"latest", "main", "unknown"}:
            errors.append(f"{key} must be an exact non-ambiguous identifier")
    return errors


def validate_run_directory(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    required = ["metadata.json", "config.yaml", "raw_loss.jsonl", "timing.jsonl", "initialization_stats.json"]
    errors = [f"missing file: {name}" for name in required if not (run_dir / name).is_file()]
    if (run_dir / "metadata.json").is_file():
        try:
            errors.extend(validate_metadata(json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid metadata.json: {exc}")
    completed = (run_dir / "COMPLETED").is_file()
    failed = (run_dir / "FAILED.json").is_file()
    if completed and failed:
        errors.append("run has both COMPLETED and FAILED.json terminal markers")
    elif not completed and not failed:
        errors.append("run has neither COMPLETED nor FAILED.json terminal marker")
    return errors
