"""Shared model, tokenizer, and dataset builders for audited revision runs."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".cache/huggingface/modules"))

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


MODEL_PATHS = {
    "openpangu": ROOT / "pretrained_models/openPangu-Embedded-7B-V1.1",
    "qwen": ROOT / "pretrained_models/Qwen2.5-7B",
}
MODEL_IDENTIFIERS = {
    "openpangu": "openpangu/openPangu-Embedded-7B-V1.1@0ae1841cbd53f5218f2ce5dc63083d5382cfc9f5",
    "qwen": "Qwen/Qwen2.5-7B@e25af2efae60472008fbeaf5fb7c4274a87f78d4",
}
DATA_PATHS = {
    "gsm8k": ROOT / "pretrained_models/gsm8k",
    "cmmlu": ROOT / "revision_experiments/data/processed/cmmlu",
    "mbpp": ROOT / "revision_experiments/data/processed/mbpp",
}


def linearize_sharegpt(record: dict) -> str:
    turns = record.get("conversation") or record.get("conversations") or record.get("messages") or []
    lines = []
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, dict):
            continue
        if "human" in turn or "assistant" in turn:
            human = str(turn.get("human") or "").strip()
            assistant = str(turn.get("assistant") or "").strip()
            if human:
                lines.append(f"human: {human}")
            if assistant:
                lines.append(f"assistant: {assistant}")
            continue
        role = turn.get("from") or turn.get("role") or "unknown"
        value = turn.get("value") or turn.get("content") or ""
        if str(value).strip():
            lines.append(f"{role}: {str(value).strip()}")
    return "\n".join(lines)


def format_record(task: str, record: dict) -> str:
    if task == "gsm8k":
        return f"Question: {record['question']}\nAnswer: {record['answer']}"
    if task == "cmmlu":
        return (
            f"问题：{record['Question']}\n选项：A. {record['A']} B. {record['B']} "
            f"C. {record['C']} D. {record['D']}\n答案：{record['Answer']}"
        )
    if task == "mbpp":
        return f"# Problem\n{record['text']}\n\n# Solution\n{record['code']}"
    if task == "sharegpt":
        return linearize_sharegpt(record)
    raise ValueError(f"Unsupported task: {task}")


def load_task_records(task: str) -> tuple[list[dict], list[dict]]:
    if task in DATA_PATHS:
        dataset = load_from_disk(str(DATA_PATHS[task]))
        return [dict(row) for row in dataset["train"]], [dict(row) for row in dataset["test"]]
    if task != "sharegpt":
        raise ValueError(f"Unsupported task: {task}")
    source = ROOT / "pretrained_models/sharegpt_datasets/computer_en_26k.jsonl"
    split = json.loads((ROOT / "revision_experiments/data/processed/sharegpt_split.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [records[index] for index in split["train_indices"]], [records[index] for index in split["test_indices"]]


class CausalTextDataset(Dataset):
    """Tokenized causal-LM dataset that always ignores padding in labels."""

    def __init__(self, tokenizer, texts: list[str], max_length: int):
        self.tokenizer = tokenizer
        self.texts = [text for text in texts if text.strip()]
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_datasets(task: str, tokenizer, max_length: int):
    train_records, test_records = load_task_records(task)
    train_texts = [format_record(task, row) for row in train_records]
    test_texts = [format_record(task, row) for row in test_records]
    return CausalTextDataset(tokenizer, train_texts, max_length), CausalTextDataset(tokenizer, test_texts, max_length)


def partition_validation_records(task: str, records: list[dict], indices, expected_count: int):
    """Partition records while rejecting duplicate or out-of-range frozen indices."""
    validation_indices = {int(index) for index in indices}
    if len(validation_indices) != int(expected_count):
        raise RuntimeError(f"Duplicate or invalid frozen validation indices for {task}")
    if any(index < 0 or index >= len(records) for index in validation_indices):
        raise RuntimeError(f"Frozen validation split contains out-of-range indices for {task}")
    screening_train = [row for index, row in enumerate(records) if index not in validation_indices]
    validation = [row for index, row in enumerate(records) if index in validation_indices]
    if len(screening_train) + len(validation) != len(records) or len(validation) != int(expected_count):
        raise RuntimeError(f"Frozen validation partition is incomplete for {task}")
    return screening_train, validation


def build_datasets_with_validation(task: str, tokenizer, max_length: int):
    """Return screening-train, frozen-validation, official-test datasets."""
    train_records, test_records = load_task_records(task)
    manifest_path = ROOT / "revision_experiments/data/processed/validation_split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["tasks"][task]
    if int(entry["source_train_count"]) != len(train_records):
        raise RuntimeError(f"Frozen validation split no longer matches {task} train records")
    screening_train, validation = partition_validation_records(
        task, train_records, entry["validation_indices"], int(entry["validation_count"])
    )
    encode = lambda records: CausalTextDataset(tokenizer, [format_record(task, row) for row in records], max_length)
    return encode(screening_train), encode(validation), encode(test_records)


def load_tokenizer(model_key: str):
    from transformers import AutoTokenizer
    model_path = MODEL_PATHS[model_key]
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing model directory: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_key: str, device: torch.device, precision: str, *, move_to_device: bool = True,
):
    from transformers import AutoModelForCausalLM
    from revision_experiments.scripts.openpangu_cuda_compat import openpangu_cuda_overlay

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    source = (
        openpangu_cuda_overlay(MODEL_PATHS[model_key])
        if model_key == "openpangu" and device.type == "cuda"
        else nullcontext(MODEL_PATHS[model_key])
    )
    with source as model_path:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
    return model.to(device) if move_to_device else model
