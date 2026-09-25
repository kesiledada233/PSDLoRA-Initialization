#!/usr/bin/env python3
"""CPU-only tokenizer and real-record formatting smoke audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from revision_experiments.scripts.schema import canonical_hash
from revision_experiments.scripts.training_support import (
    MODEL_IDENTIFIERS,
    format_record,
    load_task_records,
    load_tokenizer,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "revision_experiments/results/audits/tokenizer_smoke.json",
    )
    args = parser.parse_args()
    samples = {}
    for task in ("gsm8k", "cmmlu", "mbpp", "sharegpt"):
        train, test = load_task_records(task)
        text = format_record(task, test[0])
        if not text.strip():
            raise RuntimeError(f"Empty formatted text for {task}")
        samples[task] = {"text": text, "train_count": len(train), "test_count": len(test)}
    models = {}
    for model_key in ("openpangu", "qwen"):
        tokenizer = load_tokenizer(model_key)
        tasks = {}
        for task, sample in samples.items():
            encoded = tokenizer(sample["text"], truncation=True, max_length=512)
            token_count = len(encoded["input_ids"])
            if token_count <= 0 or token_count > 512:
                raise RuntimeError(f"Invalid {model_key}/{task} token count: {token_count}")
            tasks[task] = {
                "token_count": token_count,
                "train_count": sample["train_count"],
                "test_count": sample["test_count"],
            }
        models[model_key] = {
            "checkpoint": MODEL_IDENTIFIERS[model_key],
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "chat_template_hash": canonical_hash(getattr(tokenizer, "chat_template", None) or ""),
            "tasks": tasks,
        }
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
