#!/usr/bin/env python3
"""Freeze validation-only screening indices without touching official tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from revision_experiments.scripts.schema import canonical_hash
from revision_experiments.scripts.training_support import load_task_records


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COUNTS = {"gsm8k": 256, "cmmlu": 67, "mbpp": 100, "sharegpt": 500}


def select_indices(task: str, records: list[dict], count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed) + sum((index + 1) * ord(char) for index, char in enumerate(task)))
    if task == "cmmlu":
        by_subject: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            by_subject.setdefault(str(record["subject"]), []).append(index)
        if count != len(by_subject):
            raise ValueError("CMMLU validation count must equal the number of subjects")
        selected = [int(rng.choice(indices)) for _, indices in sorted(by_subject.items())]
    else:
        if count >= len(records):
            raise ValueError(f"Validation count {count} must be smaller than {task} train size {len(records)}")
        selected = rng.choice(len(records), size=int(count), replace=False).tolist()
    return sorted(int(index) for index in selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "revision_experiments/data/processed/validation_split.json",
    )
    args = parser.parse_args()
    tasks = {}
    for task, count in DEFAULT_COUNTS.items():
        train_records, test_records = load_task_records(task)
        indices = select_indices(task, train_records, count, args.seed)
        tasks[task] = {
            "source_train_count": len(train_records),
            "official_test_count": len(test_records),
            "validation_count": len(indices),
            "validation_indices": indices,
            "validation_indices_hash": canonical_hash(indices),
        }
    payload = {"schema_version": 1, "selection_seed": args.seed, "tasks": tasks}
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"Refusing to overwrite different frozen validation split: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
