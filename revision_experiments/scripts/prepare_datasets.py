#!/usr/bin/env python3
"""Create isolated, deterministic dataset artifacts without touching source data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from revision_experiments.scripts.schema import canonical_hash, file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "revision_experiments" / "data" / "processed"


def _dataset_ready(path: Path) -> bool:
    try:
        dataset = load_from_disk(str(path))
        return isinstance(dataset, DatasetDict) and {"train", "test"} <= set(dataset)
    except Exception:
        return False


def prepare_cmmlu(output_root: Path) -> dict:
    source = ROOT / "pretrained_models" / "cmmlu"
    output = output_root / "cmmlu"
    if _dataset_ready(output):
        dataset = load_from_disk(str(output))
        return {"status": "existing", "train": len(dataset["train"]), "test": len(dataset["test"])}
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete output: {output}")
    rows = {"train": [], "test": []}
    source_hashes = {}
    for split, directory in (("train", source / "dev"), ("test", source / "test")):
        for path in sorted(directory.glob("*.csv")):
            source_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows[split].append({
                        "Question": row["Question"], "A": row["A"], "B": row["B"],
                        "C": row["C"], "D": row["D"], "Answer": row["Answer"],
                        "subject": path.stem,
                    })
    dataset = DatasetDict({split: Dataset.from_list(values) for split, values in rows.items()})
    dataset.save_to_disk(str(output))
    return {
        "status": "created", "train": len(rows["train"]), "test": len(rows["test"]),
        "source_hash": canonical_hash(source_hashes),
    }


def prepare_mbpp(output_root: Path) -> dict:
    source = ROOT / "pretrained_models" / "mbpp" / "mbpp.jsonl"
    output = output_root / "mbpp"
    if _dataset_ready(output):
        dataset = load_from_disk(str(output))
        return {"status": "existing", "train": len(dataset["train"]), "test": len(dataset["test"])}
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete output: {output}")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = ("task_id", "text", "code", "test_setup_code", "test_list", "challenge_test_list")
    normalized = [{key: record.get(key) for key in fields} for record in records]
    train = [record for record in normalized if int(record["task_id"]) <= 600]
    test = [record for record in normalized if int(record["task_id"]) > 600]
    DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)}).save_to_disk(str(output))
    return {
        "status": "created", "train": len(train), "test": len(test),
        "source_hash": file_sha256(source),
    }


def prepare_sharegpt_split(output_root: Path, split_seed: int = 20260903, test_ratio: float = 0.1) -> dict:
    source = ROOT / "pretrained_models" / "sharegpt_datasets" / "computer_en_26k.jsonl"
    output = output_root / "sharegpt_split.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("source_sha256") != file_sha256(source):
            raise RuntimeError("Existing ShareGPT split belongs to a different source file")
        return {"status": "existing", "train": len(payload["train_indices"]), "test": len(payload["test_indices"])}
    line_count = sum(1 for line in source.open("r", encoding="utf-8") if line.strip())
    indices = list(range(line_count))
    random.Random(split_seed).shuffle(indices)
    test_count = int(round(line_count * test_ratio))
    payload = {
        "schema_version": 1,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": file_sha256(source),
        "split_seed": split_seed,
        "test_ratio": test_ratio,
        "train_indices": sorted(indices[test_count:]),
        "test_indices": sorted(indices[:test_count]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"status": "created", "train": len(payload["train_indices"]), "test": len(payload["test_indices"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=["cmmlu", "mbpp", "sharegpt"], default=["cmmlu", "mbpp", "sharegpt"])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = {}
    if "cmmlu" in args.datasets:
        result["cmmlu"] = prepare_cmmlu(args.output_root)
    if "mbpp" in args.datasets:
        result["mbpp"] = prepare_mbpp(args.output_root)
    if "sharegpt" in args.datasets:
        result["sharegpt"] = prepare_sharegpt_split(args.output_root)
    manifest = args.output_root / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
