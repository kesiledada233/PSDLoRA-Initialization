#!/usr/bin/env python3
"""Freeze a deterministic prompt subset from the held-out ShareGPT split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]


def first_user_prompt(record: dict) -> str | None:
    turns = record.get("conversation") or record.get("conversations") or record.get("messages") or []
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, dict):
            continue
        paired_human = str(turn.get("human") or "").strip()
        if paired_human:
            return paired_human
        role = str(turn.get("from") or turn.get("role") or "").lower()
        content = str(turn.get("value") or turn.get("content") or "").strip()
        if role in {"human", "user"} and content:
            return content
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--selection-seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "revision_experiments/data/processed/sharegpt_judge_prompts.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite frozen prompts: {args.output}")
    source = ROOT / "pretrained_models/sharegpt_datasets/computer_en_26k.jsonl"
    split = json.loads((ROOT / "revision_experiments/data/processed/sharegpt_split.json").read_text())
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = []
    for index in split["test_indices"]:
        prompt = first_user_prompt(records[index])
        if prompt:
            eligible.append({"prompt_id": f"sharegpt_{index:06d}", "source_index": index, "prompt": prompt})
    selected = random.Random(args.selection_seed).sample(eligible, args.count)
    selected.sort(key=lambda row: row["prompt_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    manifest = {
        "count": len(selected), "selection_seed": args.selection_seed,
        "split_sha256": file_sha256(ROOT / "revision_experiments/data/processed/sharegpt_split.json"),
        "prompts_sha256": file_sha256(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
