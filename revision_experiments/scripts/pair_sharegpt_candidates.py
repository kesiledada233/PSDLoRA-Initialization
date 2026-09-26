#!/usr/bin/env python3
"""Join two candidate files by prompt ID for blinded pairwise judging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["prompt_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prompt IDs in {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite pair file: {args.output}")
    left, right = load(args.left), load(args.right)
    if set(left) != set(right):
        raise SystemExit("Candidate prompt ID sets differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        for prompt_id in sorted(left):
            target.write(json.dumps({
                "prompt_id": prompt_id, "prompt": left[prompt_id]["prompt"],
                "left_run_id": left[prompt_id]["run_id"], "right_run_id": right[prompt_id]["run_id"],
                "response_left": left[prompt_id]["response"], "response_right": right[prompt_id]["response"],
            }, ensure_ascii=False) + "\n")
    print(f"paired {len(left)} prompts -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
