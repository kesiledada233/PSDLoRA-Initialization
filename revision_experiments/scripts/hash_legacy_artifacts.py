#!/usr/bin/env python3
"""Create a deterministic SHA256 manifest for imported submission artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "legacy_artifacts/submission")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "SHA256SUMS"
    files = [path for path in sorted(args.root.rglob("*")) if path.is_file() and path != output]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{file_sha256(path)}  {path.relative_to(args.root)}\n" for path in files), encoding="utf-8")
    print(f"hashed {len(files)} files -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
