#!/usr/bin/env python3
"""Fail-closed verification for a pinned Hugging Face safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open

from revision_experiments.scripts.schema import file_sha256


HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def verify_checkpoint(model_dir: str | Path, revision: str) -> dict:
    model_dir = Path(model_dir).resolve()
    if not HEX40.fullmatch(revision):
        raise RuntimeError("revision must be an exact 40-character commit SHA")
    incomplete = sorted(str(path.relative_to(model_dir)) for path in model_dir.rglob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"checkpoint has incomplete downloads: {incomplete}")
    for name in ("config.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"checkpoint is missing {name}")

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid weight index: {exc}") from exc
        shard_names = sorted(set(weight_map.values()))
    elif (model_dir / "model.safetensors").is_file():
        shard_names = ["model.safetensors"]
        weight_map = None
    else:
        raise RuntimeError("checkpoint has neither a safetensors index nor a single safetensors file")

    shard_reports = []
    observed_keys: set[str] = set()
    cache_dir = model_dir / ".cache/huggingface/download"
    for shard_name in shard_names:
        shard = model_dir / shard_name
        metadata = cache_dir / f"{shard_name}.metadata"
        if not shard.is_file():
            raise RuntimeError(f"checkpoint is missing indexed shard {shard_name}")
        try:
            lines = metadata.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"checkpoint is missing Hub metadata for {shard_name}") from exc
        if len(lines) < 2 or lines[0] != revision or not HEX64.fullmatch(lines[1]):
            raise RuntimeError(f"invalid pinned revision/LFS SHA metadata for {shard_name}")
        actual_sha = file_sha256(shard)
        if actual_sha != lines[1]:
            raise RuntimeError(f"SHA256 mismatch for {shard_name}: {actual_sha} != {lines[1]}")
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
        except Exception as exc:
            raise RuntimeError(f"cannot read safetensors header for {shard_name}: {exc}") from exc
        if observed_keys & keys:
            raise RuntimeError(f"duplicate tensor names across shards: {sorted(observed_keys & keys)[:5]}")
        observed_keys.update(keys)
        if weight_map is not None:
            indexed_keys = {key for key, value in weight_map.items() if value == shard_name}
            if keys != indexed_keys:
                raise RuntimeError(f"weight index/header mismatch for {shard_name}")
        shard_reports.append({
            "file": shard_name,
            "size_bytes": shard.stat().st_size,
            "sha256": actual_sha,
            "tensor_count": len(keys),
        })
    if weight_map is not None and observed_keys != set(weight_map):
        raise RuntimeError("weight index does not cover the exact tensor header set")
    return {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "revision": revision,
        "verified": True,
        "shard_count": len(shard_names),
        "tensor_count": len(observed_keys),
        "weight_index_sha256": file_sha256(index_path) if index_path.is_file() else None,
        "shards": shard_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_checkpoint(args.model_dir, args.revision)
    if args.output:
        if args.output.exists():
            raise SystemExit(f"Refusing to overwrite verification evidence: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
