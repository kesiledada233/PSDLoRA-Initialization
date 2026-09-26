#!/usr/bin/env python3
"""Freeze projection-module and LoRA-parameter contracts from checkpoint headers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml
from safetensors import safe_open

from revision_experiments.scripts.schema import file_sha256
from revision_experiments.scripts.training_support import MODEL_PATHS


ROOT = Path(__file__).resolve().parents[2]
PROJECTION_GROUPS = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}
PROJECTION_SUFFIXES = tuple(name for names in PROJECTION_GROUPS.values() for name in names)
WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


def expected_projection_shapes(config: dict) -> dict[str, tuple[int, int]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    if hidden <= 0 or intermediate <= 0 or heads <= 0 or kv_heads <= 0 or hidden % heads:
        raise RuntimeError("invalid model dimensions in config.json")
    kv_width = hidden // heads * kv_heads
    return {
        "q_proj": (hidden, hidden), "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden), "o_proj": (hidden, hidden),
        "gate_proj": (intermediate, hidden), "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }


def _weight_shapes(model_dir: Path, weight_map: dict[str, str], names: list[str]) -> dict[str, tuple[int, ...]]:
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    shapes = {}
    for shard_name, shard_keys in sorted(by_shard.items()):
        shard = model_dir / shard_name
        if not shard.is_file():
            raise RuntimeError(f"missing checkpoint shard: {shard}")
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    shapes[key] = tuple(handle.get_slice(key).get_shape())
        except Exception as exc:
            raise RuntimeError(f"cannot read safetensors header {shard}: {exc}") from exc
    return shapes


def resolve_projection_contract(model_dir: str | Path, rank: int = 16) -> dict:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid checkpoint config/index under {model_dir}: {exc}") from exc
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise RuntimeError("LoRA rank must be a positive integer")
    layer_count = int(config["num_hidden_layers"])
    matched = {key: WEIGHT_RE.fullmatch(key) for key in weight_map}
    projection_names = sorted(key for key, match in matched.items() if match)
    expected_names = []
    for layer in range(layer_count):
        for group, suffixes in PROJECTION_GROUPS.items():
            expected_names.extend(f"model.layers.{layer}.{group}.{suffix}.weight" for suffix in suffixes)
    if set(projection_names) != set(expected_names):
        missing = sorted(set(expected_names) - set(projection_names))
        extra = sorted(set(projection_names) - set(expected_names))
        raise RuntimeError(f"projection coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    shapes = _weight_shapes(model_dir, weight_map, projection_names)
    expected_shapes = expected_projection_shapes(config)
    resolved_paths = []
    suffix_totals = {suffix: 0 for suffix in PROJECTION_SUFFIXES}
    for name in expected_names:
        match = WEIGHT_RE.fullmatch(name)
        assert match is not None
        suffix = match.group(3)
        if shapes[name] != expected_shapes[suffix]:
            raise RuntimeError(f"projection shape mismatch for {name}: {shapes[name]} != {expected_shapes[suffix]}")
        output_features, input_features = shapes[name]
        suffix_totals[suffix] += rank * (input_features + output_features)
        resolved_paths.append(name.removesuffix(".weight"))
    return {
        "config_sha256": file_sha256(config_path),
        "weight_index_sha256": file_sha256(index_path),
        "architectures": config.get("architectures", []),
        "num_hidden_layers": layer_count,
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_attention_heads": int(config["num_attention_heads"]),
        "num_key_value_heads": int(config.get("num_key_value_heads", config["num_attention_heads"])),
        "projection_suffixes": list(PROJECTION_SUFFIXES),
        "projection_shapes": {key: list(value) for key, value in expected_shapes.items()},
        "resolved_module_paths": resolved_paths,
        "resolved_module_count": len(resolved_paths),
        "lora_rank": rank,
        "lora_trainable_parameters_by_suffix": suffix_totals,
        "qv_lora_trainable_parameters": suffix_totals["q_proj"] + suffix_totals["v_proj"],
        "all_linear_lora_trainable_parameters": sum(suffix_totals.values()),
    }


def bind_model_identity(audit: dict, model_key: str, local_map: dict | None = None) -> dict:
    """Bind a header-derived contract to the exact locally frozen Hub identity."""
    if local_map is None:
        local_map = yaml.safe_load(
            (ROOT / "revision_experiments/config/local_repository_map.yaml").read_text(encoding="utf-8")
        )
    map_key = f"{model_key}_checkpoint"
    try:
        model_entry = local_map["models"][map_key]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"local repository map is missing models.{map_key}") from exc
    bound = dict(audit)
    bound.update({
        "model": model_key,
        "hub_id": model_entry["hub_id"],
        "revision": model_entry["revision"],
    })
    return bound


def validate_scope_contract(audit: dict, scope_matrix: dict) -> None:
    declared = scope_matrix.get("architecture_contract")
    if not isinstance(declared, dict):
        raise RuntimeError("scope_matrix.yaml is missing architecture_contract")
    checks = {
        "model": audit.get("model"),
        "hub_id": audit.get("hub_id"),
        "revision": audit.get("revision"),
        "config_sha256": audit["config_sha256"],
        "weight_index_sha256": audit["weight_index_sha256"],
        "num_hidden_layers": audit["num_hidden_layers"],
        "projection_suffixes": audit["projection_suffixes"],
        "lora_rank": audit["lora_rank"],
        "qv_lora_trainable_parameters": audit["qv_lora_trainable_parameters"],
        "all_linear_lora_trainable_parameters": audit["all_linear_lora_trainable_parameters"],
    }
    mismatches = {key: (declared.get(key), value) for key, value in checks.items() if declared.get(key) != value}
    if mismatches:
        raise RuntimeError(f"scope architecture contract mismatch: {mismatches}")
    if scope_matrix["all_linear"]["target_modules"] != audit["projection_suffixes"]:
        raise RuntimeError("all_linear target_modules do not match resolved projection suffixes")


def target_module_contract(audit: dict, target_modules: list[str]) -> dict:
    """Resolve exact full module paths and the expected LoRA parameter count."""
    if not isinstance(target_modules, list) or not target_modules or len(set(target_modules)) != len(target_modules):
        raise RuntimeError("target_modules must be a nonempty unique list")
    unknown = sorted(set(target_modules) - set(audit["projection_suffixes"]))
    if unknown:
        raise RuntimeError(f"target_modules are not resolved checkpoint projections: {unknown}")
    resolved = [
        path for path in audit["resolved_module_paths"]
        if path.rsplit(".", 1)[-1] in set(target_modules)
    ]
    expected_count = audit["num_hidden_layers"] * len(target_modules)
    if len(resolved) != expected_count:
        raise RuntimeError(f"resolved target path coverage mismatch: {len(resolved)} != {expected_count}")
    trainable = sum(audit["lora_trainable_parameters_by_suffix"][name] for name in target_modules)
    return {
        "model": audit.get("model"),
        "hub_id": audit.get("hub_id"),
        "revision": audit.get("revision"),
        "config_sha256": audit["config_sha256"],
        "weight_index_sha256": audit["weight_index_sha256"],
        "lora_rank": audit["lora_rank"],
        "target_modules": list(target_modules),
        "resolved_target_module_paths": resolved,
        "expected_lora_trainable_parameters": int(trainable),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "revision_experiments/results/audits/model_architecture.json")
    args = parser.parse_args()
    local_map = yaml.safe_load((ROOT / "revision_experiments/config/local_repository_map.yaml").read_text())
    results = {"schema_version": 1, "models": {}}
    for model_key in ("openpangu", "qwen"):
        audit = bind_model_identity(
            resolve_projection_contract(MODEL_PATHS[model_key], rank=16), model_key, local_map
        )
        results["models"][model_key] = audit
    scope = yaml.safe_load((ROOT / "revision_experiments/config/scope_matrix.yaml").read_text())
    validate_scope_contract(results["models"]["openpangu"], scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "openpangu_qv": results["models"]["openpangu"]["qv_lora_trainable_parameters"],
        "openpangu_all_linear": results["models"]["openpangu"]["all_linear_lora_trainable_parameters"],
        "qwen_qv": results["models"]["qwen"]["qv_lora_trainable_parameters"],
        "qwen_all_linear": results["models"]["qwen"]["all_linear_lora_trainable_parameters"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
