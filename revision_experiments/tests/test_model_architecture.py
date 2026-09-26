from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from revision_experiments.scripts.verify_model_architecture import (
    bind_model_identity,
    resolve_projection_contract,
    target_module_contract,
    validate_scope_contract,
)


class ModelArchitectureTests(unittest.TestCase):
    def _fixture(self, root: Path, *, omit: str | None = None) -> Path:
        config = {
            "architectures": ["FixtureForCausalLM"], "hidden_size": 4, "intermediate_size": 6,
            "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 2,
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        tensors = {}
        shapes = {
            "q_proj": (4, 4), "k_proj": (2, 4), "v_proj": (2, 4), "o_proj": (4, 4),
            "gate_proj": (6, 4), "up_proj": (6, 4), "down_proj": (4, 6),
        }
        groups = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
                  "mlp": ("gate_proj", "up_proj", "down_proj")}
        for layer in range(2):
            for group, suffixes in groups.items():
                for suffix in suffixes:
                    name = f"model.layers.{layer}.{group}.{suffix}.weight"
                    if name != omit:
                        tensors[name] = torch.zeros(shapes[suffix])
        shard = "model-00001-of-00001.safetensors"
        save_file(tensors, root / shard)
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: shard for name in tensors},
        }), encoding="utf-8")
        return root

    def test_resolves_exact_paths_shapes_and_lora_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = resolve_projection_contract(self._fixture(Path(temporary)), rank=2)
        self.assertEqual(audit["resolved_module_count"], 14)
        self.assertEqual(audit["qv_lora_trainable_parameters"], 56)
        self.assertEqual(audit["all_linear_lora_trainable_parameters"], 232)
        self.assertIn("model.layers.1.mlp.down_proj", audit["resolved_module_paths"])
        qv = target_module_contract(audit, ["q_proj", "v_proj"])
        self.assertEqual(qv["expected_lora_trainable_parameters"], 56)
        self.assertEqual(len(qv["resolved_target_module_paths"]), 4)

    def test_rejects_missing_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root, omit="model.layers.1.mlp.down_proj.weight")
            with self.assertRaisesRegex(RuntimeError, "projection coverage mismatch"):
                resolve_projection_contract(root, rank=2)

    def test_scope_contract_binds_hub_identity_and_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = resolve_projection_contract(self._fixture(Path(temporary)), rank=2)
        local_map = {"models": {"fixture_checkpoint": {
            "hub_id": "owner/fixture", "revision": "a" * 40,
        }}}
        audit = bind_model_identity(audit, "fixture", local_map)
        matrix = {"architecture_contract": {
            "model": "fixture", "hub_id": "owner/fixture", "revision": "a" * 40,
            "config_sha256": audit["config_sha256"],
            "weight_index_sha256": audit["weight_index_sha256"],
            "num_hidden_layers": 2,
            "projection_suffixes": audit["projection_suffixes"],
            "lora_rank": 2,
            "qv_lora_trainable_parameters": 56,
            "all_linear_lora_trainable_parameters": 232,
        }, "all_linear": {"target_modules": audit["projection_suffixes"]}}
        validate_scope_contract(audit, matrix)
        qv = target_module_contract(audit, ["q_proj", "v_proj"])
        self.assertEqual(qv["hub_id"], "owner/fixture")
        self.assertEqual(qv["revision"], "a" * 40)
        matrix["architecture_contract"]["revision"] = "b" * 40
        with self.assertRaisesRegex(RuntimeError, "revision"):
            validate_scope_contract(audit, matrix)


if __name__ == "__main__":
    unittest.main()
