from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from revision_experiments.scripts.build_gate1_spec import build_spec
from revision_experiments.scripts.schema import file_sha256


class Gate1SpecTests(unittest.TestCase):
    def test_builds_only_when_replay_is_hash_bound_and_in_range(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            replay = root / "replay"
            replay.mkdir()
            (replay / "checkpoint").mkdir()
            for name, content in {
                "config.yaml": "x: 1\n", "raw_loss.jsonl": "{}\n", "data_order.jsonl": "{}\n",
                "checkpoint/adapter_model.safetensors": "weights\n",
            }.items():
                (replay / name).write_text(content, encoding="utf-8")
            summary = {
                "status": "completed_reconstructed_replay", "reusable_as_revision_result": False,
                "config_sha256": file_sha256(replay / "config.yaml"),
                "raw_loss_sha256": file_sha256(replay / "raw_loss.jsonl"),
                "data_order_sha256": file_sha256(replay / "data_order.jsonl"),
                "checkpoint_adapter_weights_sha256": file_sha256(replay / "checkpoint/adapter_model.safetensors"),
                "trapezoid_auc_steps_1_to_500": 12.0,
                "loss_at_microstep_500": 0.5, "first_100_median_loss": 15.0,
            }
            (replay / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (replay / "COMPLETED").write_text("complete\n", encoding="utf-8")
            audit = root / "legacy.json"
            audit.write_text(json.dumps({"reference_ranges": {
                "trapezoid_auc_steps_1_to_500": [10.0, 20.0],
                "loss_at_microstep_500": [0.4, 0.6],
            }}), encoding="utf-8")
            config = root / "gate1.yaml"
            config.write_text(yaml.safe_dump({
                "task": "gsm8k", "replay_seed": 1107,
                "declared_adaptations": ["cuda"], "limitations": ["reconstructed"],
            }), encoding="utf-8")
            with patch("revision_experiments.scripts.build_gate1_spec.ROOT", root):
                spec = build_spec(replay, audit, config, "operator")
        self.assertEqual(spec["schema_version"], 2)
        self.assertTrue(spec["comparison"]["auc500_within_submitted_seed_range"])
        self.assertEqual(spec["comparison"]["configuration_status"], "reconstructed_from_legacy_evidence")


if __name__ == "__main__":
    unittest.main()
