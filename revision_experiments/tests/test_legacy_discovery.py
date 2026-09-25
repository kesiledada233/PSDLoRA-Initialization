from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.audit_legacy_discovery import build_discovery


class LegacyDiscoveryTests(unittest.TestCase):
    def test_derived_summary_does_not_count_as_raw_gate1_evidence(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "time_to_threshold_results.json").write_text("[]\n", encoding="utf-8")
            (root / "train_openpangu_fda_lora_final.py").write_text("# submitted source\n", encoding="utf-8")
            payload = build_discovery(root)
        self.assertFalse(payload["sufficient_for_gate1_reproduction"])
        self.assertTrue(payload["derived_candidates"])
        self.assertFalse(payload["gate1_required_categories_found"]["raw_per_step_loss_log"])

    def test_detects_all_three_raw_categories_inside_output_tree(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            run = root / "outputs_cmmlu/baseline"
            run.mkdir(parents=True)
            (run / "config.yaml").write_text("seed: 1107\n", encoding="utf-8")
            (run / "training_log.csv").write_text("step,train_loss\n1,2.0\n", encoding="utf-8")
            (run / "adapter_model.safetensors").write_bytes(b"test")
            payload = build_discovery(root)
        self.assertTrue(payload["sufficient_for_gate1_reproduction"])
        self.assertTrue(all(payload["gate1_required_categories_found"].values()))

    def test_rejects_categories_scattered_across_unrelated_runs(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            config_run = root / "outputs_cmmlu/config_only"
            log_run = root / "outputs_cmmlu/log_only"
            checkpoint_run = root / "outputs_cmmlu/checkpoint_only"
            for path in (config_run, log_run, checkpoint_run):
                path.mkdir(parents=True, exist_ok=True)
            (config_run / "config.yaml").write_text("seed: 1107\n", encoding="utf-8")
            (log_run / "training_log.csv").write_text("step,train_loss\n1,2.0\n", encoding="utf-8")
            (checkpoint_run / "adapter_model.safetensors").write_bytes(b"test")
            payload = build_discovery(root)
        self.assertTrue(all(payload["gate1_required_categories_found"].values()))
        self.assertFalse(payload["sufficient_for_gate1_reproduction"])


if __name__ == "__main__":
    unittest.main()
