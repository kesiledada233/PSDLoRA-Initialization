from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from revision_experiments.scripts.finalize_step_zero_audit import collect_cases
from revision_experiments.scripts.schema import file_sha256


class StepZeroArtifactTests(unittest.TestCase):
    def _case(self, root: Path, case_id: str):
        model, task = case_id.split("__")
        csv_path = root / f"{case_id}.csv"
        pd.DataFrame([
            {"model": model, "task": task, "seed": 1107, "batch": batch,
             "method": method, "loss": 1.0}
            for method in ("base", "peft_default", "powerlaw_global_a06")
            for batch in range(100)
        ]).to_csv(csv_path, index=False)
        (root / f"{case_id}.json").write_text(json.dumps({
            "schema_version": 4, "case_id": case_id, "model": model, "task": task,
            "batch_losses_file": csv_path.name, "batch_losses_sha256": file_sha256(csv_path),
            "tolerance": 1e-3,
            "equivalence": {
                method: {
                    "b_nonzero": 0, "max_abs_logit_difference": 0.0,
                    "mean_abs_logit_difference": 0.0, "max_abs_loss_difference": 0.0,
                    "mean_abs_loss_difference": 0.0,
                } for method in ("peft_default", "powerlaw_global_a06")
            },
            "batch_target_diagnostics": [
                {"batch": batch, "valid_target_tokens": 8, "padding_target_tokens": 0}
                for batch in range(100)
            ],
            "equivalence_passed": True, "loss_anomaly": False,
        }), encoding="utf-8")

    def test_collects_both_required_cases_with_csv_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._case(root, "openpangu__gsm8k")
            self._case(root, "qwen__cmmlu")
            cases, losses = collect_cases(root)
        self.assertEqual(len(cases), 2)
        self.assertEqual(len(losses), 600)

    def test_rejects_tampered_case_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._case(root, "openpangu__gsm8k")
            self._case(root, "qwen__cmmlu")
            with (root / "qwen__cmmlu.csv").open("a", encoding="utf-8") as target:
                target.write("qwen,cmmlu,1107,1,base,2.0\n")
            with self.assertRaisesRegex(RuntimeError, "CSV binding"):
                collect_cases(root)

    def test_preserves_loss_decimal_strings_when_combining(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._case(root, "openpangu__gsm8k")
            self._case(root, "qwen__cmmlu")
            case_csv = root / "openpangu__gsm8k.csv"
            content = case_csv.read_text(encoding="utf-8").replace(
                ",1.0\n", ",3.5868778228759766\n", 1,
            )
            case_csv.write_text(content, encoding="utf-8")
            case_json = root / "openpangu__gsm8k.json"
            payload = json.loads(case_json.read_text(encoding="utf-8"))
            payload["batch_losses_sha256"] = file_sha256(case_csv)
            case_json.write_text(json.dumps(payload), encoding="utf-8")
            _, losses = collect_cases(root)
        self.assertEqual(losses.iloc[0]["loss"], "3.5868778228759766")


if __name__ == "__main__":
    unittest.main()
