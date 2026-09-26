from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from revision_experiments.scripts.analyze_early_psd import load_gradient_directory


class GradientPsdInputTests(unittest.TestCase):
    def test_reads_logger_memmap_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "p.npy", np.arange(12, dtype=np.float32).reshape(4, 3))
            manifest = {
                "schema_version": 1,
                "max_steps": 3,
                "parameters": [{"name": "layer.lora_A", "indices": [2, 5, 9], "file": "p.npy"}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            series = load_gradient_directory(root, through_step=3)
            self.assertEqual(series["layer.lora_A[5]"], [1.0, 4.0, 7.0, 10.0])

    def test_rejects_unwritten_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.zeros((4, 1), dtype=np.float32)
            values[2, 0] = np.nan
            np.save(root / "p.npy", values)
            manifest = {
                "schema_version": 1,
                "max_steps": 3,
                "parameters": [{"name": "p", "indices": [0], "file": "p.npy"}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing/non-finite"):
                load_gradient_directory(root, through_step=3)


if __name__ == "__main__":
    unittest.main()
