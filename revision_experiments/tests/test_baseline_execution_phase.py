from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.matrix import load_matrix
from revision_experiments.scripts.run_matrix import validate_baseline_execution_phase


ROOT = Path(__file__).resolve().parents[2]
MATRIX = load_matrix(ROOT / "revision_experiments/config/baseline_search_matrix.yaml")


class BaselineExecutionPhaseTests(unittest.TestCase):
    def test_negative_limit_is_rejected_without_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "positive integer"):
                validate_baseline_execution_phase(MATRIX, root, root / "selected.yaml", limit=-1)

    def test_zero_limit_is_rejected_without_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "positive integer"):
                validate_baseline_execution_phase(MATRIX, root, root / "selected.yaml", limit=0)

    def test_overbound_limit_cannot_cross_into_final_runs_without_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "--limit 48"):
                validate_baseline_execution_phase(MATRIX, root, root / "selected.yaml", limit=49)

    def test_no_limit_cannot_cross_into_final_runs_without_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "--limit 48"):
                validate_baseline_execution_phase(MATRIX, root, root / "selected.yaml", limit=None)

    def test_screening_only_limit_is_allowed_without_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase = validate_baseline_execution_phase(MATRIX, root, root / "selected.yaml", limit=48)
            self.assertEqual(phase, "screening")


if __name__ == "__main__":
    unittest.main()
