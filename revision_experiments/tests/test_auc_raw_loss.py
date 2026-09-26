from __future__ import annotations

import unittest

import numpy as np

from revision_experiments.scripts.metrics import raw_trapezoid_auc, trapezoid_auc_at_steps


class RawAucTests(unittest.TestCase):
    def test_trapezoid_not_sum(self):
        losses = [1.0, 2.0, 4.0, 8.0]
        self.assertEqual(raw_trapezoid_auc(losses, 0, 4), 10.5)
        self.assertNotEqual(raw_trapezoid_auc(losses, 0, 4), sum(losses))

    def test_requires_full_raw_window(self):
        with self.assertRaisesRegex(ValueError, "at least 500"):
            raw_trapezoid_auc(np.ones(499))

    def test_rejects_smoothed_nan_input(self):
        losses = np.ones(500)
        losses[2] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            raw_trapezoid_auc(losses)

    def test_sparse_validation_auc_uses_actual_steps(self):
        self.assertEqual(trapezoid_auc_at_steps([0, 250, 500], [2.0, 1.0, 0.0]), 500.0)

    def test_validation_auc_requires_declared_endpoints(self):
        with self.assertRaisesRegex(ValueError, "endpoints"):
            trapezoid_auc_at_steps([25, 500], [2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
