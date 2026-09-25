from __future__ import annotations

import math
import unittest

import torch

from revision_experiments.scripts.gradient_logger import coordinate_invariant_diagnostics


class SubspaceDiagnosticTests(unittest.TestCase):
    def test_exact_row_space_has_full_capture_and_zero_angles(self):
        a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        gradient = torch.tensor([[2.0, 1.0, 0.0], [-1.0, 3.0, 0.0]])
        result = coordinate_invariant_diagnostics(gradient, a)
        self.assertAlmostEqual(result["gradient_capture_ratio"], 1.0, places=6)
        self.assertEqual(result["a_effective_rank"], 2)
        self.assertTrue(all(abs(angle) < 0.05 for angle in result["principal_angles_degrees"]))

    def test_orthogonal_gradient_has_zero_capture(self):
        a = torch.tensor([[1.0, 0.0, 0.0]])
        gradient = torch.tensor([[0.0, 0.0, 4.0]])
        result = coordinate_invariant_diagnostics(gradient, a)
        self.assertAlmostEqual(result["gradient_capture_ratio"], 0.0, places=6)
        self.assertTrue(math.isfinite(result["principal_angles_degrees"][0]))
        self.assertAlmostEqual(result["principal_angles_degrees"][0], 90.0, places=4)


if __name__ == "__main__":
    unittest.main()
