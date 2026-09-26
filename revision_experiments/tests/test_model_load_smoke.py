from __future__ import annotations

import unittest

import torch

from revision_experiments.scripts.audit_model_load import validate_forward


class ModelLoadSmokeTests(unittest.TestCase):
    def test_accepts_finite_causal_lm_logits(self):
        result = validate_forward(torch.zeros(1, 3, 5), batch_size=1, sequence_length=3)
        self.assertEqual(result["logits_shape"], [1, 3, 5])

    def test_rejects_nonfinite_or_wrong_shape(self):
        with self.assertRaisesRegex(RuntimeError, "shape"):
            validate_forward(torch.zeros(3, 5), batch_size=1, sequence_length=3)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            validate_forward(torch.full((1, 3, 5), float("nan")), batch_size=1, sequence_length=3)


if __name__ == "__main__":
    unittest.main()
