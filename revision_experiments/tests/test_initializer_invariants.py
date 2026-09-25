from __future__ import annotations

import unittest

import torch

from revision_experiments.initializers import initialize_A
from revision_experiments.initializers.audit import (
    collect_lora_b_gradient_statistics,
    collect_lora_parameter_statistics,
)


class _AuditedFactors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
        self.lora_B = torch.nn.Parameter(torch.zeros(3, 2))

    def forward(self, values):
        return (values @ self.lora_A.T) @ self.lora_B.T


class InitializerInvariantTests(unittest.TestCase):
    shape = (16, 512)
    seed = 1107

    def test_deterministic_shape_dtype_and_device(self):
        first = initialize_A(self.shape, "powerlaw_global_a06", init_seed=self.seed, dtype=torch.float64)
        second = initialize_A(self.shape, "powerlaw_global_a06", init_seed=self.seed, dtype=torch.float64)
        self.assertEqual(tuple(first.shape), self.shape)
        self.assertEqual(first.dtype, torch.float64)
        self.assertEqual(first.device.type, "cpu")
        self.assertTrue(torch.equal(first, second))

    def test_scale_matched_controls(self):
        reference = initialize_A(self.shape, "powerlaw_global_a06", init_seed=self.seed)
        for method in ("iid_matched", "fft_white_a0", "powerlaw_row_a06", "powerlaw_col_a06"):
            candidate = initialize_A(self.shape, method, init_seed=self.seed)
            self.assertAlmostEqual(float(candidate.std(unbiased=False)), float(reference.std(unbiased=False)), places=6)
            self.assertAlmostEqual(float(torch.linalg.vector_norm(candidate)), float(torch.linalg.vector_norm(reference)), places=4)

    def test_shuffle_preserves_exact_values_but_changes_order(self):
        reference = initialize_A(self.shape, "powerlaw_global_a06", init_seed=self.seed)
        shuffled = initialize_A(self.shape, "powerlaw_shuffle_a06", init_seed=self.seed)
        self.assertTrue(torch.equal(torch.sort(reference.flatten()).values, torch.sort(shuffled.flatten()).values))
        self.assertFalse(torch.equal(reference, shuffled))

    def test_default_must_use_real_peft_reset(self):
        with self.assertRaisesRegex(ValueError, "PEFT reset"):
            initialize_A(self.shape, "peft_default", init_seed=self.seed)

    def test_inconsistent_target_scales_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            initialize_A(self.shape, "iid_matched", init_seed=self.seed, target_std=1.0, target_fro_norm=1.0)

    def test_instantiated_parameter_audit_covers_a_and_b(self):
        model = _AuditedFactors()
        rows = collect_lora_parameter_statistics(model)
        self.assertEqual({row["factor"] for row in rows}, {"A", "B"})
        a_row = next(row for row in rows if row["factor"] == "A")
        b_row = next(row for row in rows if row["factor"] == "B")
        self.assertAlmostEqual(a_row["frobenius_norm"], 5.0)
        self.assertAlmostEqual(a_row["spectral_norm"], 5.0)
        self.assertEqual(b_row["nonzero_count"], 0)
        self.assertEqual(b_row["spectral_norm"], 0.0)

    def test_initial_b_gradient_audit_uses_real_backward_gradient(self):
        model = _AuditedFactors()
        loss = model(torch.ones(2, 2)).sum()
        loss.backward()
        rows = collect_lora_b_gradient_statistics(model)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["gradient_frobenius_norm"], 0.0)
        self.assertGreater(rows[0]["gradient_nonzero_count"], 0)


if __name__ == "__main__":
    unittest.main()
