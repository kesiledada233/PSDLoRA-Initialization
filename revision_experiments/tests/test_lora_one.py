from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from revision_experiments.initializers.lora_one import apply_lora_one_initialization, lora_one_factors


class _FactorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(5, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})


class _FactorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _FactorModule()


class LoraOneTests(unittest.TestCase):
    def test_factor_shapes_and_nonzero_b(self):
        torch.manual_seed(7)
        gradient = torch.randn(12, 9)
        a, b, singular_values = lora_one_factors(gradient, rank=3, stable_gamma=128, q=9)
        self.assertEqual(tuple(a.shape), (3, 9))
        self.assertEqual(tuple(b.shape), (12, 3))
        self.assertEqual(tuple(singular_values.shape), (3,))
        self.assertGreater(float(torch.linalg.vector_norm(a)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(b)), 0.0)

    def test_rejects_zero_gradient(self):
        with self.assertRaisesRegex(ValueError, "non-zero leading singular value"):
            lora_one_factors(torch.zeros(4, 4), rank=2, q=4)

    def test_stable_gamma_scales_product(self):
        torch.manual_seed(11)
        gradient = torch.randn(10, 8)
        torch.manual_seed(99)
        a1, b1, _ = lora_one_factors(gradient, rank=2, stable_gamma=1, q=8)
        torch.manual_seed(99)
        a4, b4, _ = lora_one_factors(gradient, rank=2, stable_gamma=4, q=8)
        self.assertTrue(torch.allclose(b4 @ a4, (b1 @ a1) / 4, atol=1e-5, rtol=1e-4))

    def test_apply_moves_each_cpu_accumulator_to_adapter_dtype_and_device(self):
        model = _FactorModel()
        observed = []
        real_factors = lora_one_factors

        def capture(gradient, *args, **kwargs):
            observed.append((gradient.device, gradient.dtype))
            return real_factors(gradient, *args, q=4, **kwargs)

        with patch("revision_experiments.initializers.lora_one.lora_one_factors", side_effect=capture):
            result = apply_lora_one_initialization(
                model, {"proj": torch.randn(4, 5, dtype=torch.float64)}, stable_gamma=128,
            )
        self.assertEqual(observed, [(model.proj.lora_A["default"].weight.device, torch.float32)])
        self.assertEqual(len(result["initialized_modules"]), 1)
        self.assertGreater(result["initialized_modules"][0]["b_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
