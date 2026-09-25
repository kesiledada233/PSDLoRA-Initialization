from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch

from revision_experiments.initializers.variants import initialize_A
from revision_experiments.scripts.audit_step_zero import (
    adapter_equivalence, target_diagnostics, with_legacy_unmasked_labels,
)


class ToyLora(torch.nn.Module):
    def __init__(self, base: torch.Tensor, method: str):
        super().__init__()
        self.register_buffer("base", base.clone())
        self.lora_A = torch.nn.Parameter(initialize_A((4, base.shape[1]), method, init_seed=1107))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.shape[0], 4))

    def forward(self, values):
        return values @ self.base.T + (values @ self.lora_A.T) @ self.lora_B.T


class StepZeroTests(unittest.TestCase):
    def test_all_revision_initializers_preserve_base_function_at_b_zero(self):
        generator = torch.Generator().manual_seed(7)
        base = torch.randn(6, 8, generator=generator)
        inputs = torch.randn(3, 8, generator=generator)
        expected = inputs @ base.T
        methods = ["iid_matched", "fft_white_a0", "powerlaw_global_a03", "powerlaw_global_a06",
                   "powerlaw_shuffle_a06", "powerlaw_row_a06", "powerlaw_col_a06"]
        for method in methods:
            model = ToyLora(base, method)
            self.assertEqual(torch.count_nonzero(model.lora_B).item(), 0)
            torch.testing.assert_close(model(inputs), expected, atol=1e-6, rtol=1e-6)

    def test_equivalence_aggregates_every_batch(self):
        class ToggleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.disabled = False

            @contextmanager
            def disable_adapter(self):
                previous, self.disabled = self.disabled, True
                try:
                    yield
                finally:
                    self.disabled = previous

            def forward(self, input_ids, attention_mask, labels):
                delta = 0.0 if self.disabled else float(input_ids[0, 0])
                logits = input_ids.float().unsqueeze(-1).repeat(1, 1, 2) + delta
                return SimpleNamespace(logits=logits, loss=logits.mean())

        batches = [
            {"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
             "labels": torch.tensor([[0, 1]])},
            {"input_ids": torch.tensor([[2, 1]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
             "labels": torch.tensor([[2, 1]])},
        ]
        losses, differences = adapter_equivalence(ToggleModel(), batches, torch.device("cpu"))
        self.assertEqual(len(losses), 2)
        self.assertEqual(differences["max_abs_logit_difference"], 2.0)
        self.assertEqual(differences["mean_abs_logit_difference"], 1.0)
        self.assertEqual(differences["max_abs_loss_difference"], 2.0)
        self.assertEqual(differences["mean_abs_loss_difference"], 1.0)

    def test_target_diagnostics_cover_each_batch_and_padding(self):
        batches = [{
            "labels": torch.tensor([[1, -100, -100]]),
            "attention_mask": torch.tensor([[1, 0, 0]]),
        }]
        self.assertEqual(target_diagnostics(batches), [{
            "batch": 0, "valid_target_tokens": 1, "padding_target_tokens": 0,
        }])

    def test_legacy_label_copy_scores_padding(self):
        batches = [{
            "input_ids": torch.tensor([[1, 2, 9, 9]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
            "labels": torch.tensor([[1, 2, -100, -100]]),
        }]
        legacy = with_legacy_unmasked_labels(batches)[0]
        self.assertEqual(int(((legacy["attention_mask"] == 0) & (legacy["labels"] != -100)).sum()), 2)
        self.assertEqual(int(((batches[0]["attention_mask"] == 0) & (batches[0]["labels"] != -100)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
