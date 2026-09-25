from __future__ import annotations

import unittest

import torch

from revision_experiments.scripts.training_support import CausalTextDataset, linearize_sharegpt


class FakeTokenizer:
    def __call__(self, text, truncation, max_length, padding, return_tensors):
        del text, truncation, max_length, padding, return_tensors
        return {
            "input_ids": torch.tensor([[7, 8, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }


class DatasetLabelTests(unittest.TestCase):
    def test_padding_is_always_ignored(self):
        row = CausalTextDataset(FakeTokenizer(), ["valid text"], 4)[0]
        self.assertTrue(torch.equal(row["labels"], torch.tensor([7, 8, -100, -100])))
        self.assertEqual(int(((row["labels"] != -100) & (row["attention_mask"] == 0)).sum()), 0)

    def test_actual_sharegpt_paired_turn_shape(self):
        text = linearize_sharegpt({"conversation": [{"human": "question", "assistant": "answer"}]})
        self.assertEqual(text, "human: question\nassistant: answer")


if __name__ == "__main__":
    unittest.main()
