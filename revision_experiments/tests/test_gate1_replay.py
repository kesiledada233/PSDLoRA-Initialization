from __future__ import annotations

import unittest

import torch

from revision_experiments.scripts.run_gate1_replay import LegacyUnmaskedDataset, format_legacy_record


class ToyTokenizer:
    last_text = None

    def __call__(self, text, **kwargs):
        self.last_text = text
        del kwargs
        return {
            "input_ids": torch.tensor([[7, 8, 2, 2]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }


class Gate1ReplayTests(unittest.TestCase):
    def test_legacy_dataset_intentionally_scores_padding(self):
        dataset = LegacyUnmaskedDataset(
            ToyTokenizer(), [{"question": "a sufficiently long question", "answer": "42"}], "gsm8k", 4,
        )
        row = dataset[0]
        self.assertTrue(torch.equal(row["labels"], row["input_ids"]))
        self.assertEqual(int(((row["attention_mask"] == 0) & (row["labels"] != -100)).sum()), 2)
        self.assertEqual(int(row["source_index"]), 0)

    def test_gsm8k_text_matches_submitted_chinese_template(self):
        tokenizer = ToyTokenizer()
        record = {"question": "a sufficiently long question", "answer": "42"}
        LegacyUnmaskedDataset(tokenizer, [record], "gsm8k", 4)
        self.assertEqual(tokenizer.last_text, "问题：a sufficiently long question\n解答：42")
        self.assertEqual(format_legacy_record("gsm8k", record), tokenizer.last_text)


if __name__ == "__main__":
    unittest.main()
