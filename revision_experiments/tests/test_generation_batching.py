"""Tests for batched greedy generation (implementation detail of the frozen protocol).

Batching must be order-preserving, slice only the continuation under left
padding, fall back to an eos pad token, and keep the batch-size-1 path exactly
on the legacy single-stream implementation.
"""

from __future__ import annotations

import unittest

import torch

from revision_experiments.scripts.evaluate_checkpoints import generate, generate_all, generate_batch


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 7
    eos_token_id = 1
    padding_side = "right"

    def __call__(self, prompts, return_tensors="pt", padding=False, truncation=False, max_length=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        sequences = [[1] + [ord(ch) % 90 + 10 for ch in prompt] for prompt in prompts]
        width = max(len(seq) for seq in sequences)
        pad_id = 7
        ids, mask = [], []
        for seq in sequences:
            pad = width - len(seq)
            ids.append([pad_id] * pad + seq)
            mask.append([0] * pad + [1] * len(seq))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((int(token) - 10) % 90 + 33) for token in ids.tolist() if int(token) >= 10)


class FakeModel:
    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        mask = kwargs.get("attention_mask")
        new_tokens = kwargs["max_new_tokens"]
        rows = []
        for index, row in enumerate(input_ids):
            # Real models mask padded positions; the continuation must depend
            # only on unpadded prompt content, never on the padding amount.
            active = row.tolist() if mask is None else [
                token for token, keep in zip(row.tolist(), mask[index].tolist()) if keep
            ]
            marker = 10 + int(sum(active)) % 90
            rows.append(list(row.tolist()) + [marker] * new_tokens)
        return torch.tensor(rows)


def expected_marker(prompt: str) -> str:
    return chr(int(sum([1] + [ord(ch) % 90 + 10 for ch in prompt])) % 90 + 33)


class GenerateBatchTest(unittest.TestCase):
    def test_batch_preserves_order_and_slices_continuation(self):
        prompts = ["short", "a much longer prompt for padding", "mid length"]
        outputs = generate_batch(FakeModel(), FakeTokenizer(), prompts, "cpu", 4)
        self.assertEqual(len(outputs), 3)
        # Each row's continuation is its own content-derived marker, in order,
        # and contains only the new tokens (prompt tokens are sliced away).
        for prompt, output in zip(prompts, outputs):
            self.assertEqual(output, expected_marker(prompt) * 4)

    def test_batch_size_one_matches_legacy_single_stream(self):
        prompts = ["alpha", "beta", "gamma"]
        legacy = [generate(FakeModel(), FakeTokenizer(), prompt, "cpu", 3) for prompt in prompts]
        batched_off = generate_all(FakeModel(), FakeTokenizer(), prompts, "cpu", 3, batch_size=1)
        self.assertEqual(legacy, batched_off)

    def test_generate_all_chunks_preserve_order(self):
        prompts = [f"prompt {i}" for i in range(7)]
        outputs = generate_all(FakeModel(), FakeTokenizer(), prompts, "cpu", 2, batch_size=3)
        self.assertEqual(len(outputs), 7)
        for prompt, output in zip(prompts, outputs):
            self.assertEqual(output, expected_marker(prompt) * 2)
        # Any chunking reproduces the single-stream outputs exactly.
        single = generate_all(FakeModel(), FakeTokenizer(), prompts, "cpu", 2, batch_size=1)
        self.assertEqual(outputs, single)

    def test_length_bucketing_preserves_output_order_for_heterogeneous_prompts(self):
        prompts = ["tiny", "a medium length prompt", "x" * 80, "mid", "y" * 50, "zz", "tail"]
        outputs = generate_all(FakeModel(), FakeTokenizer(), prompts, "cpu", 3, batch_size=2)
        for prompt, output in zip(prompts, outputs):
            self.assertEqual(output, expected_marker(prompt) * 3)
        single = generate_all(FakeModel(), FakeTokenizer(), prompts, "cpu", 3, batch_size=1)
        self.assertEqual(outputs, single)

    def test_missing_pad_token_falls_back_to_eos(self):
        class NoPad(FakeTokenizer):
            pad_token = None

        tokenizer = NoPad()
        generate_batch(FakeModel(), tokenizer, ["x", "yy"], "cpu", 2)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)

    def test_left_padding_requested(self):
        side_seen = {}

        class SideRecorder(FakeTokenizer):
            def __call__(self, prompts, **kwargs):
                side_seen["during"] = self.padding_side
                return super().__call__(prompts, **kwargs)

        tokenizer = SideRecorder()
        self.assertEqual(tokenizer.padding_side, "right")
        generate_batch(FakeModel(), tokenizer, ["abc", "de"], "cpu", 2)
        self.assertEqual(side_seen["during"], "left")
        self.assertEqual(tokenizer.padding_side, "right")


if __name__ == "__main__":
    unittest.main()
