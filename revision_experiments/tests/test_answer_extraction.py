"""Extractor semantics under model continuation.

Fine-tuned models frequently continue past their own answer, reproducing the
consecutive-example training format. Extractors must read the FIRST answer
marker/block; anything later belongs to spurious regenerated content.
"""

from __future__ import annotations

import unittest

from revision_experiments.scripts.evaluate_checkpoints import (
    extract_choice,
    extract_code,
    extract_number,
)


class FirstMarkerExtractionTests(unittest.TestCase):
    def test_gsm8k_first_hash_marker_wins_over_continuation(self):
        text = (
            "Natalia sold 48/2 = 24 clips.\n#### 72\n\n"
            "Question: A box of pencils costs $3. How much for 100?\n"
            "Answer: 3/24 = 0.125.\n#### 12.5"
        )
        self.assertEqual(extract_number(text), "72")

    def test_gsm8k_falls_back_to_first_number_without_marker(self):
        self.assertEqual(extract_number("total is 15 apples"), "15")
        self.assertIsNone(extract_number("no numbers here"))

    def test_gsm8k_handles_thousands_separators_after_marker(self):
        self.assertEqual(extract_number("#### 70,000"), "70000")

    def test_cmmlu_first_answer_marker_wins_over_continuation(self):
        text = "答案：B\n\n问题：下一题\n答案：D"
        self.assertEqual(extract_choice(text), "B")

    def test_cmmlu_standalone_letter_fallback_takes_first(self):
        self.assertEqual(extract_choice("C is correct"), "C")

    def test_mbpp_first_code_block_wins_over_continuation(self):
        text = (
            "```python\ndef solve():\n    return 1\n```\n\n"
            "Problem: next regenerated problem\nSolution:\n```python\ndef wrong():\n    return 2\n```"
        )
        self.assertIn("return 1", extract_code(text))
        self.assertNotIn("return 2", extract_code(text))

    def test_mbpp_falls_back_to_stripped_text_without_blocks(self):
        self.assertEqual(extract_code("  x = 1  "), "x = 1")

    def test_mbpp_unfenced_continuation_truncates_at_next_problem(self):
        # Fence-less outputs follow the training format (raw code); the model
        # may continue into the next regenerated '# Problem'.
        text = (
            "def solve():\n    return 1\n\n"
            "# Problem: next regenerated problem\n# Solution\ndef wrong():\n    return 2"
        )
        code = extract_code(text)
        self.assertIn("return 1", code)
        self.assertNotIn("return 2", code)
        self.assertNotIn("# Problem", code)

    def test_mbpp_unfenced_answer_before_prompts_closing_fence(self):
        # The frozen 3-shot prompt ends with an opening ```python fence, so a
        # compliant continuation starts directly with code; the first fence in
        # the output is the answer's CLOSING fence, not an opening one.
        text = (
            "def solve(nums):\n    return sum(nums)\n```\n\n"
            "Problem: next regenerated problem\nSolution:\n```python\ndef wrong():\n    return 2\n```"
        )
        self.assertEqual(extract_code(text), "def solve(nums):\n    return sum(nums)")

    def test_mbpp_continuation_code_wins_over_shifted_fence_pairing(self):
        # Regression: naive fence pairing binds the answer's closing fence with
        # the next block's opening fence, extracting the intervening prose.
        text = (
            "def answer():\n    return 42\n```\n"
            "Problem: Write a function to find things.\nSolution:\n```python\ndef other():\n    pass\n```"
        )
        code = extract_code(text)
        self.assertIn("return 42", code)
        self.assertNotIn("Problem:", code)


if __name__ == "__main__":
    unittest.main()


class JsonSafeSanitizerTests(unittest.TestCase):
    def test_bytes_fields_are_decoded_and_probed(self):
        from revision_experiments.scripts.evaluate_checkpoints import _json_safe
        import json as _json
        hits: list = []
        row = {"output": b"raw\xffbytes", "nested": {"stderr": [b"x", "ok"]}, "ok": 1}
        cleaned = _json_safe(row, "$[0]", hits)
        _json.dumps(cleaned)
        self.assertIn("raw\\xffbytes", cleaned["output"])
        self.assertEqual(cleaned["nested"]["stderr"][1], "ok")
        self.assertIn("$[0].output", hits)
        self.assertIn("$[0].nested.stderr[0]", hits)

    def test_non_bytes_values_pass_through_unchanged(self):
        from revision_experiments.scripts.evaluate_checkpoints import _json_safe
        row = {"a": 1, "b": [True, None, {"c": "x"}]}
        self.assertEqual(_json_safe(row), row)
