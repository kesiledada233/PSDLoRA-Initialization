from __future__ import annotations

import unittest

from revision_experiments.scripts.sharegpt_judge import build_absolute_prompt, parse_score


class ShareGptJudgeTests(unittest.TestCase):
    def test_prompt_contains_candidate_reference_and_rubric(self):
        prompt = build_absolute_prompt("question", "candidate answer", "reference answer")
        self.assertIn("question", prompt)
        self.assertIn("candidate answer", prompt)
        self.assertIn("reference answer", prompt)
        self.assertIn("Score Rubric", prompt)

    def test_score_parser(self):
        self.assertEqual(parse_score("Feedback: good [RESULT] 4"), 4)
        self.assertEqual(parse_score("Feedback: good [RESULT] (5)"), 5)
        with self.assertRaises(ValueError):
            parse_score("Feedback only")
        with self.assertRaises(ValueError):
            parse_score("[RESULT] 4 and [RESULT] 5")

    def test_score_parser_clamps_out_of_range_zero(self):
        # Prometheus awards "[RESULT] 0" to degenerate candidates; the frozen
        # 1..5 protocol clamps to the minimum valid score (raw text preserved).
        self.assertEqual(parse_score("Feedback: unusable [RESULT] 0"), 1)
        self.assertEqual(parse_score("Feedback: unusable [RESULT] (0)"), 1)


if __name__ == "__main__":
    unittest.main()
