from __future__ import annotations

import unittest
from unittest.mock import patch

from revision_experiments.scripts.evaluate_checkpoints import evaluate_cmmlu, gsm8k_prompt


class EvaluationPromptTests(unittest.TestCase):
    def test_gsm8k_prompt_is_fixed_eight_shot(self):
        train = [{"question": f"train-{index}", "answer": f"work #### {index}"} for index in range(10)]
        prompt = gsm8k_prompt(train, "held-out")
        self.assertEqual(prompt.count("Question:"), 9)
        self.assertIn("train-7", prompt)
        self.assertNotIn("train-8", prompt)
        self.assertTrue(prompt.endswith("Question: held-out\nAnswer:"))

    def test_cmmlu_reports_macro_accuracy_across_subjects(self):
        def row(subject, answer="A"):
            return {"subject": subject, "Question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "Answer": answer}
        train = [row("small"), row("large")]
        test = [row("small"), row("large"), row("large"), row("large")]
        with patch(
            "revision_experiments.scripts.evaluate_checkpoints.generate",
            side_effect=["Answer: A", "Answer: B", "Answer: B", "Answer: B"],
        ):
            metrics, predictions = evaluate_cmmlu(None, None, train, test, "cpu", 4)
        self.assertEqual(metrics["macro_accuracy"], 0.5)
        self.assertEqual([item["sample_index"] for item in predictions], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
