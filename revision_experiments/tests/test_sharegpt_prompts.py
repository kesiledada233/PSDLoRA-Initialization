from __future__ import annotations

import unittest

from revision_experiments.scripts.prepare_sharegpt_prompts import first_user_prompt


class ShareGptPromptTests(unittest.TestCase):
    def test_supported_turn_shapes(self):
        self.assertEqual(first_user_prompt({"conversation": [{"from": "human", "value": "hello"}]}), "hello")
        self.assertEqual(first_user_prompt({"messages": [{"role": "user", "content": "question"}]}), "question")
        self.assertEqual(first_user_prompt({"conversation": [{"human": "paired", "assistant": "reply"}]}), "paired")

    def test_missing_user_returns_none(self):
        self.assertIsNone(first_user_prompt({"conversation": [{"from": "gpt", "value": "answer"}]}))


if __name__ == "__main__":
    unittest.main()
