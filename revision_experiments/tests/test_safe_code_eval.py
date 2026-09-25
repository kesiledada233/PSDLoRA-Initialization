from __future__ import annotations

import unittest

from revision_experiments.scripts.safe_code_eval import evaluate_candidate


class SafeCodeEvalTests(unittest.TestCase):
    def test_correct_candidate_passes(self):
        result = evaluate_candidate("def add(a, b): return a + b", "", ["assert add(2, 3) == 5"])
        self.assertTrue(result["passed"])

    def test_wrong_candidate_fails(self):
        result = evaluate_candidate("def add(a, b): return a - b", "", ["assert add(2, 3) == 5"])
        self.assertFalse(result["passed"])

    def test_timeout_is_recorded(self):
        result = evaluate_candidate("while True: pass", "", [], timeout_seconds=1)
        self.assertFalse(result["passed"])
        self.assertTrue(result["timed_out"] or result["returncode"] != 0)

    def test_null_bytes_in_candidate_fail_gracefully(self):
        # Decoder artifacts can embed \x00 in generated code; argv cannot carry
        # null bytes, so the evaluator must not crash. After stripping, the
        # candidate is judged on its remaining source.
        broken = evaluate_candidate("def add(a, b): return a - b\x00", "", ["assert add(2, 3) == 5"])
        self.assertIsInstance(broken, dict)
        self.assertFalse(broken["passed"])
        recoverable = evaluate_candidate("def add(a, b): return a + b\x00", "", ["assert add(2, 3) == 5"])
        self.assertIsInstance(recoverable, dict)
        self.assertTrue(recoverable["passed"])

    def test_name_alias_rescues_single_renamed_function(self):
        # The frozen 3-shot prompt never discloses the canonical name the
        # hidden tests call; a correct solution under a self-chosen name must
        # count as a pass under the conventional harness reading, while the
        # strict (no-alias) outcome stays visible.
        result = evaluate_candidate("def total(a, b): return a + b", "", ["assert add(2, 3) == 5"])
        self.assertTrue(result["passed"])
        self.assertFalse(result["passed_strict"])
        self.assertEqual(result["name_alias"], {"expected": "add", "aliased_from": "total"})

    def test_alias_does_not_rescue_wrong_solution(self):
        result = evaluate_candidate("def total(a, b): return a - b", "", ["assert add(2, 3) == 5"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["passed_strict"])

    def test_alias_not_attempted_with_multiple_definitions(self):
        code = "def helper(a, b):\n    return a + b\ndef total(a, b):\n    return helper(a, b)"
        result = evaluate_candidate(code, "", ["assert add(2, 3) == 5"])
        self.assertFalse(result["passed"])
        self.assertNotIn("name_alias", result)

    def test_alias_not_attempted_for_untested_name(self):
        # A NameError for a name the tests never call (e.g. an internal helper)
        # must not trigger aliasing of the tested entry point.
        code = "def add(a, b):\n    return missing_helper(a, b)"
        result = evaluate_candidate(code, "", ["assert add(2, 3) == 5"])
        self.assertFalse(result["passed"])
        self.assertNotIn("name_alias", result)

    def test_strict_matches_conventional_when_no_alias_needed(self):
        result = evaluate_candidate("def add(a, b): return a + b", "", ["assert add(2, 3) == 5"])
        self.assertTrue(result["passed"])
        self.assertTrue(result["passed_strict"])
        self.assertNotIn("name_alias", result)


if __name__ == "__main__":
    unittest.main()
