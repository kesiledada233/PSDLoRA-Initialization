"""Tests for the CPU-only rescore path over preserved prediction JSONLs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.evaluate_checkpoints import rescore_predictions_from


def write_source(rows: list[dict]) -> Path:
    temporary = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    temporary.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.close()
    return Path(temporary.name)


class RescorePredictionsTest(unittest.TestCase):
    def test_rescore_corrects_extraction_under_continuation(self):
        source = write_source([
            {"output": "reasoning\n#### 72\n\nQuestion: next\nAnswer: #### 12.5", "expected": "72",
             "predicted": "5", "correct": False},
            {"output": "half of 24 is 12.\n#### 12", "expected": "12", "predicted": "3", "correct": False},
        ])
        metrics, rows = rescore_predictions_from(source, task="gsm8k", formal_count=2)
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual([row["predicted"] for row in rows], ["72", "12"])
        self.assertTrue(all(row["correct"] for row in rows))
        # The immutable generation evidence is preserved verbatim.
        self.assertIn("#### 72", rows[0]["output"])

    def test_rescore_preserves_wrong_answers_as_wrong(self):
        source = write_source([
            {"output": "so the total is 20.\n#### 20", "expected": "18", "predicted": "20", "correct": False},
        ])
        metrics, rows = rescore_predictions_from(source, task="gsm8k", formal_count=1)
        self.assertEqual(metrics["exact_match"], 0.0)
        self.assertFalse(rows[0]["correct"])

    def test_rescore_rejects_wrong_row_count(self):
        source = write_source([{"output": "#### 1", "expected": "1"}])
        with self.assertRaisesRegex(RuntimeError, "expected 3"):
            rescore_predictions_from(source, task="gsm8k", formal_count=3)

    def test_rescore_rejects_missing_fields(self):
        source = write_source([{"output": "#### 1"}])
        with self.assertRaisesRegex(RuntimeError, "output.*expected|expected.*output"):
            rescore_predictions_from(source, task="gsm8k", formal_count=1)

    def test_rescore_mbpp_reextracts_and_reexecutes(self):
        source = write_source([
            {"output": "def add(a, b):\n    return a + b\n```\nProblem: next\nSolution:",
             "task_id": "t/1", "code": "broken extraction", "sample_index": 0},
            {"output": "def sub(a, b):\n    return a - b\n```", "task_id": "t/2",
             "code": "broken extraction", "sample_index": 1},
        ])
        records = [
            {"task_id": "t/1", "test_list": ["assert add(1, 2) == 3"]},
            {"task_id": "t/2", "test_list": ["assert sub(5, 3) == 99"]},
        ]
        metrics, rows = rescore_predictions_from(
            source, task="mbpp", formal_count=2, records=records,
        )
        self.assertEqual(metrics["pass_at_1"], 0.5)
        self.assertIn("return a + b", rows[0]["code"])
        self.assertNotIn("Problem:", rows[0]["code"])
        self.assertTrue(rows[0]["execution"]["passed"])
        self.assertFalse(rows[1]["execution"]["passed"])

    def test_rescore_mbpp_requires_records_and_matching_ids(self):
        source = write_source([{"output": "x = 1", "task_id": "t/1"}])
        with self.assertRaisesRegex(RuntimeError, "records"):
            rescore_predictions_from(source, task="mbpp", formal_count=1)
        records = [{"task_id": "t/OTHER", "test_list": []}]
        with self.assertRaisesRegex(RuntimeError, "task_id mismatch"):
            rescore_predictions_from(source, task="mbpp", formal_count=1, records=records)

    def test_rescore_is_gsm8k_only_for_now(self):
        with self.assertRaisesRegex(RuntimeError, "gsm8k and mbpp only"):
            rescore_predictions_from(Path("/dev/null"), task="cmmlu", formal_count=1)


if __name__ == "__main__":
    unittest.main()
