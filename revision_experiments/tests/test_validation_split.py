from __future__ import annotations

import unittest

from revision_experiments.scripts.prepare_validation_split import select_indices
from revision_experiments.scripts.training_support import partition_validation_records


class ValidationSplitTests(unittest.TestCase):
    def test_random_selection_is_deterministic_and_unique(self):
        records = [{"value": index} for index in range(30)]
        first = select_indices("gsm8k", records, 7, 20260903)
        second = select_indices("gsm8k", records, 7, 20260903)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_cmmlu_selects_one_per_subject(self):
        records = [
            {"subject": subject, "value": index}
            for subject in ("a", "b", "c") for index in range(4)
        ]
        selected = select_indices("cmmlu", records, 3, 20260903)
        self.assertEqual({records[index]["subject"] for index in selected}, {"a", "b", "c"})

    def test_partition_removes_holdout_from_screening_training(self):
        records = [{"value": index} for index in range(6)]
        training, validation = partition_validation_records("fixture", records, [1, 4], 2)
        self.assertEqual([row["value"] for row in training], [0, 2, 3, 5])
        self.assertEqual([row["value"] for row in validation], [1, 4])

    def test_partition_rejects_out_of_range_indices(self):
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            partition_validation_records("fixture", [{"value": 0}], [1], 1)


if __name__ == "__main__":
    unittest.main()
