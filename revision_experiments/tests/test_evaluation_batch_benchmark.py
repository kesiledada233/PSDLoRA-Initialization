import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.benchmark_evaluation_batches import (
    build_benchmark_payload,
    compare_output_sets,
    parse_batch_sizes,
    validate_output_path,
)


class EvaluationBatchBenchmarkTests(unittest.TestCase):
    def test_batch_sizes_are_unique_sorted_and_require_reference_one(self):
        self.assertEqual(parse_batch_sizes("4,1,2,4"), (1, 2, 4))
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            parse_batch_sizes("2,4")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_batch_sizes("1,0")

    def test_output_comparison_reports_exact_mismatch_indices(self):
        self.assertEqual(compare_output_sets(["a", "b"], ["a", "b"]), [])
        self.assertEqual(compare_output_sets(["a", "b", "c"], ["a", "x"]), [1, 2])

    def test_payload_is_explicitly_nonformal_and_fails_if_any_batch_differs(self):
        payload = build_benchmark_payload(
            run_id="run", checkpoint=2500, task="cmmlu", sample_count=2,
            device="cuda:1", measurements=[
                {"batch_size": 1, "seconds": 2.0, "peak_memory_bytes": 10,
                 "outputs": ["A", "B"]},
                {"batch_size": 2, "seconds": 1.0, "peak_memory_bytes": 20,
                 "outputs": ["A", "C"]},
            ],
            metric_evaluator=lambda outputs: outputs,
        )
        self.assertIs(payload["formal_result"], False)
        self.assertIs(payload["all_equivalent"], False)
        self.assertIs(payload["all_metric_equivalent"], False)
        self.assertEqual(payload["measurements"][1]["mismatch_indices"], [1])
        self.assertEqual(payload["measurements"][1]["metric_mismatch_indices"], [1])
        self.assertNotIn("outputs", payload["measurements"][0])

    def test_metric_equivalence_ignores_nonmetric_text_drift(self):
        # Different decoded texts whose extracted metric agrees must count as
        # metric-equivalent (the reported metric is the invariant).
        payload = build_benchmark_payload(
            run_id="run", checkpoint=2500, task="gsm8k", sample_count=2,
            device="cuda:1", measurements=[
                {"batch_size": 1, "seconds": 2.0, "peak_memory_bytes": 10,
                 "outputs": ["reasoning... 42", "so the answer is 7"]},
                {"batch_size": 8, "seconds": 0.5, "peak_memory_bytes": 20,
                 "outputs": ["different reasoning 42", "altogether different 7"]},
            ],
            metric_evaluator=lambda outputs: [text.split()[-1] for text in outputs],
        )
        self.assertIs(payload["all_equivalent"], False)
        self.assertIs(payload["all_metric_equivalent"], True)
        self.assertIs(payload["measurements"][1]["metric_match_to_batch_1"], True)

    def test_payload_without_evaluator_reports_null_metric_equivalence(self):
        payload = build_benchmark_payload(
            run_id="run", checkpoint=2500, task="sharegpt", sample_count=1,
            device="cuda:1", measurements=[
                {"batch_size": 1, "seconds": 2.0, "peak_memory_bytes": 10, "outputs": ["x"]},
                {"batch_size": 2, "seconds": 1.0, "peak_memory_bytes": 20, "outputs": ["y"]},
            ],
        )
        self.assertIsNone(payload["all_metric_equivalent"])
        self.assertIsNone(payload["measurements"][1]["metric_match_to_batch_1"])

    def test_output_must_stay_in_isolated_benchmark_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = root / "revision_experiments/results/smoke/evaluation_benchmarks/report.json"
            self.assertEqual(validate_output_path(accepted, root), accepted.resolve())
            formal = root / "revision_experiments/results/evaluations/report.json"
            with self.assertRaisesRegex(RuntimeError, "benchmark directory"):
                validate_output_path(formal, root)


if __name__ == "__main__":
    unittest.main()
