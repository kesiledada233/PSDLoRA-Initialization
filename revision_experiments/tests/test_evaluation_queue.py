import json
import sys
import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.aggregate_results import discover_expected_runs
from revision_experiments.scripts.run_evaluation_queue import (
    EvaluationJob,
    build_command,
    build_evaluation_jobs,
    effective_batch_size,
    evaluation_paths,
    validate_batch_equivalence_reports,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "revision_experiments/config"


def _expected(run_id="qwen__cmmlu__qv__peft_default__s1107__n2500", task="cmmlu"):
    return {
        "run_id": run_id,
        "analysis_contexts": ("downstream_2500step",),
        "expected_evaluations": ({
            "checkpoint": 2500,
            "task": task,
            "metrics": ("macro_accuracy",),
            "protocol": {"sample_count": 1},
        },),
    }


def _complete_checkpoint(runs_root: Path, expected: dict) -> Path:
    run_dir = runs_root / expected["run_id"]
    checkpoint = run_dir / "checkpoints/step_002500"
    checkpoint.mkdir(parents=True)
    (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    return run_dir


class EvaluationQueueTests(unittest.TestCase):
    def test_current_amendment_has_98_unique_evaluation_jobs(self):
        expected = discover_expected_runs(CONFIG_DIR)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = build_evaluation_jobs(
                expected, root / "runs", root / "evaluations", project_root=root,
            )
        keys = {(job.run_id, job.checkpoint, job.task) for job in jobs}
        self.assertEqual(len(jobs), 98)  # option B: GSM8K curve keeps only step 1500
        self.assertEqual(len(keys), 98)

    def test_completed_checkpoint_is_ready_and_headline_final_is_p0(self):
        expected = _expected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _complete_checkpoint(root / "runs", expected)
            job = build_evaluation_jobs(
                [expected], root / "runs", root / "evaluations", project_root=root,
            )[0]
        self.assertEqual(job.status, "ready")
        self.assertEqual(job.priority, "P0")
        self.assertEqual(job.stage, "evaluate")

    def test_sharegpt_candidate_stage_resumes_at_judge(self):
        expected = _expected(
            run_id="qwen__sharegpt__qv__peft_default__s1107__n2500",
            task="sharegpt",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _complete_checkpoint(root / "runs", expected)
            paths = evaluation_paths(root / "evaluations", expected["run_id"], 2500, "sharegpt")
            paths["candidate"].parent.mkdir(parents=True)
            paths["candidate"].write_text("{}\n", encoding="utf-8")
            paths["candidate_manifest"].write_text("{}\n", encoding="utf-8")
            job = build_evaluation_jobs(
                [expected], root / "runs", root / "evaluations", project_root=root,
            )[0]
        self.assertEqual(job.status, "ready")
        self.assertEqual(job.stage, "judge")

    def test_partial_sharegpt_candidate_artifacts_fail_closed(self):
        expected = _expected(
            run_id="qwen__sharegpt__qv__peft_default__s1107__n2500",
            task="sharegpt",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _complete_checkpoint(root / "runs", expected)
            paths = evaluation_paths(root / "evaluations", expected["run_id"], 2500, "sharegpt")
            paths["candidate"].parent.mkdir(parents=True)
            paths["candidate"].write_text("{}\n", encoding="utf-8")
            job = build_evaluation_jobs(
                [expected], root / "runs", root / "evaluations", project_root=root,
            )[0]
        self.assertEqual(job.status, "blocked")
        self.assertIn("partial", job.reason)

    def test_command_builder_keeps_mbpp_execution_explicit(self):
        root = Path("/project")
        job = EvaluationJob(
            priority="P1", status="ready", stage="evaluate", run_id="run", checkpoint=2500,
            task="mbpp", run_dir=root / "runs/run", result_path=root / "evaluations/result.json",
            reason="",
        )
        command = build_command(
            job, project_root=root, device="cuda:1", generation_batch_size=4,
            allow_code_execution=True,
        )
        self.assertEqual(command[:2], [sys.executable, str(root / "revision_experiments/scripts/evaluate_checkpoints.py")])
        self.assertIn("--generation-batch-size", command)
        self.assertIn("4", command)
        self.assertIn("--allow-code-execution", command)
        with self.assertRaisesRegex(RuntimeError, "MBPP"):
            build_command(
                job, project_root=root, device="cuda:1", generation_batch_size=4,
                allow_code_execution=False,
            )

    def test_nonfinal_downstream_curve_is_p2(self):
        expected = _expected()
        expected["expected_evaluations"] = ({
            **expected["expected_evaluations"][0], "checkpoint": 500,
        },)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs" / expected["run_id"]
            checkpoint = run_dir / "checkpoints/step_000500"
            checkpoint.mkdir(parents=True)
            (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
            (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
            job = build_evaluation_jobs(
                [expected], root / "runs", root / "evaluations", project_root=root,
            )[0]
        self.assertEqual(job.priority, "P2")

    def test_batch_validation_resolves_per_model_task_combination(self):
        job = EvaluationJob(
            priority="P0", status="ready", stage="evaluate",
            run_id="openpangu__cmmlu__qv__peft_default__s1107__n2500",
            checkpoint=2500, task="cmmlu", run_dir=Path("/runs/run"),
            result_path=Path("/evaluations/result.json"), reason="",
        )
        qwen_job = EvaluationJob(
            **{**job.__dict__, "run_id": "qwen__cmmlu__qv__peft_default__s1107__n2500"}
        )
        self.assertEqual(validate_batch_equivalence_reports([job], 1, []), {})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps({
                "formal_result": False, "purpose": "generation_batch_equivalence_and_throughput",
                "run_id": "openpangu__cmmlu__all_linear__peft_default__s1107__n2500",
                "task": "cmmlu", "all_equivalent": False,
                "measurements": [
                    {"batch_size": 1, "exact_match_to_batch_1": True},
                    {"batch_size": 2, "exact_match_to_batch_1": False, "mismatch_indices": [12]},
                    {"batch_size": 4, "exact_match_to_batch_1": True},
                    {"batch_size": 8, "exact_match_to_batch_1": True},
                ],
            }), encoding="utf-8")
            coverage = validate_batch_equivalence_reports([job, qwen_job], 8, [path])
            # Covered combo runs at the largest validated size under the cap;
            # an uncovered combo and the cap-1 case fall back to batch size 1.
            self.assertEqual(effective_batch_size(job, coverage, 8), 8)
            self.assertEqual(effective_batch_size(job, coverage, 4), 4)
            self.assertEqual(effective_batch_size(job, coverage, 2), 1)
            self.assertEqual(effective_batch_size(qwen_job, coverage, 8), 1)
            # A report whose validated sizes all exceed the cap still degrades to 1.
            small_cap_coverage = validate_batch_equivalence_reports([job], 2, [path])
            self.assertEqual(effective_batch_size(job, small_cap_coverage, 2), 1)
            # Malformed reports still fail closed.
            formal = Path(temporary) / "formal.json"
            formal.write_text(json.dumps({
                "formal_result": True, "run_id": "openpangu__cmmlu__x", "task": "cmmlu",
                "measurements": [{"batch_size": 1, "exact_match_to_batch_1": True}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                validate_batch_equivalence_reports([job], 8, [formal])
            no_baseline = Path(temporary) / "no_baseline.json"
            no_baseline.write_text(json.dumps({
                "formal_result": False,
                "run_id": "openpangu__cmmlu__all_linear__peft_default__s1107__n2500",
                "task": "cmmlu",
                "measurements": [{"batch_size": 4, "exact_match_to_batch_1": True}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lacks the batch-1 baseline"):
                validate_batch_equivalence_reports([job], 4, [no_baseline])

    def test_metric_standard_uses_extracted_answer_validation(self):
        job = EvaluationJob(
            priority="P0", status="ready", stage="evaluate",
            run_id="openpangu__gsm8k__qv__peft_default__s1107__n2500",
            checkpoint=2500, task="gsm8k", run_dir=Path("/runs/run"),
            result_path=Path("/evaluations/result.json"), reason="",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps({
                "formal_result": False,
                "run_id": "openpangu__gsm8k__qv__peft_default__s1107__n2500",
                "task": "gsm8k",
                "measurements": [
                    {"batch_size": 1, "exact_match_to_batch_1": True, "metric_match_to_batch_1": True},
                    {"batch_size": 4, "exact_match_to_batch_1": False,
                     "metric_match_to_batch_1": True, "metric_mismatch_indices": []},
                    {"batch_size": 8, "exact_match_to_batch_1": False,
                     "metric_match_to_batch_1": False, "metric_mismatch_indices": [3]},
                ],
            }), encoding="utf-8")
            coverage = validate_batch_equivalence_reports([job], 8, [path])
            # Exact standard cannot use batched sizes here.
            self.assertEqual(effective_batch_size(job, coverage, 8), 1)
            # Metric standard picks the largest metric-validated size (4, not 8).
            self.assertEqual(
                effective_batch_size(job, coverage, 8, standard="metric"), 4,
            )
            # Legacy reports without metric fields degrade to exact validation.
            legacy = Path(temporary) / "legacy.json"
            legacy.write_text(json.dumps({
                "formal_result": False,
                "run_id": "openpangu__gsm8k__qv__peft_default__s1107__n2500",
                "task": "gsm8k",
                "measurements": [
                    {"batch_size": 1, "exact_match_to_batch_1": True},
                    {"batch_size": 4, "exact_match_to_batch_1": False},
                ],
            }), encoding="utf-8")
            legacy_coverage = validate_batch_equivalence_reports([job], 8, [legacy])
            self.assertEqual(
                effective_batch_size(job, legacy_coverage, 8, standard="metric"), 1,
            )

    def test_judge_only_resume_does_not_require_generation_batch_report(self):
        job = EvaluationJob(
            priority="P0", status="ready", stage="judge",
            run_id="qwen__sharegpt__qv__peft_default__s1107__n2500",
            checkpoint=2500, task="sharegpt", run_dir=Path("/runs/run"),
            result_path=Path("/evaluations/result.json"), reason="",
        )
        self.assertEqual(validate_batch_equivalence_reports([job], 4, []), {})


if __name__ == "__main__":
    unittest.main()
