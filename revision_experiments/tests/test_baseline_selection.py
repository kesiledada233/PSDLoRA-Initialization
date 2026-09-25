from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from revision_experiments.scripts.baseline_selection import (
    collect_screening_results,
    load_selected_manifest,
    resolve_selected_configuration,
    verify_selection_manifest,
    write_selection_outputs,
)
from revision_experiments.scripts.matrix import expand_matrix, load_matrix
from revision_experiments.scripts.schema import canonical_hash, dataset_split_hash
from revision_experiments.scripts.train_revision import resolve_method, trainable_parameter_counts, validation_loss


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "revision_experiments/config/baseline_search_matrix.yaml"


def _metadata(run: dict, config_hash: str) -> dict:
    return {
        "run_id": run["run_id"], "git_commit": "a" * 40, "dirty_worktree": False,
        "model_checkpoint": "Qwen/Qwen2.5-7B@e25af2efae60472008fbeaf5fb7c4274a87f78d4",
        "tokenizer": "Qwen/Qwen2.5-7B@e25af2efae60472008fbeaf5fb7c4274a87f78d4",
        "chat_template_hash": "b" * 64, "dataset_split_hash": dataset_split_hash(run["task"], True),
        "method": run["method"], "seed": run["seed"], "init_seed": 1,
        "data_order_seed": run["seed"], "training_seed": run["seed"],
        "max_steps": run["max_steps"], "target_modules": run["target_modules"],
        "environment": {"packages": {"peft": "0.17.1"}},
        "hardware": {"device": "fixture", "count": 1}, "config_hash": config_hash,
        "effective_method": run["method"], "effective_learning_rate": float(run["learning_rate"]),
        "validation_selection": None,
    }


def _write_complete_screening_run(root: Path, matrix: dict, run: dict, score: float) -> None:
    run_dir = root / run["run_id"]
    run_dir.mkdir(parents=True)
    config = {
        "matrix": matrix["matrix_name"], **run,
        "effective_method": run["method"], "effective_learning_rate": float(run["learning_rate"]),
    }
    (run_dir / "metadata.json").write_text(json.dumps(_metadata(run, canonical_hash(config))), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    steps = list(range(0, 501, 25))
    evaluated = matrix["validation_split"]["counts"][run["task"]]
    validation_rows = [
        {"step": step, "validation_loss": score, "validation_examples_evaluated": evaluated}
        for step in steps
    ]
    (run_dir / "validation_loss.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in validation_rows), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(json.dumps({
        "validation_auc500": score * 500.0,
        "validation_final_loss": score,
        "validation_guardrail_metric": "validation_loss",
        "validation_guardrail_value": score,
        "validation_examples_evaluated": evaluated,
        "raw_auc500": score * 499.0,
        "steps_logged": 500,
    }), encoding="utf-8")
    (run_dir / "initialization_stats.json").write_text(json.dumps({
        "trainable_parameter_count": 1234,
        "total_parameter_count": 1_000_000,
        "adapter_initialization_seconds": 2.0,
        "gradient_estimation_seconds": 3.0 if run["search_method"] == "lora_one" else 0.0,
        "initialization_seconds_total": 5.0 if run["search_method"] == "lora_one" else 2.0,
        "gradient_batches": 8 if run["search_method"] == "lora_one" else 0,
        "gradient_batch_size": 1 if run["search_method"] == "lora_one" else 0,
        "gradient_max_length": 1024 if run["search_method"] == "lora_one" else 0,
        "stable_gamma": 128 if run["search_method"] == "lora_one" else None,
        "source_commit": matrix["provenance"]["lora_one_commit"] if run["search_method"] == "lora_one" else None,
    }), encoding="utf-8")
    (run_dir / "raw_loss.jsonl").write_text(
        "".join(json.dumps({"step": step, "train_loss": score}) + "\n" for step in range(1, 501)),
        encoding="utf-8",
    )
    (run_dir / "timing.jsonl").write_text(
        "".join(json.dumps({"step": step, "step_time_seconds": 0.1}) + "\n" for step in range(1, 501)),
        encoding="utf-8",
    )
    (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")


class BaselineSelectionTests(unittest.TestCase):
    def setUp(self):
        self.matrix = load_matrix(MATRIX_PATH)
        self.screening = [run for run in expand_matrix(self.matrix) if run["section"] == "screening"]

    def _complete_fixture(self, root: Path) -> None:
        for run in self.screening:
            variant_index = sorted(
                candidate["variant"] for candidate in self.screening
                if candidate["task"] == run["task"] and candidate["search_method"] == run["search_method"]
            ).index(run["variant"])
            _write_complete_screening_run(root, self.matrix, run, score=1.0 + variant_index)

    def test_complete_screening_selects_lowest_validation_auc_and_writes_all_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            rows, selected = collect_screening_results(self.matrix, root)
            self.assertEqual(len(rows), 48)
            self.assertEqual(len([row for row in rows if row["selected"]]), 16)
            self.assertEqual(selected["tasks"]["gsm8k"]["dora"]["learning_rate"], 1.5e-4)
            csv_path = root / "all.csv"
            yaml_path = root / "selected.yaml"
            write_selection_outputs(rows, selected, csv_path, yaml_path)
            self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 49)
            self.assertEqual(yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["selection_uses_official_test"], False)

    def test_missing_screening_run_fails_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            missing = root / self.screening[0]["run_id"]
            for path in missing.iterdir():
                path.unlink()
            missing.rmdir()
            with self.assertRaisesRegex(RuntimeError, "Missing screening runs"):
                collect_screening_results(self.matrix, root)
            self.assertFalse((root / "all.csv").exists())
            self.assertFalse((root / "selected.yaml").exists())

    def test_incomplete_validation_schedule_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            lines = (run_dir / "validation_loss.jsonl").read_text(encoding="utf-8").splitlines()
            (run_dir / "validation_loss.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "validation steps"):
                collect_screening_results(self.matrix, root)

    def test_incomplete_training_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            lines = (run_dir / "raw_loss.jsonl").read_text(encoding="utf-8").splitlines()
            (run_dir / "raw_loss.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "training steps"):
                collect_screening_results(self.matrix, root)

    def test_conflicting_terminal_markers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            (run_dir / "FAILED.json").write_text(json.dumps({"error": "fixture"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "both COMPLETED and FAILED"):
                collect_screening_results(self.matrix, root)

    def test_raw_auc_must_match_complete_raw_training_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["raw_auc500"] += 1.0
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raw_auc500"):
                collect_screening_results(self.matrix, root)

    def test_inconsistent_task_holdout_hashes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["dataset_split_hash"] = "e" * 64
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "dataset split hash"):
                collect_screening_results(self.matrix, root)

    def test_screening_training_contract_is_bound_to_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            config["training"]["lora_rank"] = 8
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "training"):
                collect_screening_results(self.matrix, root)

    def test_consistently_wrong_holdout_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            for run in self.screening:
                run_dir = root / run["run_id"]
                metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
                metadata["dataset_split_hash"] = "e" * 64
                (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "committed frozen split"):
                collect_screening_results(self.matrix, root)

    def test_metadata_config_hash_must_bind_complete_run_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["config_hash"] = "e" * 64
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "config_hash"):
                collect_screening_results(self.matrix, root)

    def test_installed_peft_version_must_match_declared_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            run_dir = root / self.screening[0]["run_id"]
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["environment"]["packages"]["peft"] = "0.16.0"
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "PEFT version"):
                collect_screening_results(self.matrix, root)

    def test_final_configuration_resolves_every_selected_method(self):
        selected = {
            "schema_version": 1,
            "matrix_name": "baseline_fairness_qwen_table3",
            "selection_metric": "validation_auc500",
            "selection_direction": "minimize",
            "selection_uses_official_test": False,
            "screening_trial_count": 48,
            "screening_manifest_hash": "a" * 64,
            "tasks": {"gsm8k": {
                "dora": {"effective_method": "dora", "learning_rate": 0.00015, "selected_run_id": "d"},
                "pissa": {"effective_method": "pissa", "learning_rate": 0.00002, "selected_run_id": "p"},
                "lora_one": {"effective_method": "lora_one", "learning_rate": 0.00005, "selected_run_id": "l"},
                "proposed": {"effective_method": "powerlaw_global_a06", "learning_rate": 0.00005, "selected_run_id": "f"},
            }},
        }
        for method, expected in (
            ("dora", ("dora", 0.00015)),
            ("pissa", ("pissa", 0.00002)),
            ("lora_one", ("lora_one", 0.00005)),
            ("validation_selected_proposed", ("powerlaw_global_a06", 0.00005)),
        ):
            with self.subTest(method=method):
                resolved = resolve_selected_configuration(selected, "gsm8k", method)
                self.assertEqual((resolved["effective_method"], resolved["learning_rate"]), expected)

    def test_selected_configuration_rejects_test_based_manifest(self):
        selected = {
            "schema_version": 1, "matrix_name": "baseline_fairness_qwen_table3",
            "selection_metric": "validation_auc500", "selection_uses_official_test": True,
            "tasks": {},
        }
        with self.assertRaisesRegex(RuntimeError, "official test"):
            resolve_selected_configuration(selected, "gsm8k", "dora")

    def test_selected_configuration_rejects_incomplete_manifest(self):
        selected = {
            "schema_version": 1, "matrix_name": "baseline_fairness_qwen_table3",
            "selection_metric": "validation_auc500", "selection_uses_official_test": False,
            "tasks": {"gsm8k": {"dora": {
                "effective_method": "dora", "learning_rate": 0.00015, "selected_run_id": "d",
            }}},
        }
        with self.assertRaisesRegex(RuntimeError, "Incomplete selection manifest"):
            resolve_selected_configuration(selected, "gsm8k", "dora")

    def test_training_resolution_applies_selected_baseline_learning_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            rows, selected = collect_screening_results(self.matrix, root)
            csv_path = root / "all.csv"
            path = root / "selected.yaml"
            write_selection_outputs(rows, selected, csv_path, path)
            kwargs, method, learning_rate, provenance = resolve_method(
                {"method": "dora", "task": "gsm8k", "selection_method": "dora"},
                path, matrix=self.matrix, runs_root=root,
            )
        self.assertEqual(kwargs, {"use_dora": True})
        self.assertEqual(method, "dora")
        self.assertEqual(learning_rate, 0.00015)
        self.assertIn("__dora__", provenance["selected_run_id"])

    def test_output_pair_refuses_overwrite_and_detects_csv_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            rows, selected = collect_screening_results(self.matrix, root)
            csv_path, yaml_path = root / "all.csv", root / "selected.yaml"
            write_selection_outputs(rows, selected, csv_path, yaml_path)
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                write_selection_outputs(rows, selected, csv_path, yaml_path)
            csv_path.write_text(csv_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                load_selected_manifest(yaml_path, require_pair=True)

    def test_final_consumer_revalidates_current_screening_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            rows, selected = collect_screening_results(self.matrix, root)
            csv_path, yaml_path = root / "all.csv", root / "selected.yaml"
            write_selection_outputs(rows, selected, csv_path, yaml_path)
            run_dir = root / self.screening[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["raw_auc500"] += 1.0
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raw_auc500"):
                verify_selection_manifest(self.matrix, root, yaml_path)

    def test_parameter_count_contract_reports_trainable_and_total(self):
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        model[0].weight.requires_grad_(False)
        model[0].bias.requires_grad_(False)
        self.assertEqual(trainable_parameter_counts(model), (3, 11))

    def test_validation_loss_scores_every_holdout_example(self):
        import types
        import torch

        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.examples_seen = 0

            def forward(self, input_ids):
                self.examples_seen += input_ids.shape[0]
                return types.SimpleNamespace(loss=input_ids.float().mean())

        dataset = [{"input_ids": torch.tensor([value])} for value in range(10)]
        model = CountingModel()
        loss = validation_loss(model, dataset, torch.device("cpu"), batch_size=4)
        self.assertEqual(model.examples_seen, 10)
        self.assertAlmostEqual(loss, 4.5)


if __name__ == "__main__":
    unittest.main()
