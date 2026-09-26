from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from revision_experiments.scripts.aggregate_results import (
    _time_to_equivalent,
    _validate_mechanism_artifacts,
    _validate_initialization_audit,
    _validate_staged_publication,
    CANONICAL_FIGURE_STEMS,
    CANONICAL_TABLE_NAMES,
    aggregate_seed_metrics,
    build_run_inventory,
    compute_paired_differences,
    discover_expected_runs,
    run_aggregation,
    validate_baseline_search_artifacts,
    validate_evaluation_artifact,
)
from revision_experiments.scripts.baseline_selection import (
    collect_screening_results,
    write_selection_outputs,
)
from revision_experiments.scripts.matrix import expand_matrix, load_matrix
from revision_experiments.scripts.check_completeness import (
    validate_aggregation_artifacts,
    validate_publication_manifest,
    validate_reviewer_tables,
)
from revision_experiments.scripts.schema import canonical_hash, file_sha256
from revision_experiments.tests.test_baseline_selection import _write_complete_screening_run


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "revision_experiments" / "config"


def _expected(*, mechanism: bool = False, evaluations: tuple[dict, ...] = ()) -> dict:
    suffix = "__gradlog" if mechanism else ""
    row = {
        "run_id": f"openpangu__cmmlu__qv__iid_matched__s1107__n500{suffix}",
        "model": "openpangu",
        "task": "cmmlu",
        "method": "iid_matched",
        "seed": 1107,
        "max_steps": 500,
        "target": "qv",
        "matrix_membership": ("mechanism_500step",) if mechanism else ("core_500step",),
        "expected_checkpoints": (100, 250, 500),
        "expected_evaluations": evaluations,
        "mechanism_required": mechanism,
        "expected_dataset_split_hashes": ("c" * 64,),
    }
    run = {
        "run_id": row["run_id"], "model": row["model"], "task": row["task"],
        "method": row["method"], "seed": row["seed"], "max_steps": row["max_steps"],
        "target": row["target"], "training": {
            "precision": "bf16", "learning_rate": 5e-5, "scheduler": "cosine",
            "gradient_accumulation_steps": 8,
            "target_modules": ["q_proj", "v_proj"],
        }, "target_modules": ["q_proj", "v_proj"],
    }
    row["expected_configurations"] = ({
        "matrix": row["matrix_membership"][0], **run,
        "effective_method": row["method"], "effective_learning_rate": 5e-5,
    },)
    return row


def _write_complete_run(root: Path, expected: dict, *, usable_checkpoints: bool = True) -> Path:
    run_dir = root / expected["run_id"]
    run_dir.mkdir(parents=True)
    config = dict(expected["expected_configurations"][0])
    metadata = {
        "run_id": expected["run_id"], "git_commit": "a" * 40, "dirty_worktree": False,
        "model_checkpoint": "openpangu/checkpoint@" + "1" * 40,
        "tokenizer": "openpangu/tokenizer@" + "2" * 40,
        "chat_template_hash": "b" * 64, "dataset_split_hash": "c" * 64,
        "method": expected["method"], "seed": expected["seed"], "init_seed": 7,
        "data_order_seed": expected["seed"], "training_seed": expected["seed"],
        "max_steps": expected["max_steps"], "target_modules": ["q_proj", "v_proj"],
        "effective_method": expected["method"], "effective_learning_rate": 5e-5,
        "validation_selection": None,
        "model_loader_provenance": {
            "mode": "temporary_cuda_source_overlay",
            "patch_id": "openpangu-cuda-disable-unconditional-torch-npu-v1",
            "original_modeling_sha256": "f15eaf322af8a0b0f16b26795eb68af836179413d3dbfa4dc44505db6c8b0d6f",
            "semantic_scope": "disable_unconditional_torch_npu_import_and_npu_fused_inference_branch",
        },
        "environment": {"python": "fixture"}, "hardware": {"device": "fixture", "count": 1},
        "config_hash": canonical_hash(config),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "raw_loss.jsonl").write_text(
        "".join(json.dumps({"step": step, "train_loss": 1.0}) + "\n" for step in range(1, 501)),
        encoding="utf-8",
    )
    (run_dir / "timing.jsonl").write_text(
        "".join(json.dumps({"step": step, "step_time_seconds": 0.1}) + "\n" for step in range(1, 501)),
        encoding="utf-8",
    )
    (run_dir / "initialization_stats.json").write_text(json.dumps({
        "adapter_initialization_seconds": 2.0,
        "gradient_estimation_seconds": 0.0,
        "initialization_seconds_total": 2.0,
        "trainable_parameter_count": 10,
        "total_parameter_count": 100,
        "matrix_statistics_schema_version": 1,
        "initialization_audit_seconds": 0.25,
        "matrix_statistics": [
            {
                "parameter": "model.q_proj.lora_A.default.weight", "factor": "A",
                "shape": [1, 2], "dtype": "torch.float32", "mean": 1.0,
                "std": 0.0, "variance": 0.0, "frobenius_norm": 2 ** 0.5,
                "spectral_norm": 2 ** 0.5, "max_abs": 1.0,
                "nonzero_count": 2, "numel": 2,
            },
            {
                "parameter": "model.q_proj.lora_B.default.weight", "factor": "B",
                "shape": [2, 1], "dtype": "torch.float32", "mean": 0.0,
                "std": 0.0, "variance": 0.0, "frobenius_norm": 0.0,
                "spectral_norm": 0.0, "max_abs": 0.0,
                "nonzero_count": 0, "numel": 2,
            },
        ],
        "initial_gradient_audit": {
            "schema_version": 1,
            "batch_scope": "one_complete_global_batch_before_optimizer_step_1",
            "microbatch_count": 8,
            "ordered_batch_sha256": "d" * 64,
            "audit_seconds": 0.5,
            "lora_b": [{
                "parameter": "model.q_proj.lora_B.default.weight", "shape": [2, 1],
                "gradient_frobenius_norm": 1.0, "gradient_max_abs": 1.0,
                "gradient_nonzero_count": 2,
            }],
        },
    }), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({
        "raw_auc500": 499.0, "steps_logged": 500,
        "train_examples": 20, "validation_examples": 0, "test_examples": 5,
    }), encoding="utf-8")
    for checkpoint in expected["expected_checkpoints"]:
        checkpoint_dir = run_dir / "checkpoints" / f"step_{checkpoint:06d}"
        checkpoint_dir.mkdir(parents=True)
        if usable_checkpoints:
            (checkpoint_dir / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 1,
                "lora_alpha": 1, "target_modules": ["q_proj", "v_proj"],
            }) + "\n", encoding="utf-8")
            save_file({
                "model.q_proj.lora_A.weight": torch.ones(1, 2),
                "model.q_proj.lora_B.weight": torch.zeros(2, 1),
            }, checkpoint_dir / "adapter_model.safetensors")
    (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return run_dir


class AggregationPublicationTests(unittest.TestCase):
    def test_initialization_audit_rejects_missing_layerwise_and_gradient_evidence(self):
        errors = _validate_initialization_audit(
            {"matrix_statistics_schema_version": 1, "matrix_statistics": []},
            {"effective_method": "iid_matched", "training": {"gradient_accumulation_steps": 8}},
        )
        self.assertTrue(any("non-empty" in error for error in errors))

    def test_inventory_lists_missing_runs_and_preserves_duplicate_matrix_membership(self):
        expected = discover_expected_runs(CONFIG_DIR)
        self.assertEqual(len(expected), 231)
        duplicate = next(row for row in expected if row["run_id"] == "qwen__cmmlu__qv__peft_default__s1107__n2500")
        self.assertEqual(
            duplicate["matrix_membership"],
            ("baseline_fairness_qwen_table3", "downstream_2500step"),
        )
        self.assertEqual(
            duplicate["analysis_contexts"],
            ("baseline_fairness_final", "downstream_2500step"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            inventory = build_run_inventory(expected, temp / "runs", temp / "evaluations", project_root=temp)
        self.assertEqual(len(inventory), 231)
        self.assertEqual(set(inventory["status"]), {"missing"})
        self.assertTrue(inventory["metadata_sha256"].isna().all())

        cmmlu = next(row for row in expected if row["run_id"] == "qwen__cmmlu__qv__peft_default__s1107__n2500")
        requirement = next(item for item in cmmlu["expected_evaluations"] if item["checkpoint"] == 2500)
        self.assertEqual(requirement["protocol"]["sample_count"], 11582)
        self.assertEqual(requirement["protocol"]["sample_selection"], "full_frozen_official_test_order")
        self.assertEqual(requirement["evaluation_config"]["sample_manifest"]["sample_count"], 11582)

    def test_inventory_recomputes_raw_auc_and_cumulative_timing(self):
        expected = _expected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _write_complete_run(root / "runs", expected)
            inventory = build_run_inventory([expected], root / "runs", root / "evaluations", project_root=root)
            row = inventory.iloc[0]
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["raw_auc500_recomputed"], 499.0)
            self.assertAlmostEqual(row["training_seconds_recomputed"], 50.0)
            self.assertEqual(row["metadata_sha256"], file_sha256(run_dir / "metadata.json"))
            self.assertEqual(set(json.loads(row["checkpoint_sha256"])), {"100", "250", "500"})

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["raw_auc500"] = 500.0
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            invalid = build_run_inventory([expected], root / "runs", root / "evaluations", project_root=root).iloc[0]
            self.assertEqual(invalid["status"], "invalid")
            self.assertIn("raw_auc500", invalid["failure_reason"])

    def test_completed_mechanism_run_requires_both_gradient_artifacts(self):
        expected = _expected(mechanism=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_complete_run(root / "runs", expected)
            row = build_run_inventory([expected], root / "runs", root / "evaluations", project_root=root).iloc[0]
        self.assertEqual(row["status"], "invalid")
        self.assertIn("gradients/manifest.json", row["failure_reason"])
        self.assertIn("gradient_diagnostics.jsonl", row["failure_reason"])

    def test_complete_run_config_must_belong_to_declared_matrix(self):
        expected = _expected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _write_complete_run(root / "runs", expected)
            config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            config["matrix"] = "rogue_matrix"
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["config_hash"] = canonical_hash(config)
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            row = build_run_inventory([expected], root / "runs", root / "evaluations", project_root=root).iloc[0]
        self.assertEqual(row["status"], "invalid")
        self.assertIn("matrix", row["failure_reason"])

    def test_complete_run_rejects_rehashed_nonprotocol_training_config(self):
        expected = _expected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _write_complete_run(root / "runs", expected)
            config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            config["training"]["learning_rate"] = 9e-3
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["config_hash"] = canonical_hash(config)
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            row = build_run_inventory([expected], root / "runs", root / "evaluations", project_root=root).iloc[0]
        self.assertEqual(row["status"], "invalid")
        self.assertIn("expected configuration fingerprint", row["failure_reason"])

    def test_empty_named_checkpoint_is_not_available(self):
        expected = _expected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_complete_run(root / "runs", expected, usable_checkpoints=False)
            row = build_run_inventory([expected], root / "runs", root / "evaluations", project_root=root).iloc[0]
        self.assertEqual(row["status"], "invalid")
        self.assertEqual(json.loads(row["available_checkpoints"]), [])
        self.assertEqual(json.loads(row["missing_checkpoints"]), [100, 250, 500])

    def test_malformed_nonempty_checkpoint_weights_are_not_available(self):
        expected = _expected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_complete_run(root / "runs", expected)
            checkpoint = root / "runs" / expected["run_id"] / "checkpoints/step_000100/adapter_model.safetensors"
            checkpoint.write_bytes(b"fixture")
            row = build_run_inventory([expected], root / "runs", root / "evaluations", project_root=root).iloc[0]
        self.assertEqual(row["status"], "invalid")
        self.assertNotIn(100, json.loads(row["available_checkpoints"]))
        self.assertIn(100, json.loads(row["missing_checkpoints"]))

    def test_mechanism_validation_requires_exact_snapshot_parameter_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            gradients = run_dir / "gradients"; gradients.mkdir()
            parameters = []
            for layer in (0, 5, 9):
                for target in ("q_proj", "v_proj"):
                    for adapter in ("A", "B"):
                        name = f"model.layers.{layer}.self_attn.{target}.lora_{adapter}.default.weight"
                        filename = name.replace(".", "_") + ".npy"
                        np.save(gradients / filename, np.ones((501, 2), dtype=np.float32))
                        parameters.append({"name": name, "indices": [0, 1], "file": filename})
            (gradients / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "max_steps": 500, "coordinate_seed": 7,
                "selected_layers": [0, 5, 9], "parameters": parameters,
            }), encoding="utf-8")
            diagnostics = []
            # Deliberately omit configured snapshot step 10.
            for step in (0,):
                for layer in (0, 5, 9):
                    for target in ("q_proj", "v_proj"):
                        diagnostics.append({
                            "kind": "effective_base_weight_subspace", "step": step,
                            "parameter": f"model.layers.{layer}.self_attn.{target}.base_layer.weight",
                            "microbatch_backward_calls": 8, "gradient_frobenius_norm": 1.0,
                            "gradient_capture_ratio": 0.5, "gradient_times_a_transpose_norm": 1.0,
                            "a_singular_values": [1.0], "a_effective_rank": 1,
                            "principal_angles_degrees": [45.0],
                        })
                for parameter in parameters:
                    diagnostics.append({
                        "kind": "lora_parameter_gradient", "step": step,
                        "parameter": parameter["name"], "gradient_frobenius_norm": 1.0,
                        "parameter_frobenius_norm": 1.0,
                    })
            (run_dir / "gradient_diagnostics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in diagnostics), encoding="utf-8"
            )
            errors = _validate_mechanism_artifacts(run_dir, 500, {
                "logging": {"coordinates_per_matrix": 2, "coordinate_seed": 7, "full_matrix_steps": [0, 10]},
                "target_modules": ("q_proj", "v_proj"), "gradient_accumulation_steps": 8,
                "num_hidden_layers": 10, "hidden_size": 4, "num_attention_heads": 2,
                "num_key_value_heads": 1, "lora_rank": 1,
            })
        self.assertTrue(any("exact snapshot" in error for error in errors))

    def test_mechanism_validation_rejects_wrong_middle_layer_and_seeded_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary); gradients = run_dir / "gradients"; gradients.mkdir()
            parameters = []
            for layer in (0, 4, 9):  # sorted and in range, but configured middle is layer 5
                for target in ("q_proj", "v_proj"):
                    for adapter in ("A", "B"):
                        name = f"model.layers.{layer}.self_attn.{target}.lora_{adapter}.default.weight"
                        filename = name.replace(".", "_") + ".npy"
                        np.save(gradients / filename, np.ones((501, 2), dtype=np.float32))
                        parameters.append({"name": name, "indices": [0, 1], "file": filename})
            # The indices are sorted/in-range, but are not the RNG(seed=7) selection.
            (gradients / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "max_steps": 500, "coordinate_seed": 7,
                "selected_layers": [0, 4, 9], "parameters": parameters,
            }), encoding="utf-8")
            (run_dir / "gradient_diagnostics.jsonl").write_text(
                json.dumps({"kind": "placeholder"}) + "\n", encoding="utf-8"
            )
            errors = _validate_mechanism_artifacts(run_dir, 500, {
                "logging": {"coordinates_per_matrix": 2, "coordinate_seed": 7, "full_matrix_steps": [0]},
                "target_modules": ("q_proj", "v_proj"), "gradient_accumulation_steps": 8,
                "num_hidden_layers": 10, "hidden_size": 4, "num_attention_heads": 2,
                "num_key_value_heads": 1, "lora_rank": 1,
            })
        self.assertTrue(any("first/middle/last" in error for error in errors))
        self.assertTrue(any("seed-selected" in error for error in errors))

    def test_baseline_publication_sources_are_recomputed_exactly(self):
        matrix = load_matrix(CONFIG_DIR / "baseline_search_matrix.yaml")
        screening = [run for run in expand_matrix(matrix) if run["section"] == "screening"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for run in screening:
                variant_index = sorted(
                    candidate["variant"] for candidate in screening
                    if candidate["task"] == run["task"] and candidate["search_method"] == run["search_method"]
                ).index(run["variant"])
                _write_complete_screening_run(root, matrix, run, score=1.0 + variant_index)
            rows, selected = collect_screening_results(matrix, root)
            csv_path, selected_path = root / "all.csv", root / "selected.yaml"
            write_selection_outputs(rows, selected, csv_path, selected_path)
            checked_rows, _ = validate_baseline_search_artifacts(matrix, root, csv_path, selected_path)
            self.assertEqual(len(checked_rows), 48)
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("1.0", "9.0", 1), encoding="utf-8")
            manifest = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
            manifest["all_trials_sha256"] = file_sha256(csv_path)
            selected_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly match"):
                validate_baseline_search_artifacts(matrix, root, csv_path, selected_path)

    def test_evaluation_hash_and_identity_are_verified(self):
        protocol = {
            "metric": "macro_accuracy", "prompt": "fixed_5shot_by_subject", "decoding": "greedy",
            "sample_count": 2, "sample_selection": "first_n_frozen_test",
        }
        requirement = {
            "checkpoint": 500, "task": "cmmlu", "metrics": ("macro_accuracy",),
            "protocol": protocol, "dataset_split_hash": "c" * 64,
        }
        requirement["evaluation_config"] = {
            "schema_version": 1, "run_id": "openpangu__cmmlu__qv__iid_matched__s1107__n500",
            "checkpoint": 500, "task": "cmmlu", "protocol": protocol,
            "dataset_split_hash": "c" * 64,
        }
        requirement["sample_set_hash"] = canonical_hash({
            "dataset_split_hash": "c" * 64, "selection": "first_n_frozen_test", "count": 2,
        })
        expected = _expected(evaluations=(requirement,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"sample_index": 0, "correct": True}) + "\n" +
                json.dumps({"sample_index": 1, "correct": True}) + "\n",
                encoding="utf-8",
            )
            result = root / f"{expected['run_id']}__step_000500__cmmlu.json"
            payload = {
                "run_id": expected["run_id"], "checkpoint": 500, "task": "cmmlu",
                "metrics": {"macro_accuracy": 1.0}, "sample_count": 2,
                "prediction_artifact": "predictions.jsonl", "prediction_sha256": file_sha256(predictions),
                "sample_set_hash": requirement["sample_set_hash"],
                "evaluation_config": requirement["evaluation_config"],
                "evaluation_seconds": 4.0,
                "evaluation_config_hash": canonical_hash(requirement["evaluation_config"]),
            }
            result.write_text(json.dumps(payload), encoding="utf-8")
            checked = validate_evaluation_artifact(result, expected["expected_evaluations"][0], expected["run_id"], root)
            self.assertEqual(checked["metrics"]["macro_accuracy"], 1.0)
            predictions.write_text(json.dumps({"sample_index": 0, "correct": False}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prediction_sha256"):
                validate_evaluation_artifact(result, expected["expected_evaluations"][0], expected["run_id"], root)
            predictions.write_text("not-json\n", encoding="utf-8")
            payload["prediction_sha256"] = file_sha256(predictions)
            result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid JSONL"):
                validate_evaluation_artifact(result, expected["expected_evaluations"][0], expected["run_id"], root)

    def test_evaluation_rejects_extra_metric_and_wrong_formal_sample_count(self):
        protocol = {"metric": "score", "sample_count": 2, "sample_selection": "first_n_frozen_test"}
        requirement = {
            "checkpoint": 500, "task": "cmmlu", "metrics": ("score",), "protocol": protocol,
            "dataset_split_hash": "c" * 64, "sample_set_hash": "d" * 64,
        }
        requirement["evaluation_config"] = {
            "schema_version": 1, "run_id": "run", "checkpoint": 500, "task": "cmmlu",
            "protocol": protocol, "dataset_split_hash": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "prediction.jsonl"
            prediction.write_text(json.dumps({"sample_index": 0}) + "\n", encoding="utf-8")
            result = root / "run__step_000500__cmmlu.json"
            result.write_text(json.dumps({
                "run_id": "run", "checkpoint": 500, "task": "cmmlu",
                "metrics": {"score": 1.0, "undeclared": 2.0}, "sample_count": 1,
                "prediction_artifact": "prediction.jsonl", "prediction_sha256": file_sha256(prediction),
                "sample_set_hash": "d" * 64, "evaluation_config": requirement["evaluation_config"],
                "evaluation_config_hash": canonical_hash(requirement["evaluation_config"]),
                "evaluation_seconds": 1.0,
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "declared metric schema|sample_count"):
                validate_evaluation_artifact(result, requirement, "run", root)

    def test_evaluation_inventory_is_discovered_even_when_run_is_missing(self):
        protocol = {"metric": "score", "sample_count": 1, "sample_selection": "first_n_frozen_test"}
        requirement = {
            "checkpoint": 500, "task": "cmmlu", "metrics": ("score",), "protocol": protocol,
            "dataset_split_hash": "c" * 64,
        }
        requirement["sample_set_hash"] = canonical_hash({
            "dataset_split_hash": "c" * 64, "selection": "first_n_frozen_test", "count": 1,
        })
        requirement["evaluation_config"] = {
            "schema_version": 1, "run_id": "openpangu__cmmlu__qv__iid_matched__s1107__n500",
            "checkpoint": 500, "task": "cmmlu", "protocol": protocol, "dataset_split_hash": "c" * 64,
        }
        expected = _expected(evaluations=(requirement,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluations = root / "evaluations"; evaluations.mkdir()
            predictions = root / "prediction.jsonl"
            predictions.write_text(json.dumps({"sample_index": 0}) + "\n", encoding="utf-8")
            name = f"{expected['run_id']}__step_000500__cmmlu.json"
            (evaluations / name).write_text(json.dumps({
                "run_id": expected["run_id"], "checkpoint": 500, "task": "cmmlu",
                "metrics": {"score": 1.0}, "sample_count": 1,
                "prediction_artifact": "prediction.jsonl", "prediction_sha256": file_sha256(predictions),
                "sample_set_hash": requirement["sample_set_hash"],
                "evaluation_config": requirement["evaluation_config"],
                "evaluation_config_hash": canonical_hash(requirement["evaluation_config"]),
                "evaluation_seconds": 1.0,
            }), encoding="utf-8")
            row = build_run_inventory([expected], root / "runs", evaluations, project_root=root).iloc[0]
        self.assertEqual(row["status"], "missing")
        self.assertEqual(json.loads(row["available_evaluations"]), [name])
        self.assertEqual(json.loads(row["missing_evaluations"]), [])

    def test_statistics_use_seed_as_n_and_keep_negative_paired_differences(self):
        rows = []
        values = {
            "peft_default": [2.0, 3.0, 4.0],
            "iid_matched": [1.0, 2.0, 3.0],
            "powerlaw_global_a06": [0.0, 1.0, 2.0],
        }
        for method, scores in values.items():
            for seed, value in zip((1107, 123, 42), scores):
                rows.append({
                    "model": "openpangu", "task": "cmmlu", "method": method, "seed": seed,
                    "endpoint": "step_000500", "metric": "raw_auc500", "metric_direction": "lower",
                    "unit": "loss_step", "value": value, "run_id": f"{method}-{seed}",
                    "metadata_sha256": str(seed) * 8,
                })
        frame = pd.DataFrame(rows)
        summary = aggregate_seed_metrics(frame)
        proposed = summary[summary["method"] == "powerlaw_global_a06"].iloc[0]
        self.assertEqual(proposed["n_runs"], 3)
        self.assertEqual(proposed["mean"], 1.0)
        self.assertEqual(proposed["sd"], 1.0)
        paired = compute_paired_differences(frame)
        peft = paired[(paired["method"] == "powerlaw_global_a06") & (paired["reference_method"] == "peft_default")]
        self.assertEqual(peft["difference"].tolist(), [-2.0, -2.0, -2.0])
        self.assertEqual(peft["n_runs"].unique().tolist(), [3])

    def test_paired_comparison_refuses_nonexact_seed_join(self):
        frame = pd.DataFrame([
            {"model": "m", "task": "t", "method": method, "seed": seed,
             "endpoint": "step_000500", "metric": "score", "metric_direction": "higher",
             "value": 1.0, "run_id": f"{method}-{seed}", "metadata_sha256": "a" * 64}
            for method, seeds in (("peft_default", (1, 2)), ("candidate", (1, 2, 3)))
            for seed in seeds
        ])
        with self.assertRaisesRegex(RuntimeError, "exact paired seed join"):
            compute_paired_differences(frame)

    def test_statistics_do_not_merge_distinct_experimental_contexts(self):
        frame = pd.DataFrame([
            {"analysis_context": context, "model": "m", "task": "t", "method": "candidate",
             "seed": 1, "endpoint": "step_000500", "metric": "raw_auc500",
             "metric_direction": "lower", "unit": "loss_step", "value": value,
             "run_id": context, "metadata_sha256": "a" * 64}
            for context, value in (("core_500step", 2.0), ("mechanism_500step", 9.0))
        ])
        summary = aggregate_seed_metrics(frame)
        self.assertEqual(len(summary), 2)
        self.assertEqual(set(summary["analysis_context"]), {"core_500step", "mechanism_500step"})

    def test_time_to_equivalent_preserves_target_and_unreached_candidate_provenance(self):
        rows = []
        for method, observations in (
            ("peft_default", ((2500, 0.8),)),
            ("candidate", ((500, 0.3), (2500, 0.7))),
        ):
            for step, value in observations:
                rows.append({
                    "analysis_context": "downstream_2500step", "model": "m", "task": "t",
                    "seed": 42, "metric": "accuracy", "metric_direction": "higher", "method": method,
                    "endpoint": f"step_{step:06d}", "value": value, "training_seconds": step / 10,
                    "initialization_seconds": 2.0, "training_plus_initialization_seconds": step / 10 + 2,
                    "evaluation_seconds": 1.0, "run_id": f"{method}-{step}",
                    "metadata_sha256": ("a" if method == "peft_default" else "b") * 64,
                })
        result = _time_to_equivalent(pd.DataFrame(rows))
        candidate = result[result["method"] == "candidate"].iloc[0]
        self.assertFalse(candidate["reached"])
        self.assertEqual(candidate["target_run_id"], "peft_default-2500")
        self.assertEqual(candidate["run_id"], "candidate-2500")
        self.assertEqual(candidate["final_observed_endpoint"], "step_002500")
        self.assertEqual(json.loads(candidate["candidate_run_ids"]), ["candidate-500", "candidate-2500"])

    def test_incomplete_sources_emit_inventory_but_no_scientific_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "aggregate"
            figures = root / "figures"
            result = run_aggregation(
                config_dir=CONFIG_DIR, runs_root=root / "runs", evaluations_root=root / "evaluations",
                audits_root=root / "audits", output_root=output, figures_root=figures, project_root=root,
            )
            self.assertFalse(result["reviewer_artifacts_generated"])
            self.assertTrue((output / "run_inventory.csv").is_file())
            self.assertEqual(len(pd.read_csv(output / "run_inventory.csv")), 231)
            self.assertFalse((output / "reviewer_seed_metrics.csv").exists())
            self.assertEqual(list(figures.glob("*")), [])

    def test_staging_rejects_missing_canonical_figure_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "tables"; tables.mkdir()
            for name in CANONICAL_TABLE_NAMES:
                (tables / name).write_text("x\n1\n", encoding="utf-8")
            figures = []
            for stem in CANONICAL_FIGURE_STEMS:
                for extension in ("svg", "pdf", "provenance.json"):
                    path = root / f"{stem}.{extension}"; path.write_text("fixture", encoding="utf-8")
                    figures.append(path)
            with self.assertRaisesRegex(RuntimeError, "every canonical figure format"):
                _validate_staged_publication(tables, figures)

    def test_operational_inventory_is_reproducible_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "aggregate"
            figures = root / "figures"
            run_aggregation(
                config_dir=CONFIG_DIR, runs_root=root / "runs", evaluations_root=root / "evaluations",
                audits_root=root / "audits", output_root=output, figures_root=figures, project_root=root,
            )
            errors = validate_aggregation_artifacts(
                CONFIG_DIR, root / "runs", root / "evaluations", output, figures, root,
                require_publication=False,
            )
            self.assertEqual(errors, [])
            with (output / "run_inventory.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            errors = validate_aggregation_artifacts(
                CONFIG_DIR, root / "runs", root / "evaluations", output, figures, root,
                require_publication=False,
            )
            self.assertTrue(any("not reproducible" in error or "hash" in error for error in errors))

    def test_publication_manifest_rejects_untracked_source_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.jsonl"
            output = root / "reviewer_seed_metrics.csv"
            source.write_text("{}\n", encoding="utf-8")
            output.write_text("metric,value\nscore,1\n", encoding="utf-8")
            inventory = pd.DataFrame([{"run_id": "expected", "status": "complete"}])
            manifest_path = root / "reviewer_tables_manifest.json"
            manifest_path.write_text(json.dumps({
                "schema_version": 1, "experimental_unit": "run/seed",
                "inventory_sha256": "a" * 64,
                "source_artifacts": [{
                    "path": "raw.jsonl", "sha256": file_sha256(source), "run_id": "rogue",
                }],
                "output_artifacts": [{
                    "path": "reviewer_seed_metrics.csv", "sha256": file_sha256(output), "run_id": None,
                }],
            }), encoding="utf-8")
            errors = validate_publication_manifest(manifest_path, inventory, root, expected_inventory_sha="a" * 64)
        self.assertTrue(any("untracked run" in error for error in errors))

    def test_reviewer_tables_reject_missing_seed_without_inflating_n(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame([
                {"analysis_context": "core_500step", "model": "m", "task": "t", "method": "candidate",
                 "seed": seed, "endpoint": "step_000500", "metric": "raw_auc500",
                 "run_id": f"run-{seed}", "metadata_sha256": "a" * 64}
                for seed in (1107, 123)
            ]).to_csv(root / "reviewer_seed_metrics.csv", index=False)
            inventory = pd.DataFrame([
                {"run_id": f"run-{seed}", "status": "complete"} for seed in (1107, 123)
            ])
            errors = validate_reviewer_tables(root, inventory)
        self.assertTrue(any("exactly three seeds" in error for error in errors))
        self.assertTrue(any("missing canonical publication tables" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
