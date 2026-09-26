from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from revision_experiments.scripts.evaluate_checkpoints import (
    canonical_evaluation_paths, load_adapter_model, require_new_artifacts, resolve_evaluation_request,
    smoke_evaluation_paths,
)
from revision_experiments.scripts.matrix import expand_matrix, load_matrix
from revision_experiments.scripts.schema import canonical_hash
from revision_experiments.scripts.training_support import MODEL_IDENTIFIERS


ROOT = Path(__file__).resolve().parents[2]


class _Adapter:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class EvaluationEntrypointContractTests(unittest.TestCase):
    def _completed_run(self, parent: Path) -> tuple[Path, dict]:
        matrix = load_matrix(ROOT / "revision_experiments/config/downstream_matrix.yaml")
        run = next(row for row in expand_matrix(matrix) if row["task"] == "cmmlu")
        run_dir = parent / run["run_id"]
        checkpoint = 500
        adapter = run_dir / "checkpoints" / f"step_{checkpoint:06d}"
        adapter.mkdir(parents=True)
        config = {
            "matrix": matrix["matrix_name"], **run,
            "effective_method": run["method"],
            "effective_learning_rate": run["training"]["learning_rate"],
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        metadata = {
            "run_id": run["run_id"], "git_commit": "0" * 40, "dirty_worktree": False,
            "model_checkpoint": MODEL_IDENTIFIERS[run["model"]],
            "tokenizer": MODEL_IDENTIFIERS[run["model"]], "chat_template_hash": "0" * 64,
            "dataset_split_hash": "0" * 64, "method": run["method"], "seed": run["seed"],
            "init_seed": 1, "data_order_seed": run["seed"], "training_seed": run["seed"],
            "max_steps": run["max_steps"], "target_modules": run["target_modules"],
            "environment": {}, "hardware": {}, "config_hash": canonical_hash(config),
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        for name in ("raw_loss.jsonl", "timing.jsonl", "initialization_stats.json", "summary.json", "COMPLETED"):
            (run_dir / name).write_text("{}\n", encoding="utf-8")
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
        return run_dir, config

    def test_openpangu_adapter_load_uses_audited_shared_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "metadata.json").write_text(json.dumps({
                "model_checkpoint": (
                    "openpangu/openPangu-Embedded-7B-V1.1@"
                    "0ae1841cbd53f5218f2ce5dc63083d5382cfc9f5"
                ),
            }), encoding="utf-8")
            adapter = _Adapter()
            base = object()
            with (
                patch("revision_experiments.scripts.evaluate_checkpoints.load_model", return_value=base) as loader,
                patch("peft.PeftModel.from_pretrained", return_value=adapter) as peft_loader,
            ):
                loaded, model_key = load_adapter_model(run_dir, 500, "cuda:0")
        self.assertIs(loaded, adapter)
        self.assertTrue(adapter.eval_called)
        self.assertEqual(model_key, "openpangu")
        loader.assert_called_once_with("openpangu", torch.device("cuda:0"), "bf16")
        self.assertIn("step_000500", peft_loader.call_args.args[1])

    def test_adapter_load_rejects_unpinned_model_identifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "metadata.json").write_text(
                json.dumps({"model_checkpoint": "openpangu/latest"}), encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "pinned supported"):
                load_adapter_model(run_dir, 500, "cpu")

    def test_canonical_paths_and_no_overwrite_guard(self):
        paths = canonical_evaluation_paths("run", 5, "sharegpt")
        self.assertEqual(paths["candidate"].name, "run__step_000005__sharegpt_candidates.jsonl")
        self.assertEqual(paths["judge"].name, "run__step_000005__sharegpt_judged.jsonl")
        smoke = smoke_evaluation_paths("run", 5, "mbpp", 3)
        self.assertEqual(smoke["result"].name, "run__step_000005__mbpp__first_000003_smoke.json")
        self.assertIn("results/smoke/evaluations", smoke["result"].as_posix())
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "result.json"
            existing.touch()
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                require_new_artifacts({"result": existing}, "result")

    def test_evaluation_request_binds_run_config_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            run_dir, config = self._completed_run(parent)
            with patch("revision_experiments.scripts.evaluate_checkpoints.RUNS_ROOT", parent):
                _, run, _, count = resolve_evaluation_request(run_dir, 500, "cmmlu")
                self.assertEqual(run["run_id"], run_dir.name)
                self.assertEqual(count, 11582)
                config["effective_learning_rate"] = 9.0
                (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "config_hash"):
                    resolve_evaluation_request(run_dir, 500, "cmmlu")

    def test_mbpp_smoke_accepts_only_an_isolated_declared_smoke_run(self):
        matrix = load_matrix(ROOT / "revision_experiments/config/smoke/integration_smoke.yaml")
        run = next(row for row in expand_matrix(matrix) if row["task"] == "mbpp")
        with tempfile.TemporaryDirectory() as temporary:
            smoke_root = Path(temporary) / "smoke-runs"
            run_dir = smoke_root / run["run_id"]
            checkpoint = run_dir / "checkpoints/step_000001"
            checkpoint.mkdir(parents=True)
            config = {
                "matrix": matrix["matrix_name"], **run,
                "effective_method": run["method"],
                "effective_learning_rate": run["training"]["learning_rate"],
                "formal_result": False,
            }
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
            metadata = {
                "run_id": run["run_id"], "git_commit": "0" * 40, "dirty_worktree": False,
                "model_checkpoint": MODEL_IDENTIFIERS[run["model"]],
                "tokenizer": MODEL_IDENTIFIERS[run["model"]], "chat_template_hash": "0" * 64,
                "dataset_split_hash": "0" * 64, "method": run["method"], "seed": run["seed"],
                "init_seed": 1, "data_order_seed": run["seed"], "training_seed": run["seed"],
                "max_steps": run["max_steps"], "target_modules": run["target_modules"],
                "environment": {}, "hardware": {}, "config_hash": canonical_hash(config),
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            for name in ("raw_loss.jsonl", "timing.jsonl", "initialization_stats.json", "summary.json", "COMPLETED"):
                (run_dir / name).write_text("{}\n", encoding="utf-8")
            (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
            with patch("revision_experiments.scripts.evaluate_checkpoints.SMOKE_RUNS_ROOT", smoke_root):
                _, resolved, _, count = resolve_evaluation_request(
                    run_dir, 1, "mbpp", smoke=True,
                )
                self.assertEqual(resolved["case"], "mbpp_adapter_and_executor")
                self.assertEqual(count, 374)
                with self.assertRaisesRegex(RuntimeError, "Formal run directory"):
                    resolve_evaluation_request(run_dir, 1, "mbpp", smoke=False)


if __name__ == "__main__":
    unittest.main()
