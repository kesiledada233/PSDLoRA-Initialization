from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from revision_experiments.scripts.execution_gates import (
    EXPECTED_MODEL_REVISIONS,
    REQUIRED_INTEGRATION_SMOKE_CASES,
    validate_execution_gates,
    validate_gate2_step_zero,
    validate_integration_smokes,
    validate_legacy_artifacts,
)
from revision_experiments.scripts.schema import file_sha256


class ExecutionGateTests(unittest.TestCase):
    def _write_csv(self, path: Path, model: str, task: str, batch_count: int = 100) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "task", "seed", "batch", "method", "loss"])
            writer.writeheader()
            for method in ("base", "peft_default", "powerlaw_global_a06"):
                for batch in range(batch_count):
                    writer.writerow({
                        "model": model, "task": task, "seed": 1107,
                        "batch": batch, "method": method, "loss": 2.0,
                    })

    def _fixture(self, root: Path) -> None:
        config_dir = root / "revision_experiments/config"
        audits = root / "revision_experiments/results/audits"
        cases_dir = audits / "step_zero_cases"
        replay_dir = root / "revision_experiments/results/gates/gate1_reproduction"
        legacy = root / "legacy_artifacts/submission"
        for path in (config_dir, audits, cases_dir, replay_dir, legacy):
            path.mkdir(parents=True, exist_ok=True)

        legacy_paths = {
            "configs": legacy / "submitted_config.yaml",
            "logs": legacy / "submitted_loss.jsonl",
            "checkpoints": legacy / "submitted_adapter.bin",
        }
        for category, path in legacy_paths.items():
            path.write_text(category + "\n", encoding="utf-8")
        manifest = legacy / "SHA256SUMS"
        manifest.write_text("".join(
            f"{file_sha256(path)}  {path.relative_to(legacy).as_posix()}\n"
            for path in legacy_paths.values()
        ), encoding="utf-8")
        mapped = {category: [str(path.relative_to(root))] for category, path in legacy_paths.items()}
        (config_dir / "local_repository_map.yaml").write_text(yaml.safe_dump({
            "submitted_artifacts": {
                "import_root": "legacy_artifacts/submission",
                **mapped,
                "manifest": "legacy_artifacts/submission/SHA256SUMS",
            }
        }), encoding="utf-8")

        replay_files = {
            "config": replay_dir / "config.yaml",
            "raw_loss": replay_dir / "raw_loss.jsonl",
            "summary": replay_dir / "summary.json",
            "checkpoint": replay_dir / "adapter_model.safetensors",
        }
        for kind, path in replay_files.items():
            path.write_text(kind + "\n", encoding="utf-8")
        (audits / "gate1_reproduction.json").write_text(json.dumps({
            "schema_version": 2, "gate_id": "gate1-r", "gate_passed": True,
            "provenance_status": "reconstructed_without_exact_submitted_config_or_git_history",
            "reconstructed_git_commit": "a" * 40,
            "submitted_manifest_sha256": file_sha256(manifest),
            "submitted_artifacts": mapped,
            "replay": {
                "model": "openpangu", "method": "peft_default", "task": "cmmlu", "seed": 1107,
                "command": ["python", "submitted_train.py"], "exit_code": 0,
                "evidence_files": [{
                    "kind": kind, "path": str(path.relative_to(root)), "sha256": file_sha256(path),
                } for kind, path in replay_files.items()],
            },
            "comparison": {
                "mode": "single_seed_shape_only", "configuration_status": "reconstructed_from_legacy_evidence",
                "data_order_status": "deterministic_reconstruction", "curve_shape_match": True,
                "auc500_within_submitted_seed_range": None,
                "endpoint_loss_within_submitted_seed_range": None,
                "reused_as_revision_result": False,
            },
            "declared_adaptations": ["test platform adaptation"],
            "limitations": ["test reconstruction limitation"],
            "reviewed_by": "test", "notes": "The deterministic replay matches the submitted single-seed curve.",
        }), encoding="utf-8")

        all_rows = []
        cases = []
        legacy_evidence = audits / "legacy_evidence.json"
        legacy_evidence.write_text("{}\n", encoding="utf-8")
        for case_id in ("openpangu__gsm8k", "qwen__cmmlu"):
            model, task = case_id.split("__", 1)
            verification_path = audits / f"{model}_checkpoint_verification.json"
            verification_path.write_text(json.dumps({
                "verified": True, "revision": EXPECTED_MODEL_REVISIONS[model],
            }), encoding="utf-8")
            (audits / f"{model}_gpu_load_smoke.json").write_text(json.dumps({
                "passed": True,
                "checkpoint": f"owner/{model}@{EXPECTED_MODEL_REVISIONS[model]}",
                "checkpoint_verification_sha256": file_sha256(verification_path),
            }), encoding="utf-8")
            csv_path = cases_dir / f"{case_id}.csv"
            self._write_csv(csv_path, model, task)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                all_rows.extend(csv.DictReader(handle))
            case = {
                "schema_version": 4, "case_id": case_id, "model": model, "task": task,
                "seed": 1107, "batches": 100,
                "checkpoint": f"owner/{model}@{EXPECTED_MODEL_REVISIONS[model]}",
                "checkpoint_verification_sha256": file_sha256(verification_path),
                "precision": "bf16", "tolerance": 1e-3,
                "equivalence": {
                    method: {
                        "b_nonzero": 0, "max_abs_logit_difference": 0.0,
                        "mean_abs_logit_difference": 0.0, "max_abs_loss_difference": 0.0,
                        "mean_abs_loss_difference": 0.0,
                    } for method in ("peft_default", "powerlaw_global_a06")
                },
                "median_losses": {method: 2.0 for method in ("base", "peft_default", "powerlaw_global_a06")},
                "batch_target_diagnostics": [
                    {"batch": batch, "valid_target_tokens": 8, "padding_target_tokens": 0}
                    for batch in range(100)
                ],
                "valid_target_tokens_total": 800, "padding_target_tokens_total": 0,
                "tokenizer_artifacts_hash": "0" * 64, "tokenizer_artifacts": ["tokenizer.json"],
                "label_shift": "Transformers causal-LM internal one-token shift",
                "ignore_index": -100, "loss_reduction": "mean over non-ignored shifted tokens",
                "legacy_label_comparison": ({
                    "status": "resolved_by_padding_label_masking", "same_batches": True,
                    "masked_median_loss": 2.0, "legacy_unmasked_median_loss": 16.0,
                    "median_loss_increase": 14.0, "legacy_padding_targets_scored": 10,
                    "legacy_evidence_sha256": file_sha256(legacy_evidence),
                } if model == "openpangu" else {"status": "not_applicable"}),
                "loss_anomaly_diagnostic": {
                    "trigger_threshold": 10.0, "observed_base_median_loss": 2.0,
                    "status": "resolved_legacy_padding_labels" if model == "openpangu" else "not_triggered",
                    "checks": {"optimizer_updates": 0},
                },
                "equivalence_passed": True, "loss_anomaly": False,
                "batch_losses_file": csv_path.name, "batch_losses_sha256": file_sha256(csv_path),
            }
            (cases_dir / f"{case_id}.json").write_text(json.dumps(case), encoding="utf-8")
            cases.append(case)
        combined_path = audits / "step_zero_batch_losses.csv"
        with combined_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "task", "seed", "batch", "method", "loss"])
            writer.writeheader(); writer.writerows(all_rows)
        (audits / "step_zero_audit.json").write_text(json.dumps({
            "schema_version": 4, "required_cases": ["openpangu__gsm8k", "qwen__cmmlu"],
            "case_count": 2, "gate_passed": True, "cases": cases,
            "combined_batch_losses_sha256": file_sha256(combined_path),
        }), encoding="utf-8")
        smoke_matrix = config_dir / "smoke/integration_smoke.yaml"
        smoke_matrix.parent.mkdir(parents=True, exist_ok=True)
        smoke_matrix.write_text("matrix_name: integration_smoke\npurpose: integration_smoke\n", encoding="utf-8")
        smoke_artifact = root / "revision_experiments/results/smoke/evidence.json"
        smoke_artifact.parent.mkdir(parents=True, exist_ok=True)
        smoke_artifact.write_text("{}\n", encoding="utf-8")
        (audits / "integration_smoke.json").write_text(json.dumps({
            "schema_version": 1, "passed": True, "formal_results": False,
            "matrix": str(smoke_matrix.relative_to(root)), "matrix_sha256": file_sha256(smoke_matrix),
            "cases": [
                {"case": case, "run_id": f"{case}__fixture", "passed": True}
                for case in REQUIRED_INTEGRATION_SMOKE_CASES
            ],
            "artifacts": [{
                "path": str(smoke_artifact.relative_to(root)), "sha256": file_sha256(smoke_artifact),
            }],
        }), encoding="utf-8")

    def test_complete_hash_bound_gate_fixture_passes(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self._fixture(root)
            payload = validate_execution_gates(root)
        self.assertTrue(payload["ready_for_formal_training"], payload["errors"])

    def test_empty_legacy_map_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            config = root / "revision_experiments/config"
            config.mkdir(parents=True)
            (config / "local_repository_map.yaml").write_text(yaml.safe_dump({
                "submitted_artifacts": {
                    "import_root": "legacy_artifacts/submission", "manifest": "legacy_artifacts/submission/SHA256SUMS",
                    "configs": [], "logs": [], "checkpoints": [],
                }
            }), encoding="utf-8")
            errors, _ = validate_legacy_artifacts(root)
        self.assertTrue(any("must contain at least one" in error for error in errors))

    def test_tampered_legacy_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self._fixture(root)
            (root / "legacy_artifacts/submission/submitted_loss.jsonl").write_text("tampered\n", encoding="utf-8")
            errors, _ = validate_legacy_artifacts(root)
        self.assertTrue(any("SHA256 mismatch" in error for error in errors))

    def test_gate2_rejects_short_case_even_if_report_claims_100(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self._fixture(root)
            short = root / "revision_experiments/results/audits/step_zero_cases/qwen__cmmlu.csv"
            self._write_csv(short, "qwen", "cmmlu", batch_count=99)
            errors, _ = validate_gate2_step_zero(root)
        self.assertTrue(any("3 methods x 100" in error or "incomplete" in error for error in errors))

    def test_gate1_and_gate2_authorize_smokes_but_not_formal_runs_without_smoke_report(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self._fixture(root)
            (root / "revision_experiments/results/audits/integration_smoke.json").unlink()
            payload = validate_execution_gates(root)
            prerequisites = validate_execution_gates(root, require_integration_smokes=False)
        self.assertTrue(payload["ready_for_integration_smokes"])
        self.assertFalse(payload["ready_for_formal_training"])
        self.assertTrue(prerequisites["ready_for_integration_smokes"])
        self.assertFalse(prerequisites["ready_for_formal_training"])

    def test_tampered_integration_smoke_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            self._fixture(root)
            (root / "revision_experiments/results/smoke/evidence.json").write_text("tampered\n", encoding="utf-8")
            errors, _ = validate_integration_smokes(root)
        self.assertTrue(any("hash mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
