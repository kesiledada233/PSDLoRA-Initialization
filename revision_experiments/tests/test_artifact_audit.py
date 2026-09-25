from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from revision_experiments.scripts import audit_artifacts


class ArtifactAuditTests(unittest.TestCase):
    def test_partial_checkpoint_is_not_ready(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            index = {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
            (model / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            (model / "model-1.safetensors").write_bytes(b"one")
            (model / "part.incomplete").write_bytes(b"partial")
            with patch.object(audit_artifacts, "ROOT", root):
                result = audit_artifacts.inspect_hf_checkpoint(model)
            self.assertFalse(result["ready"])
            self.assertEqual(result["missing_files"], ["model-2.safetensors"])
            self.assertEqual(result["incomplete_files"], ["part.incomplete"])

    def test_prometheus_readiness_checks_semantics_and_hash_binding(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            audits = Path(directory)
            revision = "66ffb1fc20beebfb60a3964a957d9011723116c5"
            verification = audits / "prometheus_checkpoint_verification.json"
            verification.write_text(json.dumps({
                "verified": True, "revision": revision,
            }), encoding="utf-8")
            verification_hash = audit_artifacts.file_sha256(verification)
            (audits / "prometheus_gpu_load_smoke.json").write_text(json.dumps({
                "passed": True, "checkpoint": f"prometheus-eval/prometheus-7b-v2.0@{revision}",
                "checkpoint_verification_sha256": verification_hash,
            }), encoding="utf-8")
            scoring = audits / "prometheus_scoring_smoke.json"
            scoring.write_text(json.dumps({
                "passed": True, "judge_revision": revision,
                "checkpoint_verification_sha256": verification_hash,
                "parseable_count": 3, "correct_above_incorrect": True,
            }), encoding="utf-8")
            ready, errors = audit_artifacts.prometheus_evidence_ready(audits)
            self.assertTrue(ready, errors)
            scoring.write_text(json.dumps({
                "passed": False, "judge_revision": revision,
                "checkpoint_verification_sha256": verification_hash,
                "parseable_count": 3, "correct_above_incorrect": True,
            }), encoding="utf-8")
            ready, errors = audit_artifacts.prometheus_evidence_ready(audits)
            self.assertFalse(ready)
            self.assertTrue(any("scoring evidence" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
