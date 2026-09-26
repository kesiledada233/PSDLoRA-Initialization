from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from revision_experiments.scripts.schema import file_sha256
from revision_experiments.scripts.verify_hf_checkpoint import verify_checkpoint


class HfCheckpointVerificationTests(unittest.TestCase):
    REVISION = "a" * 40

    def _fixture(self, root: Path) -> Path:
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        shard = root / "model-00001-of-00001.safetensors"
        save_file({"model.layer.weight": torch.zeros(2, 2)}, shard)
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"model.layer.weight": shard.name},
        }), encoding="utf-8")
        metadata = root / ".cache/huggingface/download" / f"{shard.name}.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"{self.REVISION}\n{file_sha256(shard)}\n0\n", encoding="utf-8")
        return root

    def test_verifies_revision_sha_and_exact_index_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = verify_checkpoint(self._fixture(Path(temporary)), self.REVISION)
        self.assertTrue(report["verified"])
        self.assertEqual(report["tensor_count"], 1)

    def test_rejects_incomplete_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._fixture(Path(temporary))
            (root / ".cache/huggingface/download/partial.incomplete").write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "incomplete downloads"):
                verify_checkpoint(root, self.REVISION)

    def test_rejects_wrong_pinned_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "revision/LFS"):
                verify_checkpoint(root, "b" * 40)


if __name__ == "__main__":
    unittest.main()
