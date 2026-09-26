from __future__ import annotations

import unittest
from pathlib import Path

from revision_experiments.scripts.openpangu_cuda_compat import (
    CUDA_BLOCK,
    NPU_IMPORT_BLOCK,
    openpangu_cuda_overlay,
    patched_modeling_source,
)


class OpenPanguCudaCompatTests(unittest.TestCase):
    def test_patch_is_exact_and_removes_literal_npu_import(self):
        patched = patched_modeling_source("before\n" + NPU_IMPORT_BLOCK + "after\n")
        self.assertIn(CUDA_BLOCK, patched)
        self.assertNotIn("import torch_npu", patched)

    def test_rejects_unrecognized_upstream_source(self):
        with self.assertRaisesRegex(RuntimeError, "no longer matches"):
            patched_modeling_source("import torch\n")

    def test_real_overlay_preserves_files_and_patches_only_modeling_source(self):
        model_dir = Path(__file__).resolve().parents[2] / "pretrained_models/openPangu-Embedded-7B-V1.1"
        with openpangu_cuda_overlay(model_dir) as overlay:
            self.assertTrue((overlay / "config.json").is_symlink())
            source = (overlay / "modeling_openpangu_dense.py").read_text(encoding="utf-8")
            self.assertIn(CUDA_BLOCK, source)
            self.assertNotIn("import torch_npu", source)
        self.assertFalse(overlay.exists())


if __name__ == "__main__":
    unittest.main()
