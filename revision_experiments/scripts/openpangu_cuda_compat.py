"""Audited CUDA-only overlay for openPangu's unconditional Ascend import."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from revision_experiments.scripts.schema import file_sha256


ORIGINAL_MODELING_SHA256 = "f15eaf322af8a0b0f16b26795eb68af836179413d3dbfa4dc44505db6c8b0d6f"
PATCH_ID = "openpangu-cuda-disable-unconditional-torch-npu-v1"
NPU_IMPORT_BLOCK = """import torch_npu
from torch_npu.contrib import transfer_to_npu
if "910" in torch.npu.get_device_name():
    NPU_ATTN_INFR = True
    print("[INFO] torch_npu detected. Using NPU fused infer attention.")
else:
    NPU_ATTN_INFR = False
"""
CUDA_BLOCK = """# CUDA compatibility overlay: the original checkpoint imports torch_npu unconditionally.
torch_npu = None
NPU_ATTN_INFR = False
"""


def patched_modeling_source(source: str) -> str:
    if source.count(NPU_IMPORT_BLOCK) != 1:
        raise RuntimeError("openPangu NPU import block no longer matches the audited CUDA overlay")
    return source.replace(NPU_IMPORT_BLOCK, CUDA_BLOCK, 1)


def loader_provenance(model_key: str) -> dict:
    if model_key != "openpangu":
        return {"mode": "native_transformers_local_checkpoint"}
    return {
        "mode": "temporary_cuda_source_overlay",
        "patch_id": PATCH_ID,
        "original_modeling_sha256": ORIGINAL_MODELING_SHA256,
        "semantic_scope": "disable_unconditional_torch_npu_import_and_npu_fused_inference_branch",
    }


@contextmanager
def openpangu_cuda_overlay(model_dir: str | Path):
    """Expose untouched weights through a temporary source-only CUDA overlay."""
    model_dir = Path(model_dir).resolve()
    source_path = model_dir / "modeling_openpangu_dense.py"
    if file_sha256(source_path) != ORIGINAL_MODELING_SHA256:
        raise RuntimeError("openPangu modeling source hash changed; CUDA overlay requires re-audit")
    patched = patched_modeling_source(source_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="openpangu-cuda-overlay-") as temporary:
        overlay = Path(temporary)
        for entry in model_dir.iterdir():
            if entry.name.startswith(".") or not entry.is_file() or entry.name == source_path.name:
                continue
            (overlay / entry.name).symlink_to(entry)
        (overlay / source_path.name).write_text(patched, encoding="utf-8")
        yield overlay
