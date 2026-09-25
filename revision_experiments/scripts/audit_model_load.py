#!/usr/bin/env python3
"""Run a one-model, one-forward GPU load smoke and preserve machine-readable evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from revision_experiments.scripts.schema import file_sha256
from revision_experiments.scripts.openpangu_cuda_compat import loader_provenance
from revision_experiments.scripts.training_support import (
    MODEL_IDENTIFIERS,
    MODEL_PATHS,
    load_model,
    load_tokenizer,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT_MODEL_PATHS = {
    **MODEL_PATHS,
    "prometheus": ROOT / "pretrained_models/prometheus-7b-v2.0",
}
AUDIT_MODEL_IDENTIFIERS = {
    **MODEL_IDENTIFIERS,
    "prometheus": "prometheus-eval/prometheus-7b-v2.0@66ffb1fc20beebfb60a3964a957d9011723116c5",
}


def validate_forward(logits: torch.Tensor, *, batch_size: int, sequence_length: int) -> dict:
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (batch_size, sequence_length):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    if logits.shape[-1] <= 0 or not torch.isfinite(logits).all().item():
        raise RuntimeError("model forward produced empty or non-finite logits")
    return {
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_abs_max": float(logits.float().abs().max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(AUDIT_MODEL_PATHS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite load-smoke evidence: {args.output}")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("GPU load smoke requires an available CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    if args.model == "prometheus":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = AUDIT_MODEL_PATHS[args.model]
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
    else:
        tokenizer = load_tokenizer(args.model)
        model = load_model(args.model, device, args.precision)
    encoded = tokenizer("Question: What is 1 + 1?\nAnswer:", return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    forward = validate_forward(
        logits, batch_size=int(input_ids.shape[0]), sequence_length=int(input_ids.shape[1])
    )
    checkpoint_evidence = (
        ROOT / "revision_experiments/results/audits" / f"{args.model}_checkpoint_verification.json"
    )
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint": AUDIT_MODEL_IDENTIFIERS[args.model],
        "checkpoint_verification_sha256": file_sha256(checkpoint_evidence),
        "model_loader_provenance": loader_provenance(args.model),
        "precision": args.precision,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "input_token_count": int(input_ids.numel()),
        **forward,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
