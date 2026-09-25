"""Lightweight, exact audits for instantiated LoRA parameters and gradients."""

from __future__ import annotations

import math

import torch


def _small_dimension_spectral_norm(values: torch.Tensor) -> float:
    """Compute the exact matrix 2-norm through the smaller Gram matrix."""

    values = values.detach().float()
    if torch.count_nonzero(values).item() == 0:
        return 0.0
    gram = values @ values.T if values.shape[0] <= values.shape[1] else values.T @ values
    largest = torch.linalg.eigvalsh(gram).max().clamp_min(0.0)
    return float(torch.sqrt(largest))


def collect_lora_parameter_statistics(model: torch.nn.Module) -> list[dict]:
    """Return one finite statistics row for every instantiated LoRA A/B tensor."""

    rows = []
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if parameter.ndim != 2 or ("lora_a" not in lowered and "lora_b" not in lowered):
            continue
        values = parameter.detach().float()
        factor = "A" if "lora_a" in lowered else "B"
        row = {
            "parameter": name,
            "factor": factor,
            "shape": [int(value) for value in parameter.shape],
            "dtype": str(parameter.dtype),
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "variance": float(values.var(unbiased=False)),
            "frobenius_norm": float(torch.linalg.vector_norm(values)),
            "spectral_norm": _small_dimension_spectral_norm(values),
            "max_abs": float(values.abs().max()),
            "nonzero_count": int(torch.count_nonzero(values)),
            "numel": int(values.numel()),
        }
        numeric = (row[key] for key in (
            "mean", "std", "variance", "frobenius_norm", "spectral_norm", "max_abs",
        ))
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"Non-finite LoRA initialization statistics for {name}")
        rows.append(row)
    if not rows or {row["factor"] for row in rows} != {"A", "B"}:
        raise RuntimeError("Expected instantiated LoRA A and B parameters for initialization audit")
    return rows


def collect_lora_b_gradient_statistics(model: torch.nn.Module) -> list[dict]:
    """Record initial B-gradient norms after one fixed global training batch."""

    rows = []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or "lora_b" not in name.lower():
            continue
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"Missing initial LoRA-B gradient for {name}")
        values = gradient.detach().float()
        norm = float(torch.linalg.vector_norm(values))
        maximum = float(values.abs().max())
        if not math.isfinite(norm) or not math.isfinite(maximum):
            raise RuntimeError(f"Non-finite initial LoRA-B gradient for {name}")
        rows.append({
            "parameter": name,
            "shape": [int(value) for value in parameter.shape],
            "gradient_frobenius_norm": norm,
            "gradient_max_abs": maximum,
            "gradient_nonzero_count": int(torch.count_nonzero(values)),
        })
    if not rows:
        raise RuntimeError("Expected LoRA-B gradients for initialization audit")
    return rows
