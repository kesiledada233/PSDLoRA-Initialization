"""Audited LoRA-One gradient initialization adapter.

The equations and defaults follow YuanheZ/LoRA-One commit
b797a10dcb818e9553b57fa060baa7b00df4484d (run_exp.py and
conf/init/gradient.yaml). LoRA-One intentionally initializes both A and B to
non-zero values; step-zero functional equivalence is therefore not expected.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader


OFFICIAL_COMMIT = "b797a10dcb818e9553b57fa060baa7b00df4484d"


def lora_one_factors(
    gradient: torch.Tensor,
    rank: int,
    stable_gamma: float = 128.0,
    q: int = 512,
    niter: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (A, B, singular_values) using the official stable formula."""
    if gradient.ndim != 2:
        raise ValueError("LoRA-One requires a two-dimensional weight gradient")
    if rank <= 0 or rank > min(gradient.shape):
        raise ValueError(f"Invalid rank {rank} for gradient shape {tuple(gradient.shape)}")
    if stable_gamma <= 0:
        raise ValueError("stable_gamma must be positive")
    q = min(max(int(q), rank), min(gradient.shape))
    u, singular_values, v = torch.svd_lowrank(-gradient.float(), q=q, niter=int(niter))
    leading = singular_values[0]
    if not torch.isfinite(leading) or leading <= 0:
        raise ValueError("LoRA-One requires a finite, non-zero leading singular value")
    root = torch.sqrt(singular_values[:rank])
    normalizer = torch.sqrt(leading) * float(stable_gamma) ** 0.5
    b = (u[:, :rank] * root.unsqueeze(0)) / normalizer
    a = (root.unsqueeze(1) * v[:, :rank].T) / normalizer
    return a.contiguous(), b.contiguous(), singular_values[:rank].contiguous()


def _target_weight_parameters(model, target_modules: Iterable[str]) -> dict[str, torch.nn.Parameter]:
    targets = tuple(str(item) for item in target_modules)
    selected: dict[str, torch.nn.Parameter] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if name.split(".")[-1] in targets and isinstance(weight, torch.nn.Parameter):
            selected[name] = weight
    if not selected:
        raise ValueError(f"No base weights matched target modules {targets}")
    return selected


def estimate_target_gradients(
    model,
    dataset,
    target_modules: Iterable[str],
    device: torch.device,
    batches: int = 8,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Average gradients for only the base weights needed by LoRA-One.

    Restricting ``requires_grad`` to target weights produces the same target
    derivatives while avoiding a CPU copy of every 7B-model gradient.
    """
    if batches <= 0 or batch_size <= 0:
        raise ValueError("batches and batch_size must be positive")
    selected = _target_weight_parameters(model, target_modules)
    original_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    accumulated = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for name, parameter in selected.items()
    }
    observed = 0
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in selected.values():
            parameter.requires_grad_(True)
        model.train()
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
        for batch_index, batch in enumerate(loader):
            if batch_index >= int(batches):
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            model.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
            for name, parameter in selected.items():
                if parameter.grad is None:
                    raise RuntimeError(f"No gradient produced for LoRA-One target {name}")
                accumulated[name].add_(parameter.grad.detach().float().cpu())
            observed += 1
        if observed != int(batches):
            raise RuntimeError(f"Requested {batches} LoRA-One batches but observed {observed}")
        return {name: value.div_(observed) for name, value in accumulated.items()}
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_flags[name])


def _base_name(peft_module_name: str) -> str:
    prefix = "base_model.model."
    return peft_module_name[len(prefix):] if peft_module_name.startswith(prefix) else peft_module_name


@torch.no_grad()
def apply_lora_one_initialization(
    model,
    named_gradients: dict[str, torch.Tensor],
    stable_gamma: float = 128.0,
) -> dict:
    initialized = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or "default" not in lora_a or "default" not in lora_b:
            continue
        base_name = _base_name(name)
        if base_name not in named_gradients:
            raise KeyError(f"Missing LoRA-One base gradient for {base_name}")
        a_parameter = lora_a["default"].weight
        b_parameter = lora_b["default"].weight
        # Accumulation stays on CPU to bound accelerator memory, while the
        # frozen official implementation performs its randomized SVD on CUDA.
        # Move one matrix at a time to the adapter device; otherwise a real 7B
        # run would accidentally execute every q=512 decomposition on CPU.
        gradient = named_gradients[base_name].to(
            device=a_parameter.device, dtype=torch.float32, non_blocking=False,
        )
        a, b, singular_values = lora_one_factors(
            gradient, rank=a_parameter.shape[0], stable_gamma=stable_gamma,
        )
        a_parameter.copy_(a.to(device=a_parameter.device, dtype=a_parameter.dtype))
        b_parameter.copy_(b.to(device=b_parameter.device, dtype=b_parameter.dtype))
        initialized.append({
            "module": name,
            "gradient_name": base_name,
            "a_shape": list(a_parameter.shape),
            "b_shape": list(b_parameter.shape),
            "leading_singular_value": float(singular_values[0]),
            "a_norm": float(torch.linalg.vector_norm(a_parameter.float())),
            "b_norm": float(torch.linalg.vector_norm(b_parameter.float())),
        })
    if len(initialized) != len(named_gradients):
        raise RuntimeError(
            f"Initialized {len(initialized)} LoRA modules from {len(named_gradients)} target gradients"
        )
    return {
        "method": "lora_one",
        "source_repository": "https://github.com/YuanheZ/LoRA-One",
        "source_commit": OFFICIAL_COMMIT,
        "gradient_batches": 8,
        "gradient_batch_size": 1,
        "gradient_max_length": 1024,
        "stable_gamma": float(stable_gamma),
        "step_zero_equivalence_expected": False,
        "initialized_modules": initialized,
    }
