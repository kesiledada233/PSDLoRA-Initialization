"""Deterministic FP32 initializer synthesis independent of training RNG state."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from .registry import METHOD_REGISTRY, require_registered


def _validate_shape(shape: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(value) for value in shape)
    if len(values) != 2 or min(values) <= 0:
        raise ValueError(f"LoRA A must have a positive 2-D shape, got {values}")
    return values


def _fft_signal(length: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    if length < 2:
        return np.zeros(length, dtype=np.float32)
    frequencies = np.fft.rfftfreq(length)
    amplitude = np.zeros_like(frequencies, dtype=np.float64)
    positive = frequencies > 0
    amplitude[positive] = frequencies[positive] ** (-float(alpha) / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.size)
    spectrum = amplitude * np.exp(1j * phase)
    if length % 2 == 0:
        spectrum[-1] = complex(float(spectrum[-1].real), 0.0)
    signal = np.fft.irfft(spectrum, n=length).astype(np.float32)
    signal -= signal.mean(dtype=np.float64)
    return signal


def _standardize(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    values = values - values.mean()
    std = values.std(unbiased=False)
    if not torch.isfinite(std) or std <= 0:
        raise ValueError("Cannot standardize a zero-variance initializer")
    return values / std


def _rescale(
    values: torch.Tensor,
    *,
    target_std: float | None,
    target_fro_norm: float | None,
) -> torch.Tensor:
    values = values.float() - values.float().mean()
    if target_std is not None and target_fro_norm is not None:
        implied = float(target_std) * math.sqrt(values.numel())
        if not math.isclose(implied, float(target_fro_norm), rel_tol=1e-4, abs_tol=1e-7):
            raise ValueError("target_std and target_fro_norm describe inconsistent zero-mean scales")
    if target_std is not None:
        values = _standardize(values) * float(target_std)
    elif target_fro_norm is not None:
        norm = torch.linalg.vector_norm(values)
        if norm <= 0:
            raise ValueError("Cannot Frobenius-rescale a zero tensor")
        values = values * (float(target_fro_norm) / norm)
    return values


def _reference_scale(shape: tuple[int, int], init_seed: int) -> float:
    reference = _global(shape, 0.6, np.random.default_rng(init_seed))
    reference = _rescale(reference, target_std=_xavier_std(shape), target_fro_norm=None)
    return float(reference.std(unbiased=False))


def _xavier_std(shape: tuple[int, int]) -> float:
    return math.sqrt(2.0 / float(shape[0] + shape[1]))


def _global(shape: tuple[int, int], alpha: float, rng: np.random.Generator) -> torch.Tensor:
    return torch.from_numpy(_fft_signal(shape[0] * shape[1], alpha, rng).reshape(shape))


def _rowwise(shape: tuple[int, int], alpha: float, rng: np.random.Generator) -> torch.Tensor:
    rows = [_fft_signal(shape[1], alpha, rng) for _ in range(shape[0])]
    return torch.from_numpy(np.stack(rows, axis=0))


def _colwise(shape: tuple[int, int], alpha: float, rng: np.random.Generator) -> torch.Tensor:
    columns = [_fft_signal(shape[0], alpha, rng) for _ in range(shape[1])]
    return torch.from_numpy(np.stack(columns, axis=1))


def initialize_A(
    shape: tuple[int, int],
    method: str,
    alpha: float | None = None,
    init_seed: int = 1107,
    target_std: float | None = None,
    target_fro_norm: float | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create a deterministic LoRA-A matrix; synthesis always happens in FP32.

    ``peft_default`` is intentionally rejected here. It must be applied by calling
    the installed PEFT layer's real reset method, not by copying a formula.
    """

    shape = _validate_shape(shape)
    require_registered(method)
    if method == "peft_default":
        raise ValueError("peft_default must use the installed PEFT reset path")

    rng = np.random.default_rng(int(init_seed))
    configured_alpha = METHOD_REGISTRY[method]["alpha"]
    effective_alpha = float(alpha if alpha is not None else (configured_alpha or 0.0))
    already_scaled = False

    if method == "iid_matched":
        values = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
    elif method == "fft_white_a0":
        values = _global(shape, 0.0, rng)
    elif method.startswith("powerlaw_global"):
        values = _global(shape, effective_alpha, rng)
    elif method == "powerlaw_shuffle_a06":
        values = _global(shape, 0.6, rng)
        if target_std is None and target_fro_norm is None:
            # Scale before permutation so this control contains the exact same
            # FP32 values as powerlaw_global_a06, only in a different order.
            values = _rescale(values, target_std=_xavier_std(shape), target_fro_norm=None)
            already_scaled = True
        permutation = torch.from_numpy(rng.permutation(values.numel()).astype(np.int64))
        values = values.flatten()[permutation].reshape(shape)
    elif method == "powerlaw_row_a06":
        values = _rowwise(shape, 0.6, rng)
    elif method == "powerlaw_col_a06":
        values = _colwise(shape, 0.6, rng)
    else:  # registry and implementation must remain synchronized
        raise AssertionError(f"No implementation for registered method {method}")

    if target_std is None and target_fro_norm is None and not already_scaled:
        if METHOD_REGISTRY[method]["scale_match"] == "exact_values":
            target_std = _xavier_std(shape)
        elif METHOD_REGISTRY[method]["scale_match"] == "powerlaw_global_a06":
            target_std = _reference_scale(shape, int(init_seed))
        else:
            target_std = _xavier_std(shape)

    if not already_scaled or target_std is not None or target_fro_norm is not None:
        values = _rescale(values, target_std=target_std, target_fro_norm=target_fro_norm)
    return values.to(device=device, dtype=dtype)


def apply_registered_initialization(model: torch.nn.Module, method: str, init_seed: int) -> dict:
    """Initialize PEFT LoRA modules while guaranteeing exact-zero B tensors."""

    require_registered(method)
    initialized: list[str] = []
    zeroed: list[str] = []
    modules = dict(model.named_modules())
    if method == "peft_default":
        reset_count = 0
        for name, module in modules.items():
            reset = getattr(module, "reset_lora_parameters", None)
            lora_a = getattr(module, "lora_A", None)
            if reset is None or not lora_a:
                continue
            adapter_names = list(lora_a.keys())
            for adapter_name in adapter_names:
                reset(adapter_name, init_lora_weights=True)
                reset_count += 1
                initialized.append(f"{name}.lora_A.{adapter_name}")
        if reset_count == 0:
            raise RuntimeError("No PEFT LoRA reset path was found")
    else:
        for name, parameter in model.named_parameters():
            lowered = name.lower()
            if "lora_a" in lowered and parameter.ndim == 2:
                parameter.data.copy_(
                    initialize_A(tuple(parameter.shape), method, init_seed=init_seed,
                                 dtype=parameter.dtype, device=parameter.device)
                )
                initialized.append(name)

    for name, parameter in model.named_parameters():
        if "lora_b" in name.lower():
            parameter.data.zero_()
            if torch.count_nonzero(parameter).item() != 0:
                raise AssertionError(f"Failed to zero {name}")
            zeroed.append(name)
    if not initialized or not zeroed:
        raise RuntimeError("Expected both LoRA A and LoRA B parameters")
    return {"method": method, "init_seed": int(init_seed), "initialized_A": initialized, "zeroed_B": zeroed}
