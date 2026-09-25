"""Bounded, crash-tolerant gradient coordinate logging."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


class GradientLogger:
    def __init__(self, model, output_dir: Path, max_steps: int, coordinate_count: int, coordinate_seed: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        candidates = [(name, parameter) for name, parameter in model.named_parameters()
                      if ("lora_A" in name or "lora_B" in name) and parameter.requires_grad]
        layer_ids = sorted({int(match.group(1)) for name, _ in candidates
                            if (match := re.search(r"layers\.(\d+)\.", name))})
        selected = {layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]} if layer_ids else set()
        self.entries = []
        rng = np.random.default_rng(int(coordinate_seed))
        for name, parameter in candidates:
            match = re.search(r"layers\.(\d+)\.", name)
            if match and int(match.group(1)) not in selected:
                continue
            count = min(int(coordinate_count), parameter.numel())
            indices = np.sort(rng.choice(parameter.numel(), size=count, replace=False)).astype(np.int64)
            path = self.output_dir / f"{_slug(name)}.npy"
            values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(max_steps + 1, count))
            values[:] = np.nan
            self.entries.append({"name": name, "parameter": parameter, "indices": indices, "values": values, "path": path.name})
        manifest = {
            "schema_version": 1, "max_steps": max_steps, "coordinate_seed": int(coordinate_seed),
            "selected_layers": sorted(selected),
            "parameters": [{"name": item["name"], "indices": item["indices"].tolist(), "file": item["path"]}
                           for item in self.entries],
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def record(self, step: int) -> None:
        for item in self.entries:
            gradient = item["parameter"].grad
            if gradient is None:
                item["values"][step, :] = 0.0
            else:
                flat = gradient.detach().float().cpu().reshape(-1).numpy()
                item["values"][step, :] = flat[item["indices"]]
            item["values"].flush()

    def full_diagnostics(self, step: int) -> list[dict]:
        rows = []
        for item in self.entries:
            gradient = item["parameter"].grad
            rows.append({
                "kind": "lora_parameter_gradient", "step": int(step), "parameter": item["name"],
                "gradient_frobenius_norm": float(torch.linalg.vector_norm(gradient.detach().float())) if gradient is not None else 0.0,
                "parameter_frobenius_norm": float(torch.linalg.vector_norm(item["parameter"].detach().float())),
            })
        return rows


def coordinate_invariant_diagnostics(
    gradient: torch.Tensor,
    lora_a: torch.Tensor,
    randomized_seed: int = 20260903,
) -> dict:
    """Compute Task-5 subspace diagnostics without materializing P_A."""
    gradient = gradient.detach().float()
    lora_a = lora_a.detach().float()
    if gradient.ndim != 2 or lora_a.ndim != 2 or gradient.shape[1] != lora_a.shape[1]:
        raise ValueError("Expected G[out,in] and A[rank,in] with the same input dimension")
    gram = lora_a @ lora_a.T
    g_at = gradient @ lora_a.T
    projected = g_at @ torch.linalg.pinv(gram) @ lora_a
    denominator = torch.sum(gradient.square())
    capture = torch.sum(projected.square()) / denominator if denominator > 0 else torch.tensor(float("nan"), device=gradient.device)
    a_singular_values = torch.linalg.svdvals(lora_a)
    tolerance = torch.finfo(a_singular_values.dtype).eps * max(lora_a.shape) * a_singular_values.max()
    effective_rank = int(torch.sum(a_singular_values > tolerance))
    q_a, _ = torch.linalg.qr(lora_a.T, mode="reduced")
    top_rank = min(lora_a.shape[0], min(gradient.shape))
    if max(gradient.shape) <= 256:
        top_right = torch.linalg.svd(gradient, full_matrices=False).Vh[:top_rank].T
    else:
        devices = [gradient.device.index] if gradient.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(randomized_seed))
            if gradient.is_cuda:
                torch.cuda.manual_seed_all(int(randomized_seed))
            _, _, top_right = torch.svd_lowrank(gradient, q=top_rank, niter=4)
    cosines = torch.linalg.svdvals(q_a.T @ top_right).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(cosines))
    return {
        "gradient_frobenius_norm": float(torch.linalg.vector_norm(gradient)),
        "gradient_capture_ratio": float(capture),
        "gradient_times_a_transpose_norm": float(torch.linalg.vector_norm(g_at)),
        "a_singular_values": [float(value) for value in a_singular_values],
        "a_effective_rank": effective_rank,
        "principal_angles_degrees": [float(value) for value in angles],
    }


class FullMatrixDiagnosticLogger:
    """Capture effective base-weight gradients only at disclosed snapshot steps."""

    def __init__(self, model, snapshot_steps: list[int], randomized_seed: int = 20260903):
        self.snapshot_steps = {int(step) for step in snapshot_steps}
        self.randomized_seed = int(randomized_seed)
        candidates = []
        for name, module in model.named_modules():
            if not re.search(r"layers\.(\d+)\.", name):
                continue
            if getattr(module, "lora_A", None) is None or "default" not in module.lora_A:
                continue
            candidates.append((name, module))
        layer_ids = sorted({int(re.search(r"layers\.(\d+)\.", name).group(1)) for name, _ in candidates})
        selected = {layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]} if layer_ids else set()
        self.modules = {
            name: module for name, module in candidates
            if int(re.search(r"layers\.(\d+)\.", name).group(1)) in selected
        }
        self.active_step = None
        self.gradients: dict[str, torch.Tensor] = {}
        self.calls: dict[str, int] = {}
        self.hooks = [module.register_forward_hook(self._forward_hook(name)) for name, module in self.modules.items()]

    def _forward_hook(self, name: str):
        def hook(_module, inputs, output):
            if self.active_step is None:
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise RuntimeError(f"Unsupported linear hook signature for {name}")
            input_tensor = inputs[0].detach()

            def capture(output_gradient):
                input_flat = input_tensor.reshape(-1, input_tensor.shape[-1]).float()
                output_flat = output_gradient.detach().reshape(-1, output_gradient.shape[-1]).float()
                matrix_gradient = output_flat.T @ input_flat
                self.gradients[name].add_(matrix_gradient)
                self.calls[name] += 1
                return output_gradient

            output.register_hook(capture)
        return hook

    def start(self, step: int) -> None:
        if int(step) not in self.snapshot_steps:
            self.active_step = None
            return
        if self.active_step is not None:
            raise RuntimeError("A full-matrix diagnostic snapshot is already active")
        self.active_step = int(step)
        self.gradients = {}
        self.calls = {}
        for name, module in self.modules.items():
            shape = module.base_layer.weight.shape
            self.gradients[name] = torch.zeros(shape, device=module.base_layer.weight.device, dtype=torch.float32)
            self.calls[name] = 0

    def finish(self, step: int) -> list[dict]:
        if self.active_step is None:
            return []
        if int(step) != self.active_step:
            raise RuntimeError(f"Finishing step {step} while snapshot {self.active_step} is active")
        rows = []
        for name, module in self.modules.items():
            if self.calls[name] == 0:
                raise RuntimeError(f"No effective base gradient captured for {name} at step {step}")
            diagnostics = coordinate_invariant_diagnostics(
                self.gradients[name], module.lora_A["default"].weight, self.randomized_seed,
            )
            rows.append({
                "kind": "effective_base_weight_subspace",
                "step": int(step),
                "parameter": f"{name}.base_layer.weight",
                "microbatch_backward_calls": self.calls[name],
                **diagnostics,
            })
        self.active_step = None
        self.gradients = {}
        self.calls = {}
        return rows

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
