#!/usr/bin/env python3
"""Generate CPU initialization statistics for all standalone registry methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from revision_experiments.initializers import METHOD_REGISTRY, initialize_A
from revision_experiments.scripts.metrics import temporal_psd_slope


def matrix_stats(method: str, shape: tuple[int, int], seed: int) -> dict:
    matrix = initialize_A(shape, method, init_seed=seed).float().cpu()
    flat = matrix.numpy().reshape(-1)
    singular_values = torch.linalg.svdvals(matrix)
    try:
        psd = temporal_psd_slope(flat, (0.01, 0.08))
    except ValueError:
        psd = {"slope": np.nan, "alpha": np.nan, "r_squared": np.nan}
    return {
        "method": method, "seed": seed, "rows": shape[0], "columns": shape[1],
        "mean": float(flat.mean()), "std": float(flat.std()), "variance": float(flat.var()),
        "frobenius_norm": float(torch.linalg.vector_norm(matrix)),
        "spectral_norm": float(singular_values.max()), "max_abs": float(np.abs(flat).max()),
        "skewness": float(stats.skew(flat)), "kurtosis": float(stats.kurtosis(flat)),
        "psd_slope": psd["slope"], "psd_alpha": psd["alpha"], "psd_r_squared": psd["r_squared"],
    }


def write_diagnostic_figure(shape: tuple[int, int], seed: int, output: Path) -> list[Path]:
    """Render the Task-3 representative spatial PSD/autocorrelation diagnostic."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [name for name in METHOD_REGISTRY if name != "peft_default"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in methods:
        values = initialize_A(shape, method, init_seed=seed).float().cpu().numpy().reshape(-1)
        centered = values.astype(np.float64) - float(values.mean())
        frequencies = np.fft.rfftfreq(centered.size)[1:]
        density = np.square(np.abs(np.fft.rfft(centered)))[1:] / centered.size
        axes[0].loglog(frequencies, density, linewidth=1.0, alpha=0.85, label=method)
        fft_size = 1 << (2 * centered.size - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=fft_size)
        autocovariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:129]
        autocorrelation = autocovariance / autocovariance[0]
        axes[1].plot(np.arange(1, len(autocorrelation)), autocorrelation[1:], linewidth=1.0, label=method)
    axes[0].set(title="Representative LoRA-A spatial periodogram", xlabel="Flattened spatial frequency", ylabel="Power")
    axes[1].set(title="Representative LoRA-A autocorrelation", xlabel="Flattened lag", ylabel="Autocorrelation")
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    figure.suptitle(f"Initialization structure diagnostic (shape={shape}, seed={seed})")
    destinations = [output.with_name(output.stem + "_diagnostic." + extension) for extension in ("svg", "png")]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=220 if destination.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return destinations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs=2, type=int, default=[16, 4096])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1107, 123, 42])
    parser.add_argument("--output", type=Path, default=Path("revision_experiments/results/audits/initialization_statistics.parquet"))
    args = parser.parse_args()
    methods = [name for name in METHOD_REGISTRY if name != "peft_default"]
    rows = [matrix_stats(method, tuple(args.shape), seed) for method in methods for seed in args.seeds]
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    figures = write_diagnostic_figure(tuple(args.shape), args.seeds[0], args.output)
    print(frame.to_string(index=False))
    print("diagnostic figures:", ", ".join(str(path) for path in figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
