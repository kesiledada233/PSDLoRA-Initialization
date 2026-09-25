"""Numerical metrics shared by tests, aggregation, and plotting."""

from __future__ import annotations

import numpy as np
from scipy import signal


def raw_trapezoid_auc(losses, start: int = 0, end: int = 500) -> float:
    values = np.asarray(losses, dtype=np.float64)
    if start < 0 or end <= start:
        raise ValueError("Expected 0 <= start < end")
    if values.size < end:
        raise ValueError(f"Need at least {end} loss values, got {values.size}")
    window = values[start:end]
    if not np.all(np.isfinite(window)):
        raise ValueError("AUC input contains non-finite losses")
    return float(np.trapezoid(window, dx=1.0))


def trapezoid_auc_at_steps(steps, values, start: int = 0, end: int = 500) -> float:
    x = np.asarray(steps, dtype=np.int64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("steps and values must be equal-length one-dimensional arrays")
    if x.size < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("steps must be strictly increasing with at least two entries")
    if x[0] != start or x[-1] != end:
        raise ValueError(f"Expected validation endpoints {start} and {end}, got {x[0]} and {x[-1]}")
    if not np.all(np.isfinite(y)):
        raise ValueError("AUC input contains non-finite values")
    return float(np.trapezoid(y, x=x))


def temporal_psd_slope(
    values,
    frequency_band: tuple[float, float] = (0.01, 0.08),
    *,
    sampling_frequency: float = 1.0,
) -> dict[str, float | int | list[float]]:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 64:
        raise ValueError("PSD estimation requires a one-dimensional series with >=64 samples")
    if not np.all(np.isfinite(series)):
        raise ValueError("PSD input contains non-finite values")
    detrended = signal.detrend(series - series.mean(), type="linear")
    nperseg = min(256, series.size)
    frequencies, density = signal.welch(
        detrended,
        fs=sampling_frequency,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
    )
    low, high = map(float, frequency_band)
    mask = (frequencies >= low) & (frequencies <= high) & (density > 0)
    if mask.sum() < 3:
        raise ValueError(f"Frequency band {frequency_band} contains too few PSD bins")
    x = np.log10(frequencies[mask])
    y = np.log10(density[mask])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.square(y - predicted).sum())
    total = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return {
        "slope": float(slope),
        "alpha": float(-slope),
        "r_squared": float(r_squared),
        "frequency_band": [low, high],
        "num_frequency_bins": int(mask.sum()),
    }
