#!/usr/bin/env python3
"""Analyze early temporal PSD from GradientLogger memmaps or legacy JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_experiments.scripts.metrics import temporal_psd_slope


def load_jsonl_series(path: Path) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    seen_steps = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if step in seen_steps:
            raise ValueError(f"Duplicate gradient step {step} in {path}")
        seen_steps.add(step)
        for key, value in row["series"].items():
            series.setdefault(key, []).append(float(value))
    if seen_steps and seen_steps != set(range(min(seen_steps), max(seen_steps) + 1)):
        raise ValueError(f"Missing gradient steps in {path}")
    return series


def load_gradient_directory(path: Path, through_step: int) -> dict[str, list[float]]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing GradientLogger manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if through_step < 1 or through_step > int(manifest["max_steps"]):
        raise ValueError(
            f"through_step {through_step} is outside recorded range 1..{manifest['max_steps']} for {path}"
        )
    series = {}
    for parameter in manifest["parameters"]:
        values = np.load(path / parameter["file"], mmap_mode="r")
        expected_rows = int(manifest["max_steps"]) + 1
        if values.ndim != 2 or values.shape[0] != expected_rows:
            raise ValueError(f"Unexpected gradient array shape {values.shape} in {parameter['file']}")
        selected = np.asarray(values[: through_step + 1], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError(f"Missing/non-finite gradient coordinates through step {through_step}: {parameter['file']}")
        for column, coordinate in enumerate(parameter["indices"]):
            key = f"{parameter['name']}[{coordinate}]"
            series[key] = selected[:, column].tolist()
    return series


def load_series(path: Path, through_step: int) -> dict[str, list[float]]:
    return load_gradient_directory(path, through_step) if path.is_dir() else load_jsonl_series(path)


def analysis_windows(through_step: int) -> list[tuple[int, int]]:
    windows = [(0, through_step)]
    for start, end in ((0, 100), (100, 250), (250, 500)):
        if end <= through_step and (start, end) not in windows:
            windows.append((start, end))
    return windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--through-step", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("revision_experiments/results/aggregate/early_psd_per_seed.parquet"))
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        for name, values in load_series(path, args.through_step).items():
            for window_start, window_end in analysis_windows(args.through_step):
                window_values = values[window_start: window_end + 1]
                for band in ((0.01, 0.08), (0.005, 0.08), (0.01, 0.10)):
                    result = temporal_psd_slope(window_values, band)
                    rows.append({
                        "source": str(path), "through_step": args.through_step, "series": name,
                        "window_start": window_start, "window_end": window_end, **result,
                    })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
