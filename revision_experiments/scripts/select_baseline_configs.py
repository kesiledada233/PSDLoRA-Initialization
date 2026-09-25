#!/usr/bin/env python3
"""Generate baseline-search outputs only after all screening runs validate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revision_experiments.scripts.baseline_selection import collect_screening_results, write_selection_outputs
from revision_experiments.scripts.matrix import load_matrix


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", type=Path,
        default=ROOT / "revision_experiments/config/baseline_search_matrix.yaml",
    )
    parser.add_argument(
        "--runs-root", type=Path,
        default=ROOT / "revision_experiments/results/runs",
    )
    parser.add_argument(
        "--all-trials-output", type=Path,
        default=ROOT / "revision_experiments/results/aggregate/baseline_search_all_trials.csv",
    )
    parser.add_argument(
        "--selected-output", type=Path,
        default=ROOT / "revision_experiments/results/aggregate/baseline_selected_configs.yaml",
    )
    args = parser.parse_args()
    rows, selected = collect_screening_results(load_matrix(args.matrix), args.runs_root)
    write_selection_outputs(rows, selected, args.all_trials_output, args.selected_output)
    print(json.dumps({
        "screening_trials": len(rows),
        "all_trials_output": str(args.all_trials_output),
        "selected_output": str(args.selected_output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
