#!/usr/bin/env python3
"""Combine immutable per-model Gate 2 cases into the canonical audit artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from revision_experiments.scripts.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CASES = ("openpangu__gsm8k", "qwen__cmmlu")
REQUIRED_METHODS = ("base", "peft_default", "powerlaw_global_a06")


def collect_cases(cases_dir: Path, required_cases=REQUIRED_CASES) -> tuple[list[dict], pd.DataFrame]:
    payloads = []
    frames = []
    for case_id in required_cases:
        json_path = cases_dir / f"{case_id}.json"
        csv_path = cases_dir / f"{case_id}.csv"
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            # Keep numeric fields as their original decimal strings.  Reading
            # loss as float64 and writing it again can round the final digit,
            # making the canonical CSV differ from its immutable case CSVs.
            frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
            raise RuntimeError(f"missing or invalid step-zero case {case_id}: {exc}") from exc
        if payload.get("schema_version") != 4 or payload.get("case_id") != case_id:
            raise RuntimeError(f"step-zero case identity mismatch: {case_id}")
        if payload.get("batch_losses_file") != csv_path.name or payload.get("batch_losses_sha256") != file_sha256(csv_path):
            raise RuntimeError(f"step-zero case CSV binding mismatch: {case_id}")
        expected_columns = {"model", "task", "seed", "batch", "method", "loss"}
        if set(frame.columns) != expected_columns or len(frame) != 300:
            raise RuntimeError(f"step-zero case has invalid batch rows: {case_id}")
        if set(frame["model"]) != {payload["model"]} or set(frame["task"]) != {payload["task"]}:
            raise RuntimeError(f"step-zero case CSV identity mismatch: {case_id}")
        for method in REQUIRED_METHODS:
            rows = frame[frame["method"] == method]
            if (
                len(rows) != 100 or set(pd.to_numeric(rows["batch"], errors="coerce")) != set(range(100))
                or not pd.to_numeric(rows["loss"], errors="coerce").map(math.isfinite).all()
            ):
                raise RuntimeError(f"step-zero case method rows are incomplete/non-finite: {case_id}/{method}")
        tolerance = payload.get("tolerance")
        equivalence = payload.get("equivalence")
        if (
            not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance)) or tolerance <= 0
            or not isinstance(equivalence, dict)
            or set(equivalence) != set(REQUIRED_METHODS[1:])
        ):
            raise RuntimeError(f"step-zero case equivalence schema is invalid: {case_id}")
        for method, values in equivalence.items():
            numeric = [
                values.get("max_abs_logit_difference"), values.get("mean_abs_logit_difference"),
                values.get("max_abs_loss_difference"), values.get("mean_abs_loss_difference"),
            ] if isinstance(values, dict) else []
            if (
                not isinstance(values, dict) or values.get("b_nonzero") != 0
                or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)) and abs(float(value)) <= float(tolerance)
                    for value in numeric
                )
            ):
                raise RuntimeError(f"step-zero case equivalence failed: {case_id}/{method}")
        diagnostics = payload.get("batch_target_diagnostics")
        if (
            not isinstance(diagnostics, list) or len(diagnostics) != 100
            or [row.get("batch") for row in diagnostics if isinstance(row, dict)] != list(range(100))
            or any(
                not isinstance(row, dict) or row.get("valid_target_tokens", 0) <= 0
                or row.get("padding_target_tokens") != 0 for row in diagnostics
            )
            or payload.get("equivalence_passed") is not True
            or payload.get("loss_anomaly") is not False
        ):
            raise RuntimeError(f"step-zero case gate claims/target diagnostics are invalid: {case_id}")
        payloads.append(payload)
        frames.append(frame)
    return payloads, pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases-dir", type=Path,
        default=ROOT / "revision_experiments/results/audits/step_zero_cases",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=ROOT / "revision_experiments/results/audits/step_zero_audit.json",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=ROOT / "revision_experiments/results/audits/step_zero_batch_losses.csv",
    )
    args = parser.parse_args()
    if args.output_json.exists() or args.output_csv.exists():
        raise SystemExit("Refusing to overwrite canonical step-zero evidence")
    cases, losses = collect_cases(args.cases_dir)
    gate_passed = all(
        case.get("equivalence_passed") is True and case.get("loss_anomaly") is False
        for case in cases
    )
    payload = {
        "schema_version": 4,
        "required_cases": list(REQUIRED_CASES),
        "case_count": len(cases),
        "gate_passed": gate_passed,
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    losses.to_csv(args.output_csv, index=False)
    payload["combined_batch_losses_sha256"] = file_sha256(args.output_csv)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
