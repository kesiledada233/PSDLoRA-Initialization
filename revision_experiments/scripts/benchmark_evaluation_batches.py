#!/usr/bin/env python3
"""Benchmark batched greedy generation against the frozen batch-size-1 path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from revision_experiments.scripts.evaluate_checkpoints import (
    build_evaluation_prompts,
    extract_choice,
    extract_code,
    extract_number,
    generate_all,
    load_adapter_model,
    resolve_evaluation_request,
)
from revision_experiments.scripts.training_support import load_task_records, load_tokenizer


ROOT = Path(__file__).resolve().parents[2]
MAX_NEW_TOKENS = {"gsm8k": 256, "cmmlu": 16, "mbpp": 1024, "sharegpt": 256}


def metric_extractor(task: str, records: list[dict] | None = None):
    """Task-specific metric evaluator for metric-level equivalence.

    GSM8K/CMMLU compare the exact formal extractors (the reported metric).
    MBPP compares the sandbox execution pass/fail vector — the honest metric
    is pass@1, so two differently-written correct solutions count as equal
    and a sample failing under both batch sizes is metric-equal. ShareGPT has
    no deterministic single-sample metric (NLL is continuous, judge scores
    free-form text); metric equivalence is reported as null.
    """
    if task == "gsm8k":
        return lambda outputs: [extract_number(text) for text in outputs]
    if task == "cmmlu":
        return lambda outputs: [extract_choice(text) for text in outputs]
    if task == "mbpp":
        from revision_experiments.scripts.safe_code_eval import evaluate_candidate

        def evaluate(outputs: list[str]) -> list[bool]:
            if records is None or len(records) < len(outputs):
                raise RuntimeError("MBPP metric equivalence requires the benchmarked records")
            return [
                bool(evaluate_candidate(
                    extract_code(text),
                    record.get("test_setup_code") or "",
                    record["test_list"],
                )["passed"])
                for text, record in zip(outputs, records)
            ]

        return evaluate
    return None


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("batch sizes must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must be positive")
    if 1 not in sizes:
        raise ValueError("batch size 1 is required as the equivalence reference")
    return tuple(sizes)


def compare_output_sets(reference: list[str], candidate: list[str]) -> list[int]:
    length = max(len(reference), len(candidate))
    return [
        index for index in range(length)
        if index >= len(reference) or index >= len(candidate) or reference[index] != candidate[index]
    ]


def build_benchmark_payload(
    *, run_id: str, checkpoint: int, task: str, sample_count: int, device: str,
    measurements: list[dict], metric_evaluator=None,
) -> dict:
    reference = measurements[0]["outputs"]
    reference_metrics = metric_evaluator(reference) if metric_evaluator else None
    public = []
    for measurement in measurements:
        mismatches = compare_output_sets(reference, measurement["outputs"])
        if metric_evaluator is None:
            metric_mismatches = None
        else:
            candidate_metrics = metric_evaluator(measurement["outputs"])
            metric_mismatches = [
                index for index in range(sample_count)
                if reference_metrics[index] != candidate_metrics[index]
            ]
        seconds = float(measurement["seconds"])
        public.append({
            "batch_size": int(measurement["batch_size"]),
            "seconds": seconds,
            "samples_per_second": sample_count / seconds,
            "peak_memory_bytes": int(measurement["peak_memory_bytes"]),
            "exact_match_to_batch_1": not mismatches,
            "mismatch_indices": mismatches,
            "metric_match_to_batch_1": None if metric_mismatches is None else not metric_mismatches,
            "metric_mismatch_indices": metric_mismatches,
        })
    metric_rows = [row for row in public if row["metric_match_to_batch_1"] is not None]
    return {
        "schema_version": 1,
        "formal_result": False,
        "purpose": "generation_batch_equivalence_and_throughput",
        "run_id": run_id,
        "checkpoint": int(checkpoint),
        "task": task,
        "sample_selection": "first_n_of_frozen_formal_order",
        "sample_count": int(sample_count),
        "device": device,
        "comparison": "exact_decoded_text_against_batch_size_1",
        "metric_comparison": "extracted_answer_against_batch_size_1",
        "all_equivalent": all(row["exact_match_to_batch_1"] for row in public),
        "all_metric_equivalent": (
            None if not metric_rows else all(row["metric_match_to_batch_1"] for row in metric_rows)
        ),
        "measurements": public,
    }


def validate_output_path(output: str | Path, project_root: str | Path = ROOT) -> Path:
    output = Path(output).resolve()
    allowed = Path(project_root).resolve() / "revision_experiments/results/smoke/evaluation_benchmarks"
    if not output.is_relative_to(allowed):
        raise RuntimeError(f"output must stay inside the isolated benchmark directory: {allowed}")
    if output.suffix != ".json":
        raise RuntimeError("benchmark output must be a .json file")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--task", choices=sorted(MAX_NEW_TOKENS), required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Required because this command loads a GPU model")
    args = parser.parse_args()
    try:
        sizes = parse_batch_sizes(args.batch_sizes)
        output = validate_output_path(args.output)
        if not 1 <= args.sample_count <= 32:
            raise RuntimeError("sample-count must be between 1 and 32")
        if output.exists():
            raise RuntimeError(f"refusing to overwrite benchmark artifact: {output}")
        _, _, _, formal_count = resolve_evaluation_request(
            args.run_dir, args.checkpoint, args.task,
        )
        if args.sample_count >= formal_count:
            raise RuntimeError("benchmark must use a strict prefix of the frozen formal sample set")
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    plan = {
        "formal_result": False, "run_id": args.run_dir.name, "checkpoint": args.checkpoint,
        "task": args.task, "sample_count": args.sample_count, "batch_sizes": sizes,
        "device": args.device, "output": str(output),
    }
    if not args.execute:
        print(json.dumps({"mode": "plan", **plan}, indent=2))
        return 0
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("batch benchmark requires an available CUDA device")

    device = torch.device(args.device)
    model, model_key = load_adapter_model(args.run_dir, args.checkpoint, device)
    tokenizer = load_tokenizer(model_key)
    benchmark_records = None
    if args.task == "sharegpt":
        from revision_experiments.scripts.generate_sharegpt_candidates import load_frozen_sharegpt_prompts
        prompts = [row["prompt"] for row in load_frozen_sharegpt_prompts()[:args.sample_count]]
    else:
        train_records, test_records = load_task_records(args.task)
        prompts = build_evaluation_prompts(args.task, train_records, test_records, args.sample_count)
        if args.task == "mbpp":
            benchmark_records = test_records[:args.sample_count]
    measurements = []
    for batch_size in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        outputs = generate_all(
            model, tokenizer, prompts, device, MAX_NEW_TOKENS[args.task], batch_size,
        )
        torch.cuda.synchronize(device)
        measurements.append({
            "batch_size": batch_size, "seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device), "outputs": outputs,
        })
    payload = build_benchmark_payload(
        run_id=args.run_dir.name, checkpoint=args.checkpoint, task=args.task,
        sample_count=args.sample_count, device=args.device, measurements=measurements,
        metric_evaluator=metric_extractor(args.task, benchmark_records),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["all_equivalent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
