#!/usr/bin/env python3
"""Plan and execute the amended formal evaluation queue without overwrites."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from revision_experiments.scripts.aggregate_results import (
    discover_expected_runs,
    validate_evaluation_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "revision_experiments/config"
RUNS_ROOT = ROOT / "revision_experiments/results/runs"
EVALUATIONS_ROOT = ROOT / "revision_experiments/results/evaluations"
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}


@dataclass(frozen=True)
class EvaluationJob:
    priority: str
    status: str
    stage: str
    run_id: str
    checkpoint: int
    task: str
    run_dir: Path
    result_path: Path
    reason: str


def evaluation_paths(
    evaluations_root: str | Path, run_id: str, checkpoint: int, task: str,
) -> dict[str, Path]:
    root = Path(evaluations_root)
    stem = f"{run_id}__step_{int(checkpoint):06d}__{task}"
    paths = {"result": root / f"{stem}.json"}
    if task == "sharegpt":
        paths.update({
            "candidate": root / f"{stem}_candidates.jsonl",
            "candidate_manifest": root / f"{stem}_candidates.manifest.json",
            "judge": root / f"{stem}_judged.jsonl",
        })
    else:
        paths["prediction"] = root / f"{stem}.jsonl"
    return paths


def _priority(expected: dict, checkpoint: int) -> str:
    contexts = set(expected.get("analysis_contexts", ()))
    if "downstream_2500step" in contexts:
        return "P0" if int(checkpoint) == 2500 else "P2"
    return "P1"


def _checkpoint_ready(run_dir: Path, checkpoint: int) -> bool:
    adapter = run_dir / "checkpoints" / f"step_{int(checkpoint):06d}"
    return (
        (run_dir / "COMPLETED").is_file()
        and not (run_dir / "FAILED.json").exists()
        and (adapter / "adapter_config.json").is_file()
        and any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    )


def build_evaluation_jobs(
    expected_runs: list[dict],
    runs_root: str | Path,
    evaluations_root: str | Path,
    *,
    project_root: str | Path = ROOT,
) -> list[EvaluationJob]:
    """Return one deduplicated, fail-closed job for every declared evaluation."""
    runs_root = Path(runs_root)
    evaluations_root = Path(evaluations_root)
    project_root = Path(project_root)
    jobs: dict[tuple[str, int, str], EvaluationJob] = {}
    for expected in expected_runs:
        run_id = expected["run_id"]
        run_dir = runs_root / run_id
        for requirement in expected.get("expected_evaluations", ()):
            checkpoint = int(requirement["checkpoint"])
            task = str(requirement["task"])
            key = (run_id, checkpoint, task)
            if key in jobs:
                continue
            paths = evaluation_paths(evaluations_root, run_id, checkpoint, task)
            common = dict(
                priority=_priority(expected, checkpoint), run_id=run_id,
                checkpoint=checkpoint, task=task, run_dir=run_dir,
                result_path=paths["result"],
            )
            if paths["result"].exists():
                try:
                    validate_evaluation_artifact(paths["result"], requirement, run_id, project_root)
                except (OSError, RuntimeError, ValueError) as exc:
                    job = EvaluationJob(**common, status="blocked", stage="none", reason=f"invalid result: {exc}")
                else:
                    job = EvaluationJob(**common, status="complete", stage="none", reason="valid result exists")
            elif not _checkpoint_ready(run_dir, checkpoint):
                job = EvaluationJob(
                    **common, status="waiting", stage="none",
                    reason="completed run or adapter checkpoint is not available",
                )
            elif task != "sharegpt":
                prediction_exists = paths["prediction"].exists()
                job = EvaluationJob(
                    **common,
                    status="blocked" if prediction_exists else "ready",
                    stage="none" if prediction_exists else "evaluate",
                    reason="orphan prediction artifact" if prediction_exists else "",
                )
            else:
                candidate = paths["candidate"].exists()
                manifest = paths["candidate_manifest"].exists()
                judged = paths["judge"].exists()
                if candidate != manifest:
                    job = EvaluationJob(
                        **common, status="blocked", stage="none",
                        reason="partial ShareGPT candidate artifacts",
                    )
                elif judged:
                    job = EvaluationJob(
                        **common, status="blocked", stage="none",
                        reason="orphan ShareGPT judge artifact without final result",
                    )
                elif candidate:
                    job = EvaluationJob(**common, status="ready", stage="judge", reason="")
                else:
                    job = EvaluationJob(**common, status="ready", stage="candidates", reason="")
            jobs[key] = job
    return sorted(
        jobs.values(),
        key=lambda job: (PRIORITY_ORDER[job.priority], job.run_id, job.checkpoint, job.task),
    )


def build_command(
    job: EvaluationJob,
    *,
    project_root: str | Path = ROOT,
    device: str,
    generation_batch_size: int,
    allow_code_execution: bool,
) -> list[str]:
    if job.status != "ready":
        raise RuntimeError(f"job is not executable: {job.status}")
    if generation_batch_size <= 0:
        raise RuntimeError("generation batch size must be positive")
    root = Path(project_root)
    if job.stage in {"evaluate", "candidates"}:
        if job.task == "mbpp" and not allow_code_execution:
            raise RuntimeError("MBPP execution requires --allow-code-execution")
        command = [
            sys.executable, str(root / "revision_experiments/scripts/evaluate_checkpoints.py"),
            "--run-dir", str(job.run_dir), "--checkpoint", str(job.checkpoint),
            "--task", job.task, "--device", device,
            "--generation-batch-size", str(generation_batch_size),
        ]
        if job.task == "mbpp":
            command.append("--allow-code-execution")
        return command
    if job.stage == "judge" and job.task == "sharegpt":
        paths = evaluation_paths(job.result_path.parent, job.run_id, job.checkpoint, job.task)
        return [
            sys.executable, str(root / "revision_experiments/scripts/sharegpt_judge.py"),
            "--input", str(paths["candidate"]), "--output", str(paths["judge"]),
            "--result", str(paths["result"]), "--run-dir", str(job.run_dir),
            "--checkpoint", str(job.checkpoint), "--device", device,
        ]
    raise RuntimeError(f"unsupported evaluation stage: {job.stage}")


def validate_batch_equivalence_reports(
    jobs: list[EvaluationJob], max_batch_size: int, report_paths: list[str | Path],
) -> dict[tuple[str, str], dict]:
    """Parse isolated benchmark reports into per model/task validated sizes.

    ``--generation-batch-size`` is a CAP: a combo whose report validated exact
    equivalence at some size <= the cap runs at the largest validated size;
    a combo without a covering report (or without any validated size) safely
    falls back to batch size 1. Long-generation tasks routinely fail exact-text
    equivalence at every size, and that outcome is a downgrade, not an error.
    """
    if max_batch_size <= 1:
        return {}
    coverage: dict[tuple[str, str], dict] = {}
    for path in report_paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid batch-equivalence report {path}: {exc}") from exc
        if payload.get("formal_result") is not False:
            raise RuntimeError(f"batch-equivalence report did not pass: {path}")
        measured = {
            int(row["batch_size"]) for row in payload.get("measurements", ())
            if isinstance(row, dict)
        }
        if 1 not in measured:
            raise RuntimeError(f"batch-equivalence report lacks the batch-1 baseline: {path}")
        validated = {
            int(row["batch_size"]) for row in payload.get("measurements", ())
            if isinstance(row, dict) and row.get("exact_match_to_batch_1") is True
        }
        key = (str(payload.get("run_id", "")).split("__", 1)[0], payload.get("task"))
        coverage[key] = {
            "report": str(Path(path).resolve()),
            "validated_sizes": sorted(validated),
            "metric_validated_sizes": sorted({
                int(row["batch_size"]) for row in payload.get("measurements", ())
                if isinstance(row, dict) and row.get("metric_match_to_batch_1") is True
            }),
        }
    return coverage


def effective_batch_size(
    job: EvaluationJob, coverage: dict[tuple[str, str], dict], max_batch_size: int,
    *, standard: str = "exact",
) -> int:
    """Largest validated batch size for the job's model/task, else 1.

    ``standard="exact"`` requires decoded-text equality against batch 1.
    ``standard="metric"`` accepts extracted-answer equality (greedy decoding
    has no canonical floating-point output; the reported metric is the
    invariant). Reports without metric fields fall back to exact validation.
    """
    if max_batch_size <= 1 or job.stage not in {"evaluate", "candidates"}:
        return 1
    entry = coverage.get((job.run_id.split("__", 1)[0], job.task))
    if entry is None:
        return 1
    field = "metric_validated_sizes" if standard == "metric" else "validated_sizes"
    validated = [size for size in entry.get(field, ()) if 1 < size <= max_batch_size]
    return max(validated) if validated else 1


def _json_job(job: EvaluationJob) -> dict:
    payload = asdict(job)
    payload["run_dir"] = str(job.run_dir)
    payload["result_path"] = str(job.result_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute ready jobs; default is a read-only plan")
    parser.add_argument("--priority", choices=["all", "P0", "P1", "P2"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument(
        "--batch-equivalence-report", type=Path, action="append", default=[],
        help="Isolated benchmark report; per model/task combos without a covering "
             "report safely fall back to batch size 1",
    )
    parser.add_argument(
        "--equivalence-standard", choices=["exact", "metric"], default="exact",
        help="exact: decoded text must match batch 1. metric: extracted answers must "
             "match (author-approved relaxation; reports without metric fields fall "
             "back to exact validation)",
    )
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.generation_batch_size <= 0:
        parser.error("--generation-batch-size must be positive")

    expected = discover_expected_runs(CONFIG_DIR)
    jobs = build_evaluation_jobs(expected, RUNS_ROOT, EVALUATIONS_ROOT)
    if args.priority != "all":
        jobs = [job for job in jobs if job.priority == args.priority]
    selected = [job for job in jobs if job.status == "ready"]
    if args.limit is not None:
        selected = selected[:args.limit]
    print(json.dumps({
        "mode": "execute" if args.execute else "plan",
        "job_count": len(jobs), "ready_count": len(selected),
        "jobs": [_json_job(job) for job in jobs],
    }, indent=2, ensure_ascii=False))
    if not args.execute:
        return 0

    try:
        coverage = validate_batch_equivalence_reports(
            selected, args.generation_batch_size, args.batch_equivalence_report,
        )
        commands = []
        batch_plan: dict[str, int] = {}
        for job in selected:
            batch = effective_batch_size(
                job, coverage, args.generation_batch_size, standard=args.equivalence_standard,
            )
            batch_plan[f"{job.run_id.split('__', 1)[0]}/{job.task}" + (
                "" if job.stage != "judge" else " (judge)"
            )] = batch
            commands.append(
                build_command(
                    job, device=args.device, generation_batch_size=batch,
                    allow_code_execution=args.allow_code_execution,
                )
            )
        if batch_plan:
            print("BATCH_PLAN", json.dumps(batch_plan, sort_keys=True), flush=True)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    for job, command in zip(selected, commands):
        print("RUN", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            return completed.returncode
        if job.task == "sharegpt" and job.stage == "candidates":
            judge_job = replace(job, stage="judge")
            judge_command = build_command(
                judge_job, device=args.device,
                generation_batch_size=effective_batch_size(
                    judge_job, coverage, args.generation_batch_size
                ),
                allow_code_execution=args.allow_code_execution,
            )
            print("RUN", " ".join(judge_command), flush=True)
            completed = subprocess.run(judge_command, cwd=ROOT, check=False)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
