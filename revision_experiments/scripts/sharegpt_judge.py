#!/usr/bin/env python3
"""Official-format Prometheus 2 absolute grading for frozen ShareGPT candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

from revision_experiments.scripts.audit_artifacts import prometheus_evidence_ready
from revision_experiments.scripts.evaluate_checkpoints import (
    canonical_evaluation_paths, require_new_artifacts, resolve_evaluation_request,
)
from revision_experiments.scripts.generate_sharegpt_candidates import load_frozen_sharegpt_prompts
from revision_experiments.scripts.schema import (
    canonical_hash, file_sha256, formal_evaluation_config, formal_sample_entries,
    formal_sample_manifest, formal_sharegpt_examples,
)


ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective feedback based on "
    "specific criteria, ensuring each assessment reflects the absolute standards set for performance."
)
RUBRIC = """Criteria: correctness, relevance, clarity, helpfulness, and safety.
Score 1: The response is incorrect, irrelevant, unsafe, or unusable.
Score 2: The response has major errors or omissions and is only marginally useful.
Score 3: The response is broadly useful but has noticeable errors, omissions, or clarity problems.
Score 4: The response is correct and helpful with only minor limitations.
Score 5: The response is fully correct, relevant, clear, helpful, and safe."""
PROMPT_TEMPLATE_ID = "prometheus2_official_absolute_mistral_v1"

# Prometheus 2 rubric feedback on real adapter candidates routinely exceeds the
# original 256-token budget, truncating the final "[RESULT] N" marker and making
# the whole 200-prompt judgment run unparseable. Some candidates also trigger
# long meta-commentary before the score. 4096 covers observed worst cases while
# keeping generation deterministic (greedy); typical feedback still ends early
# via EOS, so runtime cost is small.
JUDGE_MAX_NEW_TOKENS = 4096
FORMAL_PROTOCOL = {
    "metrics": ["heldout_nll", "prometheus_absolute_score"],
    "judge": "prometheus-7b-v2.0",
    "judge_revision": "66ffb1fc20beebfb60a3964a957d9011723116c5",
    "comparison_reference": "frozen_first_assistant_response",
    "judge_prompt": PROMPT_TEMPLATE_ID,
    "score_range": [1, 5],
    "prompt_count": 200,
    "sample_count": 200,
    "sample_selection": "frozen_sharegpt_judge_prompt_manifest_order",
}


def build_absolute_prompt(user_prompt: str, response: str, reference: str) -> str:
    return f"""###Task Description:
An instruction, a response to evaluate, a reference answer that merits a score of 5, and a score rubric are given.
1. Write detailed feedback assessing the response strictly against the score rubric.
2. After the feedback, write an integer score from 1 to 5.
3. Use exactly this ending format: [RESULT] (an integer from 1 to 5)

###The instruction to evaluate:
{user_prompt}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubric:
{RUBRIC}

###Feedback:"""


def parse_score(text: str) -> int:
    matches = re.findall(r"\[RESULT\]\s*\(?\s*([0-5])\s*\)?", text, flags=re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(f"Unparseable or ambiguous judge score: {text[:160]!r}")
    score = int(matches[0])
    # Prometheus sometimes awards "[RESULT] 0" to degenerate candidates that
    # satisfy no rubric level (e.g. ignoring the prompt's required format). The
    # frozen protocol range is 1..5, so clamp to the minimum valid score; the
    # original judge text (including the 0) is preserved verbatim in judge_raw.
    return max(score, 1)


def _read_jsonl(path: Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid ShareGPT JSONL {path}: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"ShareGPT JSONL must contain nonempty object rows: {path}")
    return rows


def _resolved_artifact(path: Path, project_root: Path) -> tuple[str, Path]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(project_root.resolve()):
        raise RuntimeError(f"ShareGPT raw artifact is missing or outside project root: {path}")
    return str(resolved.relative_to(project_root.resolve())), resolved


def build_sharegpt_judge_row(
    candidate: dict, prompt_row: dict, example: dict, binding: dict, index: int, judge_raw: str,
) -> dict:
    """Normalize one candidate's official-format absolute judgment row."""
    expected = {
        "sample_index": index, "sample_id": binding["sample_id"],
        "sample_input_sha256": binding["sample_input_sha256"],
        "prompt_id": prompt_row["prompt_id"], "source_index": prompt_row["source_index"],
        "prompt": prompt_row["prompt"],
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"ShareGPT candidate ordered sample mismatch at row {index}")
    response = candidate.get("response")
    reference = candidate.get("reference_response")
    if (
        not isinstance(response, str) or not isinstance(reference, str)
        or reference != example["reference_response"]
        or candidate.get("reference_response_sha256") != canonical_hash(reference)
    ):
        raise RuntimeError(f"ShareGPT candidate/reference mismatch at row {index}")
    try:
        score = parse_score(judge_raw)
    except ValueError:
        # After the identical-input retry the judge reply is still unparseable
        # (rare greedy degeneration). Record the sample as unjudgeable (null)
        # with the raw output preserved verbatim; aggregation reports the
        # judgeable-sample count alongside the mean score.
        score = None
    return {
        **expected, "response": response, "reference_response": reference,
        "candidate_response_sha256": canonical_hash(response),
        "reference_response_sha256": canonical_hash(reference), "score": score,
        "judge_raw": judge_raw, "rubric": RUBRIC, "judge_prompt": PROMPT_TEMPLATE_ID,
    }


def validate_sharegpt_candidate_manifest(
    candidate_path: str | Path, *, run_id: str, checkpoint: int, project_root: str | Path = ROOT,
) -> tuple[dict, str, Path]:
    """Validate the candidate timing/hash sidecar and return its resolved identity."""
    candidate_path, project_root = Path(candidate_path), Path(project_root)
    candidate_label, candidate_resolved = _resolved_artifact(candidate_path, project_root)
    manifest_path = candidate_path.with_suffix(".manifest.json")
    manifest_label, manifest_resolved = _resolved_artifact(manifest_path, project_root)
    try:
        manifest = json.loads(manifest_resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid ShareGPT candidate manifest: {exc}") from exc
    expected = {
        "schema_version": 1, "run_id": run_id, "checkpoint": int(checkpoint),
        "sample_count": 200, "sample_set_hash": canonical_hash(formal_sample_manifest("sharegpt")),
        "candidate_artifact": candidate_label, "candidate_sha256": file_sha256(candidate_resolved),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("ShareGPT candidate manifest identity/hash/sample binding mismatch")
    candidate_seconds = manifest.get("candidate_generation_seconds")
    if (
        isinstance(candidate_seconds, bool) or not isinstance(candidate_seconds, (int, float))
        or not math.isfinite(float(candidate_seconds)) or candidate_seconds < 0
    ):
        raise RuntimeError("ShareGPT candidate generation time is invalid")
    return manifest, manifest_label, manifest_resolved


def validate_sharegpt_raw_artifacts(
    candidate_path: str | Path, judge_path: str | Path, *, run_id: str, checkpoint: int,
) -> tuple[dict[str, float], list[dict]]:
    """Recompute declared ShareGPT metrics from frozen candidate and absolute-judge rows."""
    candidates, judges = _read_jsonl(Path(candidate_path)), _read_jsonl(Path(judge_path))
    prompts = load_frozen_sharegpt_prompts()
    examples = formal_sharegpt_examples()
    bindings = formal_sample_entries("sharegpt")
    if len(candidates) != 200 or len(judges) != 200:
        raise RuntimeError("ShareGPT formal evaluation requires exactly 200 candidate and judge rows")
    total_nll, total_tokens, total_score = 0.0, 0, 0
    for index, (candidate, judge, prompt, example, binding) in enumerate(
        zip(candidates, judges, prompts, examples, bindings)
    ):
        common = {
            "sample_index": index, "sample_id": binding["sample_id"],
            "sample_input_sha256": binding["sample_input_sha256"],
            "prompt_id": prompt["prompt_id"], "source_index": prompt["source_index"],
            "prompt": prompt["prompt"],
        }
        if any(candidate.get(key) != value or judge.get(key) != value for key, value in common.items()):
            raise RuntimeError(f"ShareGPT ordered frozen sample binding mismatch at row {index}")
        if candidate.get("run_id") != run_id or candidate.get("checkpoint") != int(checkpoint):
            raise RuntimeError(f"ShareGPT candidate run/checkpoint identity mismatch at row {index}")
        response = candidate.get("response")
        reference = candidate.get("reference_response")
        if not isinstance(response, str) or not isinstance(reference, str) or not reference.strip():
            raise RuntimeError(f"ShareGPT candidate response is missing at row {index}")
        if (
            reference != example["reference_response"]
            or candidate.get("reference_response_sha256") != canonical_hash(reference)
        ):
            raise RuntimeError(f"ShareGPT frozen reference response mismatch at row {index}")
        if judge.get("response") != response or judge.get("candidate_response_sha256") != canonical_hash(response):
            raise RuntimeError(f"ShareGPT judge candidate response hash mismatch at row {index}")
        if (
            judge.get("reference_response") != reference
            or judge.get("reference_response_sha256") != canonical_hash(reference)
        ):
            raise RuntimeError(f"ShareGPT judge frozen reference response mismatch at row {index}")
        if judge.get("rubric") != RUBRIC or judge.get("judge_prompt") != PROMPT_TEMPLATE_ID:
            raise RuntimeError(f"ShareGPT judge prompt/rubric contract mismatch at row {index}")
        raw_score: int | None
        try:
            raw_score = parse_score(str(judge.get("judge_raw", "")))
        except ValueError:
            raw_score = None
        if judge.get("score") != raw_score:
            raise RuntimeError(f"ShareGPT judge score is inconsistent with raw output at row {index}")
        if raw_score is not None:
            total_score += raw_score
        nll_sum, token_count = candidate.get("nll_sum"), candidate.get("nll_token_count")
        if (
            isinstance(nll_sum, bool) or not isinstance(nll_sum, (int, float))
            or not math.isfinite(float(nll_sum)) or float(nll_sum) < 0
            or isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0
        ):
            raise RuntimeError(f"ShareGPT candidate NLL evidence is invalid at row {index}")
        total_nll += float(nll_sum)
        total_tokens += token_count
    judged_count = sum(1 for judge in judges if isinstance(judge.get("score"), int))
    unjudgeable = len(judges) - judged_count
    return {
        "heldout_nll": total_nll / total_tokens,
        "prometheus_absolute_score": total_score / judged_count if judged_count else None,
        "prometheus_judged_samples": judged_count,
        "prometheus_unjudgeable_samples": unjudgeable,
    }, judges


def write_sharegpt_evaluation_result(
    candidate_path: str | Path, judge_path: str | Path, result_path: str | Path, *,
    run_id: str, checkpoint: int, protocol: dict, evaluation_seconds: float,
    project_root: str | Path = ROOT,
) -> dict:
    """Validate raw evidence and write aggregation's canonical evaluation JSON contract."""
    candidate_path, judge_path, result_path = Path(candidate_path), Path(judge_path), Path(result_path)
    project_root = Path(project_root)
    expected_name = f"{run_id}__step_{int(checkpoint):06d}__sharegpt.json"
    if result_path.name != expected_name or result_path.exists():
        raise RuntimeError(f"ShareGPT evaluation result path must be new and named {expected_name}")
    if isinstance(evaluation_seconds, bool) or not isinstance(evaluation_seconds, (int, float)) or not math.isfinite(float(evaluation_seconds)) or evaluation_seconds < 0:
        raise RuntimeError("ShareGPT evaluation_seconds must be finite and nonnegative")
    if protocol != FORMAL_PROTOCOL:
        raise RuntimeError("ShareGPT matrix protocol does not match the frozen formal protocol")
    metrics, judge_rows = validate_sharegpt_raw_artifacts(
        candidate_path, judge_path, run_id=run_id, checkpoint=checkpoint,
    )
    candidate_label, candidate_resolved = _resolved_artifact(candidate_path, project_root)
    candidate_manifest, candidate_manifest_label, candidate_manifest_resolved = validate_sharegpt_candidate_manifest(
        candidate_path, run_id=run_id, checkpoint=checkpoint, project_root=project_root,
    )
    candidate_seconds = candidate_manifest["candidate_generation_seconds"]
    judge_label, judge_resolved = _resolved_artifact(judge_path, project_root)
    evaluation_config = formal_evaluation_config(run_id, checkpoint, "sharegpt", protocol)
    payload = {
        "run_id": run_id, "checkpoint": int(checkpoint), "task": "sharegpt",
        "metrics": metrics, "sample_count": len(judge_rows),
        "prediction_artifact": judge_label, "prediction_sha256": file_sha256(judge_resolved),
        "candidate_artifact": candidate_label, "candidate_sha256": file_sha256(candidate_resolved),
        "candidate_manifest_artifact": candidate_manifest_label,
        "candidate_manifest_sha256": file_sha256(candidate_manifest_resolved),
        "judge_artifact": judge_label, "judge_sha256": file_sha256(judge_resolved),
        "sample_set_hash": canonical_hash(formal_sample_manifest("sharegpt")),
        "evaluation_config": evaluation_config,
        "evaluation_config_hash": canonical_hash(evaluation_config),
        "candidate_generation_seconds": float(candidate_seconds),
        "judge_seconds": float(evaluation_seconds),
        "evaluation_seconds": float(candidate_seconds) + float(evaluation_seconds),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Raw candidate JSONL emitted by generate_sharegpt_candidates.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument(
        "--judge-path", type=Path, default=ROOT / "pretrained_models/prometheus-7b-v2.0",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        _, _, protocol, formal_count = resolve_evaluation_request(
            args.run_dir, args.checkpoint, "sharegpt",
        )
        if formal_count != 200 or protocol != FORMAL_PROTOCOL:
            raise RuntimeError("Run matrix does not declare the frozen formal ShareGPT protocol")
        paths = canonical_evaluation_paths(args.run_dir.name, args.checkpoint, "sharegpt")
        supplied = {"candidate": args.input, "judge": args.output, "result": args.result}
        for key, path in supplied.items():
            if path.resolve() != paths[key].resolve():
                raise RuntimeError(f"ShareGPT {key} path must be canonical: {paths[key]}")
        require_new_artifacts(paths, "judge", "result")
        evidence_ready, evidence_errors = prometheus_evidence_ready(
            ROOT / "revision_experiments/results/audits",
        )
        if not evidence_ready:
            raise RuntimeError("Prometheus checkpoint/smoke evidence failed: " + "; ".join(evidence_errors))
        if args.judge_path.resolve() != (ROOT / "pretrained_models/prometheus-7b-v2.0").resolve():
            raise RuntimeError("ShareGPT judge must use the verified local Prometheus 2 checkpoint")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    candidate_rows = _read_jsonl(args.input)
    prompts = load_frozen_sharegpt_prompts()
    examples = formal_sharegpt_examples()
    bindings = formal_sample_entries("sharegpt")
    if len(candidate_rows) != 200:
        raise SystemExit("ShareGPT judge requires exactly 200 frozen candidate rows")
    validate_sharegpt_candidate_manifest(
        args.input, run_id=args.run_dir.name, checkpoint=args.checkpoint, project_root=ROOT,
    )
    for index, (row, prompt_row, example, binding) in enumerate(
        zip(candidate_rows, prompts, examples, bindings)
    ):
        if row.get("run_id") != args.run_dir.name or row.get("checkpoint") != args.checkpoint:
            raise RuntimeError(f"ShareGPT candidate identity mismatch at row {index}")
        build_sharegpt_judge_row(row, prompt_row, example, binding, index, "Feedback: smoke [RESULT] 3")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("Local 7B judge requires an available CUDA device")
    started = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.judge_path), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(str(args.judge_path), torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True).to(args.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        for index, (row, prompt_row, example, binding) in enumerate(
            zip(candidate_rows, prompts, examples, bindings)
        ):
            prompt = build_absolute_prompt(row["prompt"], row["response"], row["reference_response"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=4096)
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            raw = ""
            for attempt in range(2):
                with torch.no_grad():
                    output = model.generate(**encoded, do_sample=False, max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                if re.search(r"\[RESULT\]", raw, flags=re.IGNORECASE):
                    break
                # Rare greedy degeneration yields an empty/unparseable judge
                # reply; one identical retry, then record the sample as
                # unjudgeable rather than failing the whole evaluation.
            judged = build_sharegpt_judge_row(row, prompt_row, example, binding, index, raw)
            target.write(json.dumps(judged, ensure_ascii=False) + "\n")
            target.flush()
    payload = write_sharegpt_evaluation_result(
        args.input, args.output, args.result, run_id=args.run_dir.name,
        checkpoint=args.checkpoint, protocol=protocol,
        evaluation_seconds=time.perf_counter() - started, project_root=ROOT,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
