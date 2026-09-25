#!/usr/bin/env python3
"""Checkpoint evaluation dispatcher with immutable prediction artifacts."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import yaml

from revision_experiments.scripts.matrix import checkpoint_steps_for_run, expand_matrix, load_matrix
from revision_experiments.scripts.schema import (
    canonical_hash, file_sha256, formal_evaluation_config, formal_sample_entries, formal_sample_manifest,
    validate_run_directory,
)
from revision_experiments.scripts.safe_code_eval import evaluate_candidate
from revision_experiments.scripts.training_support import (
    MODEL_IDENTIFIERS, load_model, load_task_records, load_tokenizer,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "revision_experiments/results/runs"
EVALUATIONS_ROOT = ROOT / "revision_experiments/results/evaluations"
SMOKE_EVALUATIONS_ROOT = ROOT / "revision_experiments/results/smoke/evaluations"
SMOKE_RUNS_ROOT = ROOT / "revision_experiments/results/smoke/runs"
INTEGRATION_SMOKE_MATRIX = ROOT / "revision_experiments/config/smoke/integration_smoke.yaml"


def extract_number(text: str) -> str | None:
    """GSM8K final answer: the number after the FIRST '####' marker.

    Fine-tuned models frequently continue past their own answer, reproducing
    the consecutive-question training format; anything after the first marker
    is spurious regeneration. Fall back to the first number only when no
    canonical marker exists.
    """
    marker = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if marker:
        return marker.group(1).replace(",", "")
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return matches[0].replace(",", "") if matches else None


def extract_choice(text: str) -> str | None:
    """CMMLU choice: the FIRST answer marker answers the asked question."""
    matches = re.findall(r"(?:答案|answer)\s*[:：]?\s*([ABCD])\b", text, flags=re.I)
    if matches:
        return matches[0].upper()
    standalone = re.findall(r"\b([ABCD])\b", text.upper())
    return standalone[0] if standalone else None


def extract_code(text: str) -> str:
    """MBPP solution: the asked problem's code, robust to the prompt's trailing fence.

    The frozen 3-shot prompt ends with an opening ```python fence, so a
    compliant model continues directly with code and the first fence marker in
    the output is the answer's CLOSING fence; the answer is the text before
    it. If the model restates the problem instead (the head before the first
    fence mentions "Problem:"/"Solution:"), the first properly fenced block is
    the answer. Without any fence, the stripped text is the legacy fallback.
    """
    head, fence, _ = text.partition("```")
    if fence and head.strip() and not re.search(r"problem\s*:|solution\s*:", head, flags=re.I):
        return head.strip()
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    if blocks:
        return blocks[0].strip()
    # Fence-less outputs follow the training format (raw code); the model may
    # continue into the next regenerated '# Problem', so truncate there.
    return re.split(r"\n#?\s*Problem\b", text, maxsplit=1)[0].strip()


def _json_safe(value, path: str = "$", hits: list | None = None):
    """Make a prediction row JSON-serializable, recording byte-field probes.

    A frozen-protocol forensic probe: some evaluator outputs (sandbox-captured
    streams or frozen record fields) can surface as raw bytes depending on the
    adapter's generated content; json.dumps would then crash AFTER the GPU
    stage, losing the whole evaluation. Bytes are decoded losslessly-enough
    (backslashreplace) and every offending field path is recorded so the root
    source stays auditable in the immutable payload.
    """
    if isinstance(value, bytes):
        if hits is not None:
            hits.append(path)
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, dict):
        return {key: _json_safe(item, f"{path}.{key}", hits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[{index}]", hits) for index, item in enumerate(value)]
    return value


def generate(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_batch(model, tokenizer, prompts: list[str], device, max_new_tokens: int) -> list[str]:
    """Batched greedy generation for decoder-only models.

    Prompts are LEFT-padded so every row shares one prompt-tensor length and the
    continuation slice is uniform; padded positions are masked via the attention
    mask. Batching is an implementation detail of the same frozen greedy
    protocol (validated against single-stream output before formal use).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    finally:
        tokenizer.padding_side = original_side
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    prompt_len = encoded["input_ids"].shape[1]
    return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True) for row in output]


def generate_all(model, tokenizer, prompts: list[str], device, max_new_tokens: int, batch_size: int = 1) -> list[str]:
    """Generate continuations for ordered prompts; batch_size 1 keeps the legacy path.

    Batched mode groups prompts by character length before chunking (outputs
    are returned in the original order). Length bucketing minimizes left-
    padding waste on heterogeneous prompt sets; each sample's greedy generation
    is independent of its position within a batch.
    """
    if batch_size <= 1 or len(prompts) <= 1:
        return [generate(model, tokenizer, prompt, device, max_new_tokens) for prompt in prompts]
    order = sorted(range(len(prompts)), key=lambda index: (len(prompts[index]), index))
    sorted_prompts = [prompts[index] for index in order]
    sorted_outputs: list[str] = []
    for start in range(0, len(sorted_prompts), batch_size):
        sorted_outputs.extend(
            generate_batch(model, tokenizer, sorted_prompts[start:start + batch_size], device, max_new_tokens)
        )
    outputs: list[str] = [""] * len(prompts)
    for index, output in zip(order, sorted_outputs):
        outputs[index] = output
    return outputs


def load_adapter_model(run_dir: Path, checkpoint: int, device):
    from peft import PeftModel
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    matches = [key for key, identifier in MODEL_IDENTIFIERS.items() if metadata.get("model_checkpoint") == identifier]
    if len(matches) != 1:
        raise RuntimeError("Run metadata does not identify one pinned supported base checkpoint")
    model_key = matches[0]
    base = load_model(model_key, torch.device(device), "bf16")
    adapter = run_dir / "checkpoints" / f"step_{checkpoint:06d}"
    return PeftModel.from_pretrained(base, str(adapter)).eval(), model_key


def canonical_evaluation_paths(run_id: str, checkpoint: int, task: str) -> dict[str, Path]:
    stem = f"{run_id}__step_{int(checkpoint):06d}__{task}"
    paths = {"result": EVALUATIONS_ROOT / f"{stem}.json"}
    if task == "sharegpt":
        paths.update({
            "candidate": EVALUATIONS_ROOT / f"{stem}_candidates.jsonl",
            "candidate_manifest": EVALUATIONS_ROOT / f"{stem}_candidates.manifest.json",
            "judge": EVALUATIONS_ROOT / f"{stem}_judged.jsonl",
        })
    else:
        paths["prediction"] = EVALUATIONS_ROOT / f"{stem}.jsonl"
    return paths


def smoke_evaluation_paths(run_id: str, checkpoint: int, task: str, limit: int) -> dict[str, Path]:
    stem = f"{run_id}__step_{int(checkpoint):06d}__{task}__first_{int(limit):06d}_smoke"
    return {
        "result": SMOKE_EVALUATIONS_ROOT / f"{stem}.json",
        "prediction": SMOKE_EVALUATIONS_ROOT / f"{stem}.jsonl",
    }


def require_new_artifacts(paths: dict[str, Path], *keys: str) -> None:
    existing = [str(paths[key]) for key in keys if paths[key].exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite evaluation artifacts: " + ", ".join(existing))


def resolve_evaluation_request(
    run_dir: Path, checkpoint: int, task: str, *, smoke: bool = False,
) -> tuple[dict, dict, dict, int]:
    """Validate a completed formal/smoke run and its frozen evaluation contract."""
    run_dir = run_dir.resolve()
    expected_root = SMOKE_RUNS_ROOT if smoke else RUNS_ROOT
    if run_dir.parent != expected_root.resolve():
        label = "Smoke" if smoke else "Formal"
        raise RuntimeError(f"{label} run directory must be a direct child of {expected_root}")
    run_errors = validate_run_directory(run_dir)
    if run_errors:
        raise RuntimeError("Invalid completed run directory: " + "; ".join(run_errors))
    if not (run_dir / "COMPLETED").is_file() or not (run_dir / "summary.json").is_file():
        raise RuntimeError("Evaluation requires a completed run with summary.json")
    try:
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid run config/metadata: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise RuntimeError("Run config/metadata must be objects")
    if smoke:
        matrix_path = INTEGRATION_SMOKE_MATRIX
        smoke_matrix = load_matrix(matrix_path)
        if smoke_matrix.get("purpose") != "integration_smoke" or config.get("formal_result") is not False:
            raise RuntimeError("Smoke run does not declare the isolated integration-smoke contract")
    else:
        matrix_path = next(
            (path for path in (ROOT / "revision_experiments/config").glob("*_matrix.yaml")
             if load_matrix(path)["matrix_name"] == config.get("matrix")), None
        )
    if matrix_path is None:
        raise RuntimeError("Run config does not identify a declared matrix")
    matrix = load_matrix(matrix_path)
    run = next((row for row in expand_matrix(matrix) if row["run_id"] == run_dir.name), None)
    if run is None:
        raise RuntimeError("Run is not declared by its matrix")
    for key in ("run_id", "model", "task", "method", "seed", "max_steps", "target_modules"):
        if config.get(key) != run.get(key):
            raise RuntimeError(f"Run config differs from matrix declaration at {key}")
    if metadata.get("config_hash") != canonical_hash(config):
        raise RuntimeError("Run metadata config_hash does not bind config.yaml")
    if metadata.get("run_id") != run_dir.name or metadata.get("model_checkpoint") != MODEL_IDENTIFIERS[run["model"]]:
        raise RuntimeError("Run metadata identity/checkpoint does not match the matrix declaration")
    for key in ("method", "seed", "max_steps", "target_modules"):
        if metadata.get(key) != run.get(key):
            raise RuntimeError(f"Run metadata differs from matrix declaration at {key}")
    if metadata.get("tokenizer") != MODEL_IDENTIFIERS[run["model"]]:
        raise RuntimeError("Run tokenizer identity does not match the pinned model revision")
    if task != run["task"]:
        raise RuntimeError(f"Cannot evaluate {task} on a run trained for {run['task']}")
    if int(checkpoint) not in checkpoint_steps_for_run(matrix, run):
        raise RuntimeError("Requested checkpoint is not declared by the run matrix")
    adapter = run_dir / "checkpoints" / f"step_{int(checkpoint):06d}"
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError(f"Requested adapter checkpoint is incomplete: {adapter}")
    protocol = dict(
        matrix.get("evaluation", {}).get(task, {})
        or matrix.get("final", {}).get("evaluation", {}).get(task, {})
    )
    sample_manifest = formal_sample_manifest(task)
    formal_count = int(protocol.get("sample_count", -1))
    if formal_count != sample_manifest["sample_count"]:
        raise RuntimeError("Matrix sample count does not match frozen formal sample manifest")
    return matrix, run, protocol, formal_count


def gsm8k_prompt(train_records: list[dict], question: str) -> str:
    if len(train_records) < 8:
        raise ValueError("GSM8K fixed 8-shot evaluation requires at least 8 training records")
    shots = "\n\n".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}" for row in train_records[:8]
    )
    return (
        "Solve each problem step by step and finish with the numeric answer.\n\n"
        f"{shots}\n\nQuestion: {question}\nAnswer:"
    )


def build_evaluation_prompts(
    task: str, train_records: list[dict], records: list[dict], limit: int,
) -> list[str]:
    """Build the exact ordered generation prompts used by formal evaluation."""
    selected = records[:limit]
    if task == "gsm8k":
        return [gsm8k_prompt(train_records, record["question"]) for record in selected]
    if task == "cmmlu":
        by_subject = {}
        for row in train_records:
            by_subject.setdefault(row["subject"], []).append(row)
        prompts = []
        for record in selected:
            shots = by_subject.get(record["subject"], [])[:5]
            prefix = "".join(
                f"问题：{row['Question']}\nA. {row['A']} B. {row['B']} C. {row['C']} D. {row['D']}\n答案：{row['Answer']}\n\n"
                for row in shots
            )
            prompts.append(
                prefix + f"问题：{record['Question']}\nA. {record['A']} B. {record['B']} "
                f"C. {record['C']} D. {record['D']}\n答案："
            )
        return prompts
    if task == "mbpp":
        shots = "\n\n".join(
            f"Problem: {row['text']}\nSolution:\n```python\n{row['code']}\n```"
            for row in train_records[:3]
        )
        return [shots + f"\n\nProblem: {record['text']}\nSolution:\n```python\n" for record in selected]
    raise ValueError(f"unsupported prompt task: {task}")


def evaluate_gsm8k(model, tokenizer, train_records, records, device, limit, batch_size: int = 1):
    selected = records[:limit]
    prompts = build_evaluation_prompts("gsm8k", train_records, records, limit)
    outputs = generate_all(model, tokenizer, prompts, device, 256, batch_size)
    predictions = []
    correct = 0
    for sample_index, (record, prompt, output) in enumerate(zip(selected, prompts, outputs)):
        predicted = extract_number(output)
        expected = extract_number(record["answer"])
        correct += predicted == expected
        predictions.append({"sample_index": sample_index, "task_id": record.get("id"), "prompt": prompt, "output": output,
                            "predicted": predicted, "expected": expected, "correct": predicted == expected})
    return {"exact_match": correct / len(predictions)}, predictions


def evaluate_cmmlu(model, tokenizer, train_records, records, device, limit, batch_size: int = 1):
    selected = records[:limit]
    prompts = build_evaluation_prompts("cmmlu", train_records, records, limit)
    outputs = generate_all(model, tokenizer, prompts, device, 16, batch_size)
    predictions = []
    by_subject_results = {}
    for sample_index, (record, prompt, output) in enumerate(zip(selected, prompts, outputs)):
        predicted = extract_choice(output)
        expected = record["Answer"].strip().upper()
        by_subject_results.setdefault(record["subject"], []).append(predicted == expected)
        predictions.append({"sample_index": sample_index, "subject": record["subject"], "prompt": prompt, "output": output,
                            "predicted": predicted, "expected": expected, "correct": predicted == expected})
    macro = sum(sum(values) / len(values) for values in by_subject_results.values()) / len(by_subject_results)
    return {"macro_accuracy": macro}, predictions


def evaluate_mbpp(model, tokenizer, train_records, records, device, limit, batch_size: int = 1):
    selected = records[:limit]
    prompts = build_evaluation_prompts("mbpp", train_records, records, limit)
    # 1024 tokens: the 512-token cap truncated ~80% of openPangu formal
    # generations mid-code (SyntaxError by truncation, not capability).
    # The 512-token artifacts are archived under *.invalid_truncation_* and
    # the cap change is uniform across all arms/seeds/models.
    outputs = generate_all(model, tokenizer, prompts, device, 1024, batch_size)
    predictions = []
    passed = 0
    for sample_index, (record, prompt, output) in enumerate(zip(selected, prompts, outputs)):
        code = extract_code(output)
        execution = evaluate_candidate(code, record.get("test_setup_code") or "", record["test_list"])
        passed += bool(execution["passed"])
        predictions.append({"sample_index": sample_index, "task_id": record["task_id"], "prompt": prompt, "output": output,
                            "code": code, "execution": execution})
    # pass_at_1 = conventional harness reading (single-function NameError alias).
    # The strict (no-alias) aggregate is reported separately as
    # auxiliary_metrics in the result payload: the declared metrics schema is
    # frozen to exactly what the matrix config declares, and extra readings
    # must not break schema validation of the immutable result.
    return {"pass_at_1": passed / len(predictions)}, predictions


def rescore_predictions_from(
    source: Path, *, task: str, formal_count: int, records: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    """Re-derive scores from a preserved prediction JSONL (CPU-only).

    Generation was never broken; only the answer extraction was. The preserved
    rows keep the immutable raw outputs and prompts, so re-scoring with the
    current extractors reproduces exactly what a fresh evaluation would score,
    without spending GPU hours regenerating identical greedy outputs.
    """
    if task not in ("gsm8k", "mbpp"):
        raise RuntimeError("--rescore-from currently supports gsm8k and mbpp only")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != formal_count:
        raise RuntimeError(f"rescore source has {len(rows)} rows, expected {formal_count}")
    corrected = []
    if task == "gsm8k":
        for row in rows:
            if "output" not in row or "expected" not in row:
                raise RuntimeError("rescore source rows must carry 'output' and 'expected'")
            predicted = extract_number(row["output"])
            corrected.append({**row, "predicted": predicted, "correct": predicted == row["expected"]})
        metrics = {"exact_match": sum(1 for row in corrected if row["correct"]) / len(corrected)}
        return metrics, corrected
    # mbpp: re-extract the answer code and re-execute the frozen test list.
    if records is None:
        raise RuntimeError("mbpp rescore requires the formal task records")
    if len(records) < formal_count:
        raise RuntimeError("mbpp rescore records shorter than the formal sample count")
    for index, (row, record) in enumerate(zip(rows, records)):
        if "output" not in row or "task_id" not in row:
            raise RuntimeError("mbpp rescore source rows must carry 'output' and 'task_id'")
        if row["task_id"] != record["task_id"]:
            raise RuntimeError(f"mbpp rescore task_id mismatch at row {index}")
        code = extract_code(row["output"])
        execution = evaluate_candidate(code, record.get("test_setup_code") or "", record["test_list"])
        corrected.append({**row, "code": code, "execution": execution})
    metrics = {"pass_at_1": sum(1 for row in corrected if row["execution"]["passed"]) / len(corrected)}
    return metrics, corrected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--task", choices=["gsm8k", "cmmlu", "mbpp", "sharegpt"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Must equal the matrix's complete frozen formal sample count when supplied")
    parser.add_argument("--allow-code-execution", action="store_true",
                        help="Required for MBPP; run inside an offline container/user namespace")
    parser.add_argument("--smoke", action="store_true",
                        help="MBPP-only first-N integration smoke; writes outside formal evaluations")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Generation cap for the ShareGPT candidate stage")
    parser.add_argument("--generation-batch-size", type=int, default=1,
                        help="Implementation detail of the frozen greedy protocol; 1 keeps the "
                             "single-stream legacy path. Batched mode must pass the documented "
                             "equivalence validation before formal use.")
    parser.add_argument("--rescore-from", type=Path, default=None,
                        help="CPU-only rescore of a preserved prediction JSONL (same run/checkpoint/"
                             "task); re-extracts answers with the current extractors instead of "
                             "regenerating identical greedy outputs")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    try:
        matrix, run, protocol, formal_count = resolve_evaluation_request(
            args.run_dir, args.checkpoint, args.task, smoke=args.smoke,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.smoke:
        if args.task != "mbpp" or args.limit is None or args.limit >= formal_count:
            raise SystemExit("--smoke requires task mbpp and 0 < --limit < the formal sample count")
        paths = smoke_evaluation_paths(args.run_dir.name, args.checkpoint, args.task, args.limit)
    else:
        if args.limit is not None and args.limit != formal_count:
            raise SystemExit("--limit must equal the complete formal sample count unless --smoke is used")
        args.limit = formal_count
        paths = canonical_evaluation_paths(args.run_dir.name, args.checkpoint, args.task)
    try:
        if args.task == "sharegpt":
            require_new_artifacts(paths, "candidate", "candidate_manifest", "result")
        else:
            require_new_artifacts(paths, "prediction", "result")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.rescore_from is not None:
        if args.smoke:
            raise SystemExit("--rescore-from is a formal-only path")
        if args.task == "mbpp" and not args.allow_code_execution:
            raise SystemExit("MBPP rescore re-executes code and requires --allow-code-execution")
        started = time.perf_counter()
        rescore_records = None
        if args.task == "mbpp":
            _, rescore_records = load_task_records("mbpp")
        try:
            metrics, predictions = rescore_predictions_from(
                args.rescore_from, task=args.task, formal_count=formal_count, records=rescore_records,
            )
        except (RuntimeError, OSError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        sample_entries = formal_sample_entries(args.task)
        if len(sample_entries) != len(predictions):
            raise SystemExit("rescore source does not match the frozen formal sample set")
        for prediction, entry in zip(predictions, sample_entries):
            for key, value in entry.items():
                prediction[key] = value
        sample_manifest = formal_sample_manifest(args.task)
        byte_hits: list[str] = []
        predictions = [_json_safe(row, f"$[{index}]", byte_hits) for index, row in enumerate(predictions)]
        prediction_path = paths["prediction"]
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("x", encoding="utf-8") as target:
            target.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        evaluation_config = formal_evaluation_config(args.run_dir.name, args.checkpoint, args.task, protocol)
        payload = {
            "run_id": args.run_dir.name, "checkpoint": args.checkpoint, "task": args.task,
            "metrics": metrics, "sample_count": len(predictions),
            "prediction_artifact": str(prediction_path.relative_to(ROOT)),
            "prediction_sha256": file_sha256(prediction_path),
            "evaluation_seconds": time.perf_counter() - started,
            "sample_set_hash": canonical_hash(sample_manifest),
            "evaluation_config": evaluation_config,
            "evaluation_config_hash": canonical_hash(evaluation_config),
            "rescored": True,
            "rescore_source": str(args.rescore_from),
            "rescore_source_sha256": file_sha256(args.rescore_from),
        }
        if byte_hits:
            payload["sanitized_byte_fields"] = sorted(set(byte_hits))
        if args.task == "mbpp":
            payload["auxiliary_metrics"] = {
                "pass_at_1_strict": sum(1 for row in predictions if row["execution"].get("passed_strict", row["execution"]["passed"])) / len(predictions),
            }
        result_path = paths["result"]
        with result_path.open("x", encoding="utf-8") as target:
            target.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("Evaluation requires an available CUDA device")
    device = torch.device(args.device)
    started = time.perf_counter()
    model, model_key = load_adapter_model(args.run_dir, args.checkpoint, device)
    tokenizer = load_tokenizer(model_key)
    train_records, test_records = load_task_records(args.task)
    sample_manifest = formal_sample_manifest(args.task)
    if args.task == "sharegpt":
        from revision_experiments.scripts.generate_sharegpt_candidates import write_sharegpt_candidate_artifact
        manifest = write_sharegpt_candidate_artifact(
            paths["candidate"], model, tokenizer, run_id=args.run_dir.name,
            checkpoint=args.checkpoint, device=device, max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
        )
        print(json.dumps({
            "stage": "sharegpt_candidates", "formal_result_pending": True, **manifest,
        }, indent=2, ensure_ascii=False))
        return 0
    if args.task == "gsm8k":
        metrics, predictions = evaluate_gsm8k(model, tokenizer, train_records, test_records, device, args.limit, args.generation_batch_size)
    elif args.task == "cmmlu":
        metrics, predictions = evaluate_cmmlu(model, tokenizer, train_records, test_records, device, args.limit, args.generation_batch_size)
    else:
        if not args.allow_code_execution:
            raise SystemExit("MBPP requires --allow-code-execution and an offline sandbox")
        metrics, predictions = evaluate_mbpp(model, tokenizer, train_records, test_records, device, args.limit, args.generation_batch_size)
    sample_entries = formal_sample_entries(args.task)
    for prediction, entry in zip(predictions, sample_entries):
        prediction.update(entry)
    byte_hits: list[str] = []
    predictions = [_json_safe(row, f"$[{index}]", byte_hits) for index, row in enumerate(predictions)]
    prediction_path = paths["prediction"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("x", encoding="utf-8") as target:
        target.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    payload = {
        "run_id": args.run_dir.name, "checkpoint": args.checkpoint, "task": args.task,
        "metrics": metrics, "sample_count": len(predictions), "prediction_artifact": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": file_sha256(prediction_path), "evaluation_seconds": time.perf_counter() - started,
    }
    if args.smoke:
        payload.update({
            "schema_version": 1, "formal_result": False,
            "sample_selection": "first_n_of_frozen_formal_order",
            "formal_sample_count": formal_count,
        })
    else:
        evaluation_config = formal_evaluation_config(args.run_dir.name, args.checkpoint, args.task, protocol)
        payload.update({
            "sample_set_hash": canonical_hash(sample_manifest), "evaluation_config": evaluation_config,
            "evaluation_config_hash": canonical_hash(evaluation_config),
        })
    if byte_hits:
        payload["sanitized_byte_fields"] = sorted(set(byte_hits))
    if args.task == "mbpp":
        payload["auxiliary_metrics"] = {
            "pass_at_1_strict": sum(1 for row in predictions if row["execution"].get("passed_strict", row["execution"]["passed"])) / len(predictions),
        }
    result_path = paths["result"]
    with result_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
