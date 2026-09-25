#!/usr/bin/env python3
"""Generate immutable ShareGPT responses for one adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

from revision_experiments.scripts.evaluate_checkpoints import (
    canonical_evaluation_paths, generate, generate_all, load_adapter_model, require_new_artifacts,
    resolve_evaluation_request,
)
from revision_experiments.scripts.schema import (
    canonical_hash, file_sha256, formal_sample_entries, formal_sample_manifest,
    formal_sharegpt_examples,
)
from revision_experiments.scripts.training_support import format_record, load_tokenizer


ROOT = Path(__file__).resolve().parents[2]
FROZEN_PROMPTS = ROOT / "revision_experiments/data/processed/sharegpt_judge_prompts.jsonl"


def load_frozen_sharegpt_prompts(path: str | Path = FROZEN_PROMPTS) -> list[dict]:
    """Load only the committed, sidecar-bound 200-prompt judge manifest."""
    path = Path(path)
    if path.resolve() != FROZEN_PROMPTS.resolve():
        raise RuntimeError("ShareGPT evaluation requires the committed frozen 200-prompt manifest")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid frozen ShareGPT prompt manifest: {exc}") from exc
    examples = formal_sharegpt_examples()
    bindings = formal_sample_entries("sharegpt")
    if (
        len(rows) != 200 or len(bindings) != 200 or len(examples) != 200
        or manifest.get("count") != 200
        or manifest.get("selection_seed") != 20260903
        or manifest.get("prompts_sha256") != file_sha256(path)
    ):
        raise RuntimeError("committed frozen 200-prompt manifest or sidecar is inconsistent")
    for index, (row, example, binding) in enumerate(zip(rows, examples, bindings)):
        if (
            set(row) != {"prompt_id", "source_index", "prompt"}
            or any(row.get(key) != example.get(key) for key in row)
            or row["prompt_id"] != binding["sample_id"]
            or canonical_hash(example) != binding["sample_input_sha256"]
            or not isinstance(row["source_index"], int)
            or not isinstance(row["prompt"], str) or not row["prompt"].strip()
        ):
            raise RuntimeError(f"frozen ShareGPT prompt binding mismatch at sample {index}")
    return rows


@torch.no_grad()
def heldout_sequence_nll(model, tokenizer, text: str, device, max_length: int = 512) -> tuple[float, int]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_count = int(encoded["attention_mask"][:, 1:].sum().item())
    if token_count <= 0:
        raise RuntimeError("ShareGPT held-out sequence has no scored continuation tokens")
    output = model(**encoded, labels=encoded["input_ids"])
    nll_sum = float(output.loss.detach()) * token_count
    if not math.isfinite(nll_sum) or nll_sum < 0:
        raise RuntimeError("ShareGPT held-out sequence produced invalid NLL")
    return nll_sum, token_count


def generate_sharegpt_candidate_rows(
    model, tokenizer, *, run_id: str, checkpoint: int, device, max_new_tokens: int,
    batch_size: int = 1,
) -> list[dict]:
    """Generate the frozen candidate responses and source-bound NLL evidence."""
    prompts = load_frozen_sharegpt_prompts()
    examples = formal_sharegpt_examples()
    bindings = formal_sample_entries("sharegpt")
    source = ROOT / "pretrained_models/sharegpt_datasets/computer_en_26k.jsonl"
    source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    responses = generate_all(
        model, tokenizer, [prompt["prompt"] for prompt in prompts], device, max_new_tokens, batch_size,
    )
    # Fine-tuned models may continue past their reply into a regenerated next
    # "human:" turn (the linearized training format). Only the model's own
    # first reply is the candidate response; truncate at the next-turn marker.
    responses = [
        re.split(r"\n\s*human\s*:", response, maxsplit=1, flags=re.I)[0].rstrip()
        for response in responses
    ]
    nll_evidence = [
        heldout_sequence_nll(
            model, tokenizer, format_record("sharegpt", source_rows[prompt["source_index"]]), device,
        )
        for prompt in prompts
    ]
    rows = []
    for sample_index, (prompt, example, binding, response, (nll_sum, nll_token_count)) in enumerate(
        zip(prompts, examples, bindings, responses, nll_evidence)
    ):
        reference = example["reference_response"]
        rows.append({
            **prompt, "sample_index": sample_index, **binding,
            "run_id": run_id, "checkpoint": int(checkpoint),
            "response": response,
            "reference_response": reference,
            "reference_response_sha256": canonical_hash(reference),
            "nll_sum": nll_sum, "nll_token_count": nll_token_count,
        })
    return rows


def write_sharegpt_candidate_artifact(
    output: str | Path, model, tokenizer, *, run_id: str, checkpoint: int,
    device, max_new_tokens: int, batch_size: int = 1,
) -> dict:
    """Write one immutable candidate JSONL plus a timing/hash sidecar."""
    output = Path(output)
    if not output.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("ShareGPT candidate artifact must remain inside the project root")
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise RuntimeError(f"Refusing to overwrite ShareGPT candidate artifact: {output}")
    started = time.perf_counter()
    rows = generate_sharegpt_candidate_rows(
        model, tokenizer, run_id=run_id, checkpoint=checkpoint,
        device=device, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 1, "run_id": run_id, "checkpoint": int(checkpoint),
        "sample_count": len(rows), "sample_set_hash": canonical_hash(formal_sample_manifest("sharegpt")),
        "candidate_artifact": str(output.resolve().relative_to(ROOT.resolve())),
        "candidate_sha256": file_sha256(output),
        "candidate_generation_seconds": time.perf_counter() - started,
    }
    sidecar.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_artifact"] = str(sidecar.resolve().relative_to(ROOT.resolve()))
    manifest["manifest_sha256"] = file_sha256(sidecar)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--prompts", type=Path, default=FROZEN_PROMPTS,
                        help="Must be the committed frozen 200-prompt manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    load_frozen_sharegpt_prompts(args.prompts)
    try:
        _, _, protocol, formal_count = resolve_evaluation_request(
            args.run_dir, args.checkpoint, "sharegpt",
        )
        if formal_count != 200 or int(protocol.get("prompt_count", -1)) != 200:
            raise RuntimeError("ShareGPT matrix does not declare the frozen 200-prompt protocol")
        paths = canonical_evaluation_paths(args.run_dir.name, args.checkpoint, "sharegpt")
        if args.output.resolve() != paths["candidate"].resolve():
            raise RuntimeError(f"Candidate output must be the canonical path: {paths['candidate']}")
        require_new_artifacts(paths, "candidate", "candidate_manifest", "result")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("Candidate generation requires an available CUDA device")
    model, model_key = load_adapter_model(args.run_dir, args.checkpoint, torch.device(args.device))
    tokenizer = load_tokenizer(model_key)
    manifest = write_sharegpt_candidate_artifact(
        args.output, model, tokenizer, run_id=args.run_dir.name, checkpoint=args.checkpoint,
        device=args.device, max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
