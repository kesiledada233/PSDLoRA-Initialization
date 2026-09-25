#!/usr/bin/env python3
"""Three-case Prometheus 2 generation/score smoke using the frozen formal prompt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revision_experiments.scripts.schema import file_sha256
from revision_experiments.scripts.sharegpt_judge import (
    FORMAL_PROTOCOL, SYSTEM_PROMPT, build_absolute_prompt, parse_score,
)


ROOT = Path(__file__).resolve().parents[2]
JUDGE = ROOT / "pretrained_models/prometheus-7b-v2.0"
CASES = (
    {"case": "correct", "instruction": "What is 2 + 2?", "response": "The answer is 4.", "reference": "The answer is 4."},
    {"case": "incorrect", "instruction": "What is 2 + 2?", "response": "The answer is 5.", "reference": "The answer is 4."},
    {"case": "helpful", "instruction": "Name one way to save water at home.", "response": "Fix leaking taps promptly.", "reference": "Fix leaking taps promptly."},
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "revision_experiments/results/audits/prometheus_scoring_smoke.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite judge smoke evidence: {args.output}")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("Prometheus scoring smoke requires an available CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(str(JUDGE), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(JUDGE), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    results = []
    for case in CASES:
        prompt = build_absolute_prompt(case["instruction"], case["response"], case["reference"])
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=4096)
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({**case, "judge_raw": raw, "score": parse_score(raw)})
    ordering_passed = next(row["score"] for row in results if row["case"] == "correct") > next(
        row["score"] for row in results if row["case"] == "incorrect"
    )
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_revision": FORMAL_PROTOCOL["judge_revision"],
        "checkpoint_verification_sha256": file_sha256(
            ROOT / "revision_experiments/results/audits/prometheus_checkpoint_verification.json"
        ),
        "protocol": FORMAL_PROTOCOL,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "cases": results,
        "parseable_count": len(results),
        "correct_above_incorrect": ordering_passed,
        "passed": ordering_passed and len(results) == 3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
