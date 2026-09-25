from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.aggregate_results import validate_evaluation_artifact
from revision_experiments.scripts.generate_sharegpt_candidates import load_frozen_sharegpt_prompts
from revision_experiments.scripts.schema import (
    canonical_hash,
    dataset_split_hash,
    file_sha256,
    formal_sample_entries,
    formal_sample_manifest,
    formal_sharegpt_examples,
)
from revision_experiments.scripts.sharegpt_judge import (
    PROMPT_TEMPLATE_ID, RUBRIC,
    build_sharegpt_judge_row,
    write_sharegpt_evaluation_result,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_PROMPTS = ROOT / "revision_experiments/data/processed/sharegpt_judge_prompts.jsonl"
PROTOCOL = {
    "metrics": ["heldout_nll", "prometheus_absolute_score"],
    "judge": "prometheus-7b-v2.0",
    "judge_revision": "66ffb1fc20beebfb60a3964a957d9011723116c5",
    "comparison_reference": "frozen_first_assistant_response",
    "judge_prompt": "prometheus2_official_absolute_mistral_v1",
    "score_range": [1, 5],
    "prompt_count": 200,
    "sample_count": 200,
    "sample_selection": "frozen_sharegpt_judge_prompt_manifest_order",
}


class ShareGptEvaluationContractTests(unittest.TestCase):
    def test_candidate_loader_rejects_arbitrary_prompt_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            rogue = Path(temporary) / "prompts.jsonl"
            rogue.write_text(FROZEN_PROMPTS.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "committed frozen 200-prompt manifest"):
                load_frozen_sharegpt_prompts(rogue)

    def test_judge_row_builder_emits_validator_schema_for_frozen_reference(self):
        prompt = load_frozen_sharegpt_prompts()[0]
        example = formal_sharegpt_examples()[0]
        binding = formal_sample_entries("sharegpt")[0]
        candidate = {
            "sample_index": 0, "sample_id": binding["sample_id"],
            "sample_input_sha256": binding["sample_input_sha256"],
            **prompt, "response": "candidate", "reference_response": example["reference_response"],
            "reference_response_sha256": canonical_hash(example["reference_response"]),
        }
        raw = "Feedback: correct and helpful. [RESULT] 5"
        row = build_sharegpt_judge_row(candidate, prompt, example, binding, 0, raw)
        self.assertEqual(row["score"], 5)
        self.assertEqual(row["response"], "candidate")
        self.assertEqual(row["reference_response"], example["reference_response"])

    def test_formal_result_binds_raw_candidate_judge_and_ordered_samples(self):
        prompts = load_frozen_sharegpt_prompts()
        examples = formal_sharegpt_examples()
        entries = formal_sample_entries("sharegpt")
        run_id, checkpoint = "qwen__sharegpt__qv__peft_default__s42__n2500", 2500
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path, judge_path = root / "candidates.jsonl", root / "judge.jsonl"
            candidate_rows, judge_rows = [], []
            for index, (prompt, example, binding) in enumerate(zip(prompts, examples, entries)):
                response = f"candidate response {index}"
                reference = example["reference_response"]
                common = {
                    "sample_index": index, "sample_id": binding["sample_id"],
                    "sample_input_sha256": binding["sample_input_sha256"],
                    "prompt_id": prompt["prompt_id"], "source_index": prompt["source_index"],
                    "prompt": prompt["prompt"],
                }
                candidate_rows.append({
                    **common, "run_id": run_id, "checkpoint": checkpoint, "response": response,
                    "reference_response": reference,
                    "reference_response_sha256": canonical_hash(reference),
                    "nll_sum": 2.0, "nll_token_count": 2,
                })
                judge_rows.append({
                    **common, "response": response, "reference_response": reference,
                    "candidate_response_sha256": canonical_hash(response),
                    "reference_response_sha256": canonical_hash(reference),
                    "score": 5, "judge_raw": "Feedback: strong. [RESULT] 5", "rubric": RUBRIC,
                    "judge_prompt": PROMPT_TEMPLATE_ID,
                })
            candidate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
            )
            candidate_manifest = {
                "schema_version": 1, "run_id": run_id, "checkpoint": checkpoint,
                "sample_count": 200,
                "sample_set_hash": canonical_hash(formal_sample_manifest("sharegpt")),
                "candidate_artifact": candidate_path.name,
                "candidate_sha256": file_sha256(candidate_path),
                "candidate_generation_seconds": 3.5,
            }
            candidate_path.with_suffix(".manifest.json").write_text(
                json.dumps(candidate_manifest) + "\n", encoding="utf-8"
            )
            judge_path.write_text(
                "".join(json.dumps(row) + "\n" for row in judge_rows), encoding="utf-8"
            )
            result_path = root / f"{run_id}__step_002500__sharegpt.json"
            payload = write_sharegpt_evaluation_result(
                candidate_path, judge_path, result_path, run_id=run_id, checkpoint=checkpoint,
                protocol=PROTOCOL, evaluation_seconds=12.5, project_root=root,
            )
            self.assertEqual(payload["metrics"], {"heldout_nll": 1.0, "prometheus_absolute_score": 5.0,
                "prometheus_judged_samples": 200, "prometheus_unjudgeable_samples": 0})
            self.assertEqual(payload["candidate_sha256"], file_sha256(candidate_path))
            self.assertEqual(payload["judge_sha256"], file_sha256(judge_path))
            self.assertEqual(payload["candidate_generation_seconds"], 3.5)
            self.assertEqual(payload["judge_seconds"], 12.5)
            self.assertEqual(payload["evaluation_seconds"], 16.0)
            sample_manifest = formal_sample_manifest("sharegpt")
            evaluation_config = {
                "schema_version": 1, "run_id": run_id, "checkpoint": checkpoint,
                "task": "sharegpt", "protocol": PROTOCOL,
                "dataset_split_hash": dataset_split_hash("sharegpt", uses_validation_split=False),
                "sample_manifest": sample_manifest,
            }
            requirement = {
                "checkpoint": checkpoint, "task": "sharegpt",
                "metrics": ("heldout_nll", "prometheus_absolute_score"), "protocol": PROTOCOL,
                "dataset_split_hash": evaluation_config["dataset_split_hash"],
                "evaluation_config": evaluation_config,
                "sample_set_hash": canonical_hash(sample_manifest), "bind_formal_samples": True,
            }
            validate_evaluation_artifact(result_path, requirement, run_id, root)
            judge_path.write_text(judge_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "judge_sha256|prediction_sha256"):
                validate_evaluation_artifact(result_path, requirement, run_id, root)


if __name__ == "__main__":
    unittest.main()
