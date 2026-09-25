"""Tests for the 2026-09-12 evaluation-checkpoint protocol amendment.

The amendment restricts formal evaluations to a declared subset of saved
checkpoints and deduplicates step-0 evaluations per model-task group, based on
the Gate-2 proof that every B=0 method is functionally equivalent at step 0.
Training checkpoints, run dictionaries, and per-run configuration hashes must
be unaffected so completed runs remain valid.
"""

import unittest
from pathlib import Path

from revision_experiments.scripts.aggregate_results import (
    _evaluation_checkpoints,
    _evaluation_requirements,
)
from revision_experiments.scripts.matrix import checkpoint_steps_for_run
from revision_experiments.scripts.matrix import expand_matrix, load_matrix


CONFIG = Path(__file__).resolve().parents[1] / "config"


def base_config(**overrides):
    config = {
        "matrix_name": "downstream_2500step",
        "models": ["openpangu"],
        "tasks": ["gsm8k"],
        "methods": ["peft_default", "powerlaw_global_a06"],
        "seeds": [1107, 123, 42],
        "max_steps": 2500,
        "checkpoints": [0, 100, 250, 500, 1000, 1500, 2500],
        "evaluation": {"gsm8k": {"metric": "exact_match", "sample_count": 1319, "sample_selection": "full_frozen_official_test_order"}},
    }
    config.update(overrides)
    return config


def base_run(**overrides):
    run = {
        "run_id": "openpangu__gsm8k__qv__peft_default__s1107__n2500",
        "model": "openpangu",
        "task": "gsm8k",
        "method": "peft_default",
        "seed": 1107,
        "max_steps": 2500,
        "target": "qv",
        "training": {"learning_rate": 5.0e-5},
    }
    run.update(overrides)
    return run


class EvaluationCheckpointAmendmentTest(unittest.TestCase):
    def test_deadline_downstream_contract_has_68_requirements_and_full_headline_final(self):
        config = load_matrix(CONFIG / "downstream_matrix.yaml")
        runs = expand_matrix(config)
        requirements = {
            (run["run_id"], checkpoint)
            for run in runs
            for checkpoint in _evaluation_checkpoints(config, run)
        }
        self.assertEqual(len(requirements), 68)  # option B: GSM8K curve keeps only step 1500
        headline_final = {
            run["run_id"]
            for run in runs
            if run["method"] in {"peft_default", "powerlaw_global_a06"}
        }
        observed_final = {run_id for run_id, checkpoint in requirements if checkpoint == 2500}
        self.assertTrue(headline_final <= observed_final)

    def test_rule_contract_selects_only_matching_run_dimensions(self):
        config = base_config(
            evaluation_rules=[
                {
                    "name": "headline_final",
                    "checkpoints": [2500],
                    "models": ["openpangu"],
                    "tasks": ["gsm8k"],
                    "methods": ["peft_default"],
                    "seeds": [1107],
                },
                {
                    "name": "descriptive_curve",
                    "checkpoints": [100, 500, 1500],
                    "models": ["openpangu"],
                    "tasks": ["gsm8k"],
                    "methods": ["peft_default", "powerlaw_global_a06"],
                    "seeds": [1107],
                },
            ],
        )
        self.assertEqual(
            _evaluation_checkpoints(config, base_run()),
            {100, 500, 1500, 2500},
        )
        self.assertEqual(
            _evaluation_checkpoints(
                config,
                base_run(
                    method="powerlaw_global_a06",
                    run_id="openpangu__gsm8k__qv__powerlaw_global_a06__s1107__n2500",
                ),
            ),
            {100, 500, 1500},
        )
        self.assertEqual(
            _evaluation_checkpoints(
                config,
                base_run(seed=123, run_id="openpangu__gsm8k__qv__peft_default__s123__n2500"),
            ),
            set(),
        )

    def test_rule_contract_rejects_unknown_checkpoint(self):
        config = base_config(
            evaluation_rules=[{
                "name": "bad_checkpoint",
                "checkpoints": [700],
                "models": ["openpangu"],
            }],
        )
        with self.assertRaisesRegex(RuntimeError, "not declared training checkpoints"):
            _evaluation_checkpoints(config, base_run())

    def test_rule_contract_rejects_unknown_selector(self):
        config = base_config(
            evaluation_rules=[{
                "name": "typo",
                "checkpoints": [2500],
                "model": ["openpangu"],
            }],
        )
        with self.assertRaisesRegex(RuntimeError, "unsupported selectors"):
            _evaluation_checkpoints(config, base_run())

    def test_default_contract_unchanged_without_amendment_fields(self):
        config, run = base_config(), base_run()
        self.assertEqual(
            _evaluation_checkpoints(config, run),
            checkpoint_steps_for_run(config, run),
        )
        self.assertEqual(len(_evaluation_requirements(config, run)), 7)

    def test_amendment_restricts_evaluations_to_declared_subset(self):
        config = base_config(evaluation_checkpoints=[100, 500, 1500, 2500])
        run = base_run()
        self.assertEqual(_evaluation_checkpoints(config, run), {100, 500, 1500, 2500})
        self.assertEqual(len(_evaluation_requirements(config, run)), 4)

    def test_amendment_rejects_undeclared_evaluation_checkpoints(self):
        config = base_config(evaluation_checkpoints=[700])
        with self.assertRaisesRegex(RuntimeError, "not declared training checkpoints"):
            _evaluation_checkpoints(config, run := base_run())

    def test_shared_step_zero_kept_only_for_representative_run(self):
        config = base_config(
            evaluation_checkpoints=[100, 500, 1500, 2500],
            step_zero_evaluation="shared_peft_default_first_seed",
        )
        representative = _evaluation_checkpoints(config, base_run())
        self.assertIn(0, representative)
        non_representative = _evaluation_checkpoints(
            config, base_run(method="powerlaw_global_a06", run_id="openpangu__gsm8k__qv__powerlaw_global_a06__s1107__n2500")
        )
        self.assertNotIn(0, non_representative)
        other_seed = _evaluation_checkpoints(config, base_run(seed=123, run_id="openpangu__gsm8k__qv__peft_default__s123__n2500"))
        self.assertNotIn(0, other_seed)

    def test_section_level_amendment_and_section_seed_pool(self):
        config = base_config()
        config["long_run"] = {
            "model_task_pairs": [["openpangu", "gsm8k"]],
            "methods": ["peft_default"],
            "seeds": [1107, 123, 42],
            "max_steps": 10000,
            "checkpoints": [0, 100, 250, 500, 1000, 2500, 5000, 7500, 10000],
            "evaluation_checkpoints": [100, 500, 2500, 10000],
            "step_zero_evaluation": "shared_peft_default_first_seed",
        }
        run = base_run(section="long_run", max_steps=10000)
        self.assertEqual(_evaluation_checkpoints(config, run), {0, 100, 500, 2500, 10000})
        run_other_seed = base_run(section="long_run", max_steps=10000, seed=42, run_id="openpangu__gsm8k__qv__peft_default__s42__n10000")
        self.assertEqual(_evaluation_checkpoints(config, run_other_seed), {100, 500, 2500, 10000})

    def test_step_zero_policy_requires_seed_pool(self):
        config = base_config(step_zero_evaluation="shared_peft_default_first_seed")
        del config["seeds"]
        with self.assertRaisesRegex(RuntimeError, "seed pool"):
            _evaluation_checkpoints(config, base_run())


if __name__ == "__main__":
    unittest.main()
