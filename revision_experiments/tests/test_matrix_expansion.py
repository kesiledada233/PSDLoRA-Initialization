from __future__ import annotations

import unittest
from pathlib import Path

from revision_experiments.scripts.matrix import checkpoint_steps_for_run, expand_matrix, load_matrix


CONFIG = Path(__file__).resolve().parents[1] / "config"


class MatrixExpansionTests(unittest.TestCase):
    def test_expected_counts(self):
        expected = {
            "core_500step_matrix.yaml": 48,
            "mechanism_matrix.yaml": 48,
            "downstream_matrix.yaml": 63,
            "scope_matrix.yaml": 12,
            "baseline_search_matrix.yaml": 63,
        }
        for name, count in expected.items():
            with self.subTest(name=name):
                runs = expand_matrix(load_matrix(CONFIG / name))
                self.assertEqual(len(runs), count)
                self.assertEqual(len({run["run_id"] for run in runs}), count)

    def test_mechanism_runs_cannot_reuse_noninstrumented_core_runs(self):
        core = {run["run_id"] for run in expand_matrix(load_matrix(CONFIG / "core_500step_matrix.yaml"))}
        mechanism = {run["run_id"] for run in expand_matrix(load_matrix(CONFIG / "mechanism_matrix.yaml"))}
        self.assertFalse(core & mechanism)
        self.assertTrue(all(run_id.endswith("__gradlog") for run_id in mechanism))

    def test_baseline_screening_uses_method_specific_central_learning_rates(self):
        runs = expand_matrix(load_matrix(CONFIG / "baseline_search_matrix.yaml"))
        screening = [run for run in runs if run["section"] == "screening" and run["task"] == "gsm8k"]
        observed = {
            method: sorted(run["learning_rate"] for run in screening if run["search_method"] == method)
            for method in ("dora", "pissa", "lora_one", "proposed")
        }
        self.assertEqual(observed["dora"], [1.5e-4, 3.0e-4, 6.0e-4])
        self.assertEqual(observed["pissa"], [1.0e-5, 2.0e-5, 4.0e-5])
        self.assertEqual(observed["lora_one"], [2.5e-5, 5.0e-5, 1.0e-4])
        self.assertEqual(observed["proposed"], [5.0e-5, 5.0e-5, 5.0e-5])

    def test_selected_final_runs_declare_their_selection_dependency(self):
        runs = expand_matrix(load_matrix(CONFIG / "baseline_search_matrix.yaml"))
        selected = [
            run for run in runs
            if run["section"] == "final" and run["method"] != "peft_default"
        ]
        self.assertTrue(selected)
        self.assertTrue(all(run["selection_method"] in {"dora", "pissa", "lora_one", "proposed"} for run in selected))

    def test_baseline_final_runs_save_the_evaluable_endpoint(self):
        matrix = load_matrix(CONFIG / "baseline_search_matrix.yaml")
        final_run = next(run for run in expand_matrix(matrix) if run["section"] == "final")
        self.assertEqual(checkpoint_steps_for_run(matrix, final_run), {2500})
        self.assertEqual(matrix["final"]["task_metrics"], {
            "gsm8k": "exact_match", "cmmlu": "macro_accuracy",
            "mbpp": "pass_at_1", "sharegpt": "prometheus_absolute_score",
        })

    def test_deadline_scope_keeps_three_seed_all_linear_and_paired_single_seed_long_run(self):
        runs = expand_matrix(load_matrix(CONFIG / "scope_matrix.yaml"))
        all_linear = [run for run in runs if run["section"] == "all_linear"]
        long_run = [run for run in runs if run["section"] == "long_run"]
        self.assertEqual(len(all_linear), 9)
        self.assertEqual({run["task"] for run in all_linear}, {"cmmlu"})
        self.assertEqual({run["seed"] for run in all_linear}, {1107, 123, 42})
        self.assertEqual(len(long_run), 3)
        self.assertEqual({run["method"] for run in long_run}, {
            "peft_default", "iid_matched", "powerlaw_global_a06",
        })
        self.assertEqual({run["seed"] for run in long_run}, {1107})

    def test_deadline_baseline_keeps_four_task_screening_but_cmmlu_only_final(self):
        runs = expand_matrix(load_matrix(CONFIG / "baseline_search_matrix.yaml"))
        screening = [run for run in runs if run["section"] == "screening"]
        final = [run for run in runs if run["section"] == "final"]
        self.assertEqual(len(screening), 48)
        self.assertEqual({run["task"] for run in screening}, {
            "gsm8k", "cmmlu", "sharegpt", "mbpp",
        })
        self.assertEqual(len(final), 15)
        self.assertEqual({run["task"] for run in final}, {"cmmlu"})
        self.assertEqual({run["seed"] for run in final}, {1107, 123, 42})

    def test_lora_one_uses_its_declared_official_scaling_and_gradient_config(self):
        runs = expand_matrix(load_matrix(CONFIG / "baseline_search_matrix.yaml"))
        lora_one = [run for run in runs if run.get("search_method") == "lora_one" or run.get("selection_method") == "lora_one"]
        self.assertTrue(lora_one)
        for run in lora_one:
            self.assertEqual(run["training"]["lora_rank"], 16)
            self.assertEqual(run["training"]["lora_alpha"], 16)
            self.assertIs(run["training"]["use_rslora"], True)
            self.assertEqual(run["gradient_batches"], 8)
            self.assertEqual(run["gradient_batch_size"], 1)
            self.assertEqual(run["gradient_max_length"], 1024)
            self.assertEqual(run["stable_gamma"], 128)

    def test_integration_smoke_matrix_is_isolated_and_covers_three_risks(self):
        matrix = load_matrix(CONFIG / "smoke/integration_smoke.yaml")
        runs = expand_matrix(matrix)
        self.assertEqual(len(runs), 3)
        self.assertEqual({run["case"] for run in runs}, {
            "mbpp_adapter_and_executor", "lora_one_initialization", "gradient_logger_20step",
        })
        self.assertTrue(all("__smoke_" in run["run_id"] for run in runs))
        lora_one = next(run for run in runs if run["case"] == "lora_one_initialization")
        self.assertEqual(lora_one["gradient_batches"], 8)
        self.assertEqual(lora_one["training"]["lora_alpha"], 16)
        gradient = next(run for run in runs if run["case"] == "gradient_logger_20step")
        self.assertEqual(gradient["max_steps"], 20)
        self.assertTrue(gradient["logging"]["gradient_coordinates"])
        self.assertEqual(checkpoint_steps_for_run(matrix, gradient), {20})


if __name__ == "__main__":
    unittest.main()
