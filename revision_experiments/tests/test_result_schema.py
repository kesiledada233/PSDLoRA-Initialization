from __future__ import annotations

import unittest

from revision_experiments.scripts.schema import canonical_hash, validate_metadata


def valid_metadata():
    return {
        "run_id": "openpangu__cmmlu__qv__iid_matched__s1107__n500",
        "git_commit": "a" * 40, "dirty_worktree": False,
        "model_checkpoint": "openpangu/openPangu-Embedded-7B-V1.1@0ae1841cbd53f5218f2ce5dc63083d5382cfc9f5",
        "tokenizer": "openpangu/openPangu-Embedded-7B-V1.1@0ae1841cbd53f5218f2ce5dc63083d5382cfc9f5",
        "chat_template_hash": "b" * 64, "dataset_split_hash": "c" * 64,
        "method": "iid_matched", "seed": 1107, "init_seed": 1,
        "data_order_seed": 1107, "training_seed": 1107, "max_steps": 500,
        "target_modules": ["q_proj", "v_proj"], "environment": {"torch": "x"},
        "hardware": {"device": "cpu", "count": 1}, "config_hash": "d" * 64,
    }


class ResultSchemaTests(unittest.TestCase):
    def test_valid_metadata(self):
        self.assertEqual(validate_metadata(valid_metadata()), [])

    def test_rejects_missing_hash_and_dirty_tree(self):
        data = valid_metadata()
        del data["dataset_split_hash"]
        data["dirty_worktree"] = True
        errors = validate_metadata(data)
        self.assertTrue(any("missing field: dataset_split_hash" in error for error in errors))
        self.assertIn("dirty_worktree must be false", errors)

    def test_rejects_unregistered_method_and_ambiguous_checkpoint(self):
        data = valid_metadata()
        data["method"] = "mystery"
        data["model_checkpoint"] = "latest"
        errors = validate_metadata(data)
        self.assertTrue(any("method must be registered" in error for error in errors))
        self.assertTrue(any("model_checkpoint" in error for error in errors))

    def test_canonical_hash_is_order_independent(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
