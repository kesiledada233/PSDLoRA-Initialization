from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.audit_legacy_evidence import argparse_defaults, legacy_source_contract, loss_summary


class LegacyEvidenceTests(unittest.TestCase):
    def test_extracts_literal_defaults_without_importing_source(self):
        source = """
import argparse
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=1107)
p.add_argument('--targets', nargs='+', default=['q_proj', 'v_proj'])
p.add_argument('--flag', action='store_true')
"""
        self.assertEqual(argparse_defaults(source), {
            "seed": 1107, "targets": ["q_proj", "v_proj"], "flag": False,
        })

    def test_loss_summary_requires_contiguous_steps(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            path = Path(temporary) / "training_log.csv"
            path.write_text("step,train_loss\n1,3.0\n2,1.0\n", encoding="utf-8")
            summary = loss_summary(path, expected_steps=2)
            self.assertEqual(summary["legacy_sum_auc500"], 4.0)
            self.assertEqual(summary["trapezoid_auc_steps_1_to_500"], 2.0)

    def test_detects_unmasked_padding_contract(self):
        source = """
return {'labels': item['input_ids'].clone()}
tokenizer.pad_token = tokenizer.eos_token
loss = outputs.loss / args.grad_accum_steps
current_loss = loss.item() * args.grad_accum_steps
if step % args.grad_accum_steps == 0:
    pass
"""
        contract = legacy_source_contract(source)
        self.assertTrue(contract["labels_clone_input_ids"])
        self.assertFalse(contract["padding_labels_masked"])
        self.assertTrue(contract["reported_loss_restores_accumulation_divisor"])


if __name__ == "__main__":
    unittest.main()
