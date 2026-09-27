# Legacy exploratory entry points (superseded)

These scripts belong to the exploratory pipeline of the original submission
(August 2026). They are retained only as the auditable origin of the legacy
behavior documented in `results/audits/`:

- Their dataset construction clones `input_ids` into `labels` with
  `padding='max_length'` and never replaces padding labels with the ignore
  index, so padded positions contributed to the loss. This is the
  label-padding convention analyzed in the revision's controlled diagnostic.

The corrected, audited pipeline that produced every result in
`results/` lives in `revision_experiments/scripts/`:

- Training entry: `train_revision.py` (invoked by `run_matrix.py`)
- Label handling: `training_support.py` (`labels[attention_mask == 0] = -100`,
  enforced by `tests/test_dataset_labels.py` and `execution_gates.py`)
- Checkpoint evaluation: `evaluate_checkpoints.py` / `run_evaluation_queue.py`
- Aggregation: `aggregate_results.py`

Do not use the scripts in this directory.
