# PSDLoRA: Power-Spectral-Density LoRA Initialization

Training-free, data-free initialization for LoRA adapters. PSDLoRA synthesizes the LoRA `A` matrix from a power-law power-spectral-density (PSD) via inverse FFT at an explicit shape-based scale (`std = sqrt(2/(d_in + r))`), with `B = 0`. It requires no data, forward passes, gradient estimation, or decomposition.

## What this repository contains

- `revision_experiments/initializers/` — all initialization variants evaluated in the revision: power-law synthesis (alpha in {0.3, 0.6, 1.0}), scale-matched i.i.d., value permutation, flat spectrum (alpha = 0), row-/column-wise synthesis, and the PEFT-default baseline.
- `revision_experiments/scripts/` — the audited corrected pipeline used for every reported result: the training entry point `train_revision.py` (invoked by `run_matrix.py`), label masking in `training_support.py` (`labels[attention_mask == 0] = -100`, enforced by unit tests and execution gates), checkpoint evaluation (`evaluate_checkpoints.py`), and aggregation (`aggregate_results.py`). The superseded exploratory entry points are kept under `legacy/` for audit trail only.
- `revision_experiments/config/` — paired-seed configurations (seeds 1107/123/42; initialization and training/data-order random streams separated so that, at a given seed, all methods share the same training randomness and data order).
- `results/aggregate/` — the CSV/JSON tables underlying every table and figure of the revised manuscript: 48-trial equal-budget validation screening, three-seed endpoint metrics, construction-control matrix, initialization statistics audit, early-window PSD slopes, all-linear and extended-budget runs, paired-difference summaries, and the run inventory.
- `results/runs/` — per-run artifacts for all runs: raw per-step training loss (`raw_loss.jsonl`), timing (`timing.jsonl`), per-run `config.yaml`, `initialization_stats.json`, `metadata.json`, and `summary.json`. Model checkpoints are not redistributed.
- `results/audits/` — controlled diagnostics, including the same-batch label-masking audit of the legacy entry point (median loss 2.83 masked vs. 16.92 legacy on 100 fixed batches).
- `results/evaluations/` — per-checkpoint downstream evaluation outputs.
- `revision_experiments/tests/` — unit tests for the initializers and the aggregation checks.

## Reproducing

Python 3.10; see `requirements.txt`. Pinned environment: PEFT 0.17.1, Transformers 4.53.2. Models: openPangu-Embedded-7B-V1.1 (revision `0ae1841cbd53f5218f2ce5dc63083d5382cfc9f5`) and Qwen/Qwen2.5-7B (revision `e25af2efae60472008fbeaf5fb7c4274a87f78d4`). One step = one optimizer update at effective batch size 8.

```bash
pip install -r requirements.txt

# Training (audited entry point; see argparse for matrix/config options)
python revision_experiments/scripts/train_revision.py --help
python revision_experiments/scripts/run_matrix.py --help

# Checkpoint evaluation and aggregation
python revision_experiments/scripts/evaluate_checkpoints.py --help
python revision_experiments/scripts/aggregate_results.py
```

## Notes and scope

- The exploratory pipeline originally released with the first submission is superseded. Its legacy openPangu entry point copied input IDs into labels without masking padding positions, which produced anomalous absolute losses; the controlled audit is in `results/audits/`. Every result in this repository comes from the corrected pipeline.
- Matched-scale controls (i.i.d. samples rescaled to the same mean, standard deviation, and Frobenius norm) recover much of the early-loss advantage over PEFT-default; a substantial part of the benefit therefore follows from the initialization scale itself. Endpoint task results are reported with three paired seeds in the manuscript.
- Dataset licenses are respected: benchmarks are not redistributed; download instructions follow their original sources.

## License

The openPangu model artifacts used in this work are governed by the OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 (see OPENPANGU_LICENSE). No openPangu model weights are redistributed in this repository.
