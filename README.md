# LUMIDIFF: Few-Step Self-Distillation for Low-Light Enhancement

This repository contains the LUMIDIFF residual-diffusion model, five-step
self-distillation code, evaluation scripts, validated manifests, and compact
experiment results.

## Repository layout

- Root Python files: model, diffusion process, dataset loader, training, distillation, and evaluation code.
- `src/`: manifest-aware, deterministic source used by the integrity and validation protocols.
- `legacy_original_protocol/`: source and split documentation for the original 639/50 transferred-teacher protocol.
- `scripts/`: ablations, efficiency measurement, figure generation, and integrity checks.
- `notebooks/`: Kaggle notebooks used for the reproducibility runs.
- `configs/`: immutable split manifests and run configuration records.
- `results/`: compact CSV/JSON evidence; raw generated image directories are intentionally excluded.
- `checkpoints/`: model checkpoints tracked with Git LFS or distributed through a GitHub Release.
- `figures/`: diagnostic visualizations and model-output examples.
- `docs/`: dataset, checkpoint, and reproducibility notes.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Datasets are not committed to this repository. Follow `docs/DATASETS.md` and provide local paths to the paired test data. The evaluator expects low-light and normal-light folders with matching filenames.

## Evaluation

The preserved 22.91 dB result uses the legacy 639/50 training-derived split
documented in `legacy_original_protocol/`. The independently audited result
uses the 620/69 manifest and deterministic source under `src/`. Both evaluate
the official 100-image test split only after their respective choices are
frozen. The main five-step evaluation command is:

```bash
python evaluation.py \
  --splits lolv2_real:/path/to/LOL-v2/Real_captured/Test/Low:/path/to/LOL-v2/Real_captured/Test/Normal \
  --inference-steps 5 \
  --sampler ddim \
  --checkpoint checkpoints/student_distilled_2291.pth \
  --gate-alpha 0 \
  --gate-floor 0.5 \
  --results-root results/evaluation \
  --tag headline_s5_a000
```

See `docs/REPRODUCIBILITY.md` for the exact checkpoint hashes, protocols, and table-to-evidence mapping.

## Large files

Checkpoints are larger than GitHub's normal file limit. Use Git LFS for tracked checkpoints or attach them to a GitHub Release. Verify every downloaded checkpoint against `docs/CHECKPOINTS.md` before evaluation.
