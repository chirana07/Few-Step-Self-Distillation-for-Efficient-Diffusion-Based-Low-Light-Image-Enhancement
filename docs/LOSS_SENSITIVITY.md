# Validation-only teacher-loss sensitivity

This package addresses the reviewer request for loss-component ablations without
using the LOL-v2 Real official test split for tuning. Six compute-matched runs
start from the same source `final.pth`, use the immutable 620/69 training-derived
split, and select checkpoints with five-step validation and common per-image
latents.

The variants are the full transferred-teacher loss and removal of SSIM, VGG-19
perceptual, color-statistics, Sobel-gradient, or residual-TV loss. Charbonnier is
kept as the core reconstruction objective. All runs use the exact full-loss
weights `(1.0, 0.3, 0.1, 0.05, 0.2, 0.02)`.

## Kaggle inputs

1. `loss_sensitivity_source.zip` from this folder.
2. LOL-v2 with `Real_captured/Train/Low` and `Normal`.
3. The exact source checkpoint `final.pth` with SHA-256
   `e7f86ad8ba2baf11384e784346ef7f777cd0bd3abb2e4a33cc00809e6d16f374`.

Run `kaggle_loss_sensitivity.ipynb` on a T4 GPU with Internet enabled. Expected
runtime is approximately 4-7 hours, depending on Kaggle load. The notebook never
evaluates the official test split.

Download `/kaggle/working/loss_sensitivity_evidence.zip` after completion. The
study must be described as a continued-training sensitivity analysis, not as a
from-scratch causal attribution of the final model.
