# Reproducibility map

## Headline result

- Checkpoint: `student_distilled_2291.pth`.
- Dataset: LOL-v2 Real official test split, 100 pairs.
- Sampler: DDIM, 5 steps.
- ARR: `alpha=0` for the reported FSD headline row.
- Exact saved-output evidence: `results/headline/`.

## Step ablation

- Same exact FSD checkpoint for 5, 10, 20, 50, and 100 DDIM steps.
- ARR: `alpha=0`.
- Evidence: `results/table_vii_clean.csv`.

## Frozen integrity protocol

- Split manifest: `configs/lolv2_real_split_seed3407.json`.
- Fixed evaluation latent seed: `99173`.
- Evidence: `results/integrity/`.
