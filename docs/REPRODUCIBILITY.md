# Reproducibility map

## Headline result

- Checkpoint: `student_distilled_2291.pth`.
- Dataset: LOL-v2 Real official test split, 100 pairs.
- Sampler: DDIM, 5 steps.
- ARR: `alpha=0` for the reported FSD headline row.
- Exact saved-output evidence: `results/headline/`.

## Legacy 639/50 protocol

- Source: `legacy_original_protocol/`.
- Manifest: `configs/lolv2_real_legacy_split_639_50.json`.
- Split: first 639 sorted LOL-v2 Real training pairs for optimization and final
  50 pairs for validation; the official 100 test pairs are separate.
- Training seed: 42; validation and checkpoint selection use five DDIM steps.
- This is the provenance record for the original 22.91 dB result. It is
  distinct from the frozen 620/69 integrity protocol below.

## Step ablation

- Same exact FSD checkpoint for 5, 10, 20, 50, and 100 DDIM steps.
- ARR: `alpha=0`.
- Evidence: `results/table_vii_clean.csv`.

## Frozen integrity protocol

- Split manifest: `configs/lolv2_real_split_seed3407.json`.
- Fixed evaluation latent seed: `99173`.
- Evidence: `results/integrity/`.
- The executable manifest-aware implementation is in `src/`; run it through
  `scripts/run_integrity_protocol.py` rather than the legacy root training loop.
