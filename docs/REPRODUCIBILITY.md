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
- Training seed: 42; the transferred, FSD, and anchor checkpoints were
  selected with 20-step validation. Final test evaluation uses five-step
  DDIM.
- This is the provenance record for the original 22.91 dB result. It is
  distinct from the frozen 620/69 integrity protocol below.

## Step ablation

- Same exact FSD checkpoint for 5, 10, 20, 50, and 100 DDIM steps.
- Dataset: LOL-v2 Real official test split, 100 paired images.
- ARR: `alpha=0`.
- Fixed per-image latent seed: `99173`.
- Gate floor: `0.5`.
- Checkpoint SHA256: `5d7bac873b0b915fe6c0679b103fea9afe25f70de3958cab8da3d8779d156a37`.
- Evidence: `results/step_ablation/table_vii_fixed_latent.csv`.
- Per-image evidence and protocol records: `results/step_ablation/fixed_latent/`.
- The earlier saved-output table is retained as
  `results/step_ablation/table_vii_legacy_saved_outputs.csv`.

## Frozen integrity protocol

- Split manifest: `configs/lolv2_real_split_seed3407.json`.
- Fixed evaluation latent seed: `99173`.
- Evidence: `results/integrity/`.
- The executable manifest-aware implementation is in `src/`; run it through
  `scripts/run_integrity_protocol.py` rather than the legacy root training loop.

## Fixed-checkpoint cross-dataset evaluation

- Checkpoint SHA256: `5d7bac873b0b915fe6c0679b103fea9afe25f70de3958cab8da3d8779d156a37`.
- Datasets: LOL-v1 eval15, LOL-v2 Synthetic, LSRW Huawei/Nikon, and VE-LOL Synthetic.
- Sampler: DDIM, 5 steps; ARR `alpha=0`; fixed latent seed `99173`.
- Exact-size low/high pairs only; mismatched LSRW pairs are recorded in the audit
  and excluded without resizing.
- Aggregate evidence: `results/external_evaluation/external_summary.csv`.
- Pairing audit: `results/external_evaluation/pairing_audit.csv`.
- Per-dataset summaries, protocol records, and per-image metrics are in the
  corresponding subdirectories under `results/external_evaluation/`.
