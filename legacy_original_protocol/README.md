# Legacy 22.91 Protocol

This directory preserves the source used for the original transferred-teacher
FSD result reported as 22.91 dB on LOL-v2 Real. It is retained separately from
the deterministic frozen-integrity implementation in `src/`.

## Split

The exact split is committed at:

`configs/lolv2_real_legacy_split_639_50.json`

The legacy loader sorts the 689 paired LOL-v2 Real training filenames and uses
the first 639 pairs for optimization and the final 50 pairs for validation.
The 100 official test pairs are never included in either training or
validation. The training seed is 42 and checkpoint selection uses five-step
validation.

## Checkpoint and evidence

Use `checkpoints/student_distilled_2291.pth` for the preserved result. Its
SHA-256 is recorded in `docs/CHECKPOINTS.md`. The result evidence is stored in
`results/headline/` and reports 22.909448 dB PSNR, 0.814582 SSIM, and 0.191376
LPIPS-Alex on 100 official test pairs.

The exact legacy source is included here so the split rule and training
implementation can be inspected independently of the newer integrity runner.
