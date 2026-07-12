# Results Usage Decision

Date: 2026-07-09

## New Kaggle Results

The first one-by-one Kaggle run used `final.pth`, which is the teacher
checkpoint. The second one-by-one run in `mercon_one_by_one_outputs 2_distill`
uses `best_distill_K5.pth`, which is the correct FSD student checkpoint.

The distilled one-by-one runs are useful, but they should not be presented as
a broad SOTA table. They are best used as zero-shot cross-dataset evidence.

| Dataset | n | PSNR | SSIM | LPIPS | Raw-input PSNR | Raw-input SSIM | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| LOL-v2 Real | 100 | 23.789 | 0.806 | 0.200 | 9.718 | 0.196 | Keep as direct FSD+ARR+LPIPS result. |
| LOL-v1 eval15 | 15 | 18.412 | 0.754 | 0.250 | 7.773 | 0.191 | Keep as legacy benchmark evidence. |
| LOL-v2 Synthetic | 100 | 16.142 | 0.768 | 0.231 | 11.221 | 0.450 | Keep as zero-shot generalization, not SOTA. |
| LSRW Huawei | 23 | 17.699 | 0.529 | 0.413 | 9.039 | 0.142 | Keep only with note that 7 non-aligned pairs were skipped. |
| LSRW Nikon | 16 | 15.582 | 0.421 | 0.230 | 9.331 | 0.242 | Keep only with note that 4 non-aligned pairs were skipped. |
| VE-LOL Synthetic | 100 | 14.484 | 0.465 | 0.592 | 10.760 | 0.305 | Include only if we explicitly discuss domain shift. |

VE-LOL Captured/Real was run as `ve_lol_cap_check_only` but should not be
reported as an independent benchmark because it appears to duplicate LOL-v2 Real.

## Recommended Paper Strategy

1. Do not add these rows to the SOTA comparison table.
2. Add a separate compact table titled "Zero-shot cross-dataset evaluation".
3. Include the raw low-light input baseline or PSNR/SSIM gain over input. This
   prevents the table from looking like a weak SOTA comparison.
4. Use cautious wording:
   - "The model improves over the degraded input on all tested paired datasets."
   - "Performance drops on LSRW and VE-LOL Synthetic, indicating domain shift."
   - "No dataset-specific fine-tuning is performed."
5. Keep VE-LOL Captured out of the paper table.
6. Replace the paper's `pending` LPIPS with the direct distilled LOL-v2 Real
   LPIPS value 0.200.

## Important Consistency Issue Resolved

The 23.86 dB / 0.807 SSIM result comes from:

`kaggle/working/eval_results_distill/arr_grid_arr_a050/summary.csv`

That run was produced by `kaggle_eval_distill.ipynb`, Cell 9, using the
distilled student checkpoint selected as `best_distill_K5.pth` from the
`lumidiff-distill-ckpt` Kaggle dataset. It was run with `--gate-alpha 0.5`,
`--gate-floor 0.5`, `--inference-steps 5`, and `--no-lpips`.

The newer one-by-one run used `final.pth`, which is the Day 1 teacher
checkpoint, not the FSD student checkpoint. Therefore its LOL-v2 Real result
of 23.423 dB / 0.791 SSIM / 0.218 LPIPS should not replace the paper's FSD
student headline.

For camera-ready defensibility:

1. The old ARR grid supports 23.86 / 0.807 for PSNR/SSIM from saved predictions.
2. The direct distilled one-by-one LPIPS run gives 23.789 / 0.806 / 0.200.
3. We keep the stronger 23.86 / 0.807 PSNR/SSIM values because they are backed
   by the saved ARR-grid summary and prediction images.
4. We use 0.200 LPIPS from the direct distilled LPIPS-enabled rerun. The rerun
   is slightly lower in PSNR (23.789 / 0.806), but uses the same
   `best_distill_K5.pth`, 5-step ARR configuration, and confirms the perceptual
   metric value needed by the reviewers.
