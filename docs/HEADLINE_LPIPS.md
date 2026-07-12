# Step 1: Verify the 22.91 dB result with LPIPS

This verification evaluates the exact revised `student_distilled.pth`
checkpoint on the 100-image LOL-v2 Real test split using 5-step DDIM, ARR
alpha 0, and LPIPS-Alex. It produces PSNR, SSIM, and LPIPS in one invocation.
The original saved-output set does not record a common latent seed; this step
therefore verifies metrics on those saved outputs rather than claiming a new
controlled random-latent comparison.

## Add these Kaggle inputs

1. The LOL-v2 dataset containing `Real_captured/Test/Low` and
   `Real_captured/Test/Normal`.
2. The Kaggle dataset containing `student_distilled.pth` from
   `final_kaggle_outputs_results/output/`.
3. Upload `lumidiff_2291_eval_source.zip` from this folder as a small Kaggle
   dataset.

Open `kaggle_verify_2291_lpips.ipynb`, enable a GPU, and run all cells in
order. The notebook refuses checkpoints whose SHA-256 does not equal:

`5d7bac873b0b915fe6c0679b103fea9afe25f70de3958cab8da3d8779d156a37`

Download only `/kaggle/working/lumidiff_2291_lpips_evidence.zip`. It contains
the summary, per-image metrics, and a run manifest, not the generated images.

## Completed local saved-output verification

The existing 100 generated images were also checked directly against LOL-v2
Real ground truth. Every recomputed PSNR and SSIM value matched the original
per-image CSV exactly. The consistent result is:

`PSNR 22.909448 | SSIM 0.814582 | LPIPS-Alex 0.191376 | n=100`

Evidence is stored in `lumidiff_2291_saved_output_lpips_evidence.zip`.
