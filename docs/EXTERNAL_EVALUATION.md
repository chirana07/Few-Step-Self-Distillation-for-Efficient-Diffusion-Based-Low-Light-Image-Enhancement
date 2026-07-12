# Step 5: external paired-dataset evaluation

This notebook evaluates the exact original-protocol FSD checkpoint with
SHA-256 `5d7bac873b0b915fe6c0679b103fea9afe25f70de3958cab8da3d8779d156a37`.

It covers LOL-v1 eval15, LOL-v2 Synthetic, LSRW Huawei, LSRW Nikon, and VE-LOL
Synthetic. Every run uses five-step DDIM, alpha 0, LPIPS-Alex, and stable
per-image latent seed namespace 99173. Missing targets and shape-mismatched
pairs are recorded in `pairing_audit.csv` and excluded without resizing.

Add these Kaggle inputs:

1. `external_evaluation_source.zip`.
2. The dataset containing the exact `student_distilled.pth` checkpoint.
3. The existing `Zip_file` dataset containing all five paired test sets.

Run `kaggle_external_evaluation.ipynb` and download
`/kaggle/working/external_evaluation_evidence.zip`.
