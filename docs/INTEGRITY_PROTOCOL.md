# Step 2: transferred-teacher integrity rerun

This package verifies the revised high-performing branch without changing the
existing model, paper, checkpoints, or results.

## Kaggle inputs

Add exactly these inputs:

1. This complete folder, preferably uploaded as
   `transferred_teacher_integrity_source.zip`.
2. LOL-v2 with `Real_captured/Train` and `Real_captured/Test`.
3. The exact `teacher_illum_ft.pth` checkpoint. The notebook enforces SHA-256
   `a92c2988ced5cb0fb8e403fb143913d8fc2837fdb31dbeb1634aa3ca031697a2`.

Upload `kaggle_transferred_teacher_integrity.ipynb`, select a T4 GPU, enable
Internet for LPIPS dependencies, and run through Stage 1 first. Start with the
single configured seed `3407`. Do not run Stage 3 until Stage 2 has finished and
selected ARR alpha from validation.

Expected Stage 1 duration on a T4 is roughly 60-90 minutes for the matched FSD
and anchor branches. Stage 2 and Stage 3 are inference-only.

Download `/kaggle/working/transferred_teacher_integrity_evidence.zip` after the
last cell. Keep the checkpoint folder in Kaggle until the summary is reviewed.
