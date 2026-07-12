# Step 4: validation-only FSD ablations

This package runs seven full-budget experiments on the immutable LOL-v2 Real
training/validation manifest:

- Teacher depth: K = 1, 3, 5, 10 at anchor weight 0.5.
- Anchor weight: 0.25, 0.5, 0.75, 1.0 at K = 5.

All runs use 20 epochs, seed 3407, 5-step checkpoint selection, alpha 0,
LPIPS-Alex validation, and the exact transferred teacher. The official test set
is never evaluated by this notebook.

Add LOL-v2, `teacher_illum_ft.pth`, and `fsd_validation_ablation_source.zip` to
a Kaggle T4 notebook. Open `kaggle_validation_ablations.ipynb` and run all cells.
The seven runs can take several hours, especially K=10. The script resumes from
completed checkpoints after an interruption.

Download `/kaggle/working/fsd_validation_ablation_evidence.zip` at completion.
