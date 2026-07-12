# Revised teacher/student provenance audit

## Verified checkpoint lineage

The revised headline checkpoint was not distilled directly from the original
model trained from scratch. The actual author-trained lineage is:

1. Original source checkpoint (`final.pth`).
2. Additional supervised LOL-v2 Real fine-tuning requested for 50 epochs at
   256x256 crops and learning rate 1e-5.
3. Validation selected the fine-tuned checkpoint at epoch 4 and saved it as
   `teacher_illum_ft.pth`.
4. FSD and anchor-only branches both initialized from that same checkpoint,
   ran for at most 20 epochs, and used learning rate 1e-5, batch size 4,
   crop size 256, and seed 42.
5. The revised FSD checkpoint was selected at epoch 4; the anchor-only
   checkpoint was selected at epoch 14.

Teacher SHA-256:

`a92c2988ced5cb0fb8e403fb143913d8fc2837fdb31dbeb1634aa3ca031697a2`

## Illumination metadata finding

`teacher_illum_ft.pth`, `student_distilled.pth`, and `student_baseline.pth`
store `use_illum_prior=True`, but their tensors prove that the flag is stale:

| Checkpoint | `head.weight` | condition input | parameters |
|---|---|---|---:|
| Fine-tuned teacher | `[32, 6, 3, 3]` | 3 channels | 17,973,318 |
| FSD student | `[32, 6, 3, 3]` | 3 channels | 17,973,318 |
| Anchor-only | `[32, 6, 3, 3]` | 3 channels | 17,973,318 |

The model concatenates only the three-channel noisy residual and three-channel
low-light image. `IlluminationPrior` exists in `modules.py` but is never called
by `model.py`. Therefore these results must not be described as
illumination-prior or Retinex-conditioned results.

## Validation-step finding

The revised training source attempted to select checkpoints at five steps by
setting `conf.INFERENCE_STEPS = 5`. Its `validate()` function instead reads the
class attribute `Config.INFERENCE_STEPS`, which remains 20. The existing
teacher, FSD, and anchor checkpoints were therefore selected using 20-step
validation despite logs labelling the values as five-step validation.

This does not create test leakage, and FSD/anchor remain matched, but it must
not be described as five-step checkpoint selection. The accompanying rerun
fixes this and evaluates validation with explicit `inference_steps=5`.

## Training-loop finding

The archived `train_distill.py` performs one non-optimizing pass over the
training loader before its optimization pass. The checkpoints differ from the
teacher, confirming that optimization did occur, but the redundant pass wastes
time and advances data-augmentation and shuffle RNG state. The corrected rerun
contains one optimization pass only and records its seed and immutable split.

## Defensible paper description before corrected rerun

The current `22.91` result may be described as a transferred-teacher FSD run:

> We first continued supervised training of the author-trained residual
> diffusion checkpoint on LOL-v2 Real. A frozen checkpoint from this stage initialized
> two compute-matched 20-epoch branches: FSD with teacher-refinement and
> ground-truth anchor losses, and an anchor-only control with the distillation
> term removed. Both use the same six-channel residual-conditioned U-Net.

Do not claim that the teacher was trained from scratch in this experiment, that
the checkpoints use an illumination prior, or that the existing checkpoints
were selected using five-step validation.
