# Controlled FSD comparison

Both revised checkpoints were initialized from the same fine-tuned teacher,
used the same LOL-v2 Real training/validation data, crop size, epoch budget,
student sampler, anchor loss, and checkpoint-selection procedure. The
anchor-only control sets the distillation weight to zero; the FSD run retains
the distillation loss.

| Configuration | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| Anchor-only | 21.5000 | 0.8044 | 0.2192 |
| FSD | 22.9094 | 0.8146 | 0.1914 |
| FSD - anchor | +1.4095 | +0.01014 | -0.02779 |

Paired image-level bootstrap with 20,000 resamples and seed 3407:

| Metric difference | Mean | 95% bootstrap CI | FSD win rate |
|---|---:|---:|---:|
| PSNR | +1.4095 dB | [0.9429, 1.9150] | 74% |
| SSIM | +0.01014 | [0.00401, 0.01653] | 61% |
| LPIPS | -0.02779 | [-0.03631, -0.01998] | 74% |

These intervals quantify paired image-level variation. They are not
multi-training-seed confidence intervals.
