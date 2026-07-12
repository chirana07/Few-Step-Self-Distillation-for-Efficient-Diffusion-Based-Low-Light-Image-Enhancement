# Dataset setup

Raw datasets are excluded because of size, redistribution restrictions, and the need for users to obtain them from the original sources.

## Required paired benchmark

LOL-v2 Real official test split:

```text
LOL-v2/Real_captured/Test/Low
LOL-v2/Real_captured/Test/Normal
```

The two folders must contain 100 matching pairs. Do not resize the images for metric computation. The evaluator pads inputs to the U-Net stride and crops outputs back to the original resolution.

## External paired evaluation

The paper's fixed-checkpoint stress test also uses paired subsets from LOL-v1 eval15, LOL-v2 Synthetic, LSRW Huawei/Nikon, and VE-LOL Synthetic. Use the folder mappings recorded in the relevant Kaggle notebook and preserve exact-size LSRW pairs.

## No-reference collections

Unpaired collections such as ExDark are not used for PSNR, SSIM, or LPIPS tables because they do not provide paired normal-light ground truth.

