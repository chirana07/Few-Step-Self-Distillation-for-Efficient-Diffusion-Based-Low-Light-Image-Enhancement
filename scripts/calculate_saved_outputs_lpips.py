#!/usr/bin/env python3
"""Add LPIPS to the exact saved images from the revised 22.91 dB run."""

import argparse
import hashlib
import json
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="distilled_test")
    args = parser.parse_args()

    source_csv = args.results / "per_image.csv"
    generated_dir = args.results / args.split
    rows = pd.read_csv(source_csv)
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 source rows, found {len(rows)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    metric = lpips.LPIPS(net="alex").to(device).eval()
    verified = []

    for index, row in rows.iterrows():
        low_name = str(row["image"])
        image_id = Path(low_name).stem.removeprefix("low")
        generated_path = generated_dir / f"{args.split}_{low_name}"
        target_path = args.ground_truth / f"normal{image_id}.png"
        if not generated_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"Missing pair: {generated_path} | {target_path}")

        pred = Image.open(generated_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        if pred.size != target.size:
            raise RuntimeError(f"Shape mismatch for {low_name}: {pred.size} vs {target.size}")

        pred_np = np.asarray(pred)
        target_np = np.asarray(target)
        psnr = float(peak_signal_noise_ratio(target_np, pred_np, data_range=255))
        ssim = float(
            structural_similarity(target_np, pred_np, data_range=255, channel_axis=2)
        )
        if abs(psnr - float(row["psnr"])) > 1e-9:
            raise RuntimeError(f"Saved-image PSNR mismatch for {low_name}")
        if abs(ssim - float(row["ssim"])) > 1e-9:
            raise RuntimeError(f"Saved-image SSIM mismatch for {low_name}")

        pred_t = TF.to_tensor(pred).unsqueeze(0).to(device) * 2.0 - 1.0
        target_t = TF.to_tensor(target).unsqueeze(0).to(device) * 2.0 - 1.0
        with torch.inference_mode():
            lpips_value = float(metric(pred_t, target_t).item())

        verified.append(
            {
                "split": args.split,
                "image": low_name,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_value,
                "generated_sha256": sha256(generated_path),
            }
        )
        print(f"[{index + 1:03d}/100] {low_name} LPIPS={lpips_value:.6f}")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    per_image = pd.DataFrame(verified)
    per_image.to_csv(output / "per_image_with_lpips.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "split": args.split,
                "n": len(per_image),
                "psnr_mean": per_image.psnr.mean(),
                "psnr_std": per_image.psnr.std(ddof=0),
                "ssim_mean": per_image.ssim.mean(),
                "ssim_std": per_image.ssim.std(ddof=0),
                "lpips_mean": per_image.lpips.mean(),
                "lpips_std": per_image.lpips.std(ddof=0),
                "inference_steps": 5,
                "sampler": "ddim",
                "gate_alpha": 0.0,
                "lpips_backbone": "alex",
            }
        ]
    )
    summary.to_csv(output / "summary_with_lpips.csv", index=False)
    manifest = {
        "method": "post-hoc LPIPS on exact saved outputs",
        "source_per_image_csv": str(source_csv.resolve()),
        "source_per_image_csv_sha256": sha256(source_csv),
        "generated_directory": str(generated_dir.resolve()),
        "ground_truth_directory": str(args.ground_truth.resolve()),
        "verified_images": len(per_image),
        "lpips_backbone": "alex",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
