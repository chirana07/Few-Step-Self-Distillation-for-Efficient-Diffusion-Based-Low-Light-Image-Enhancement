#!/usr/bin/env python3
"""Evaluate the exact 22.91 checkpoint on paired external LLIE datasets."""

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "src"
EXPECTED_CHECKPOINT_SHA256 = "5d7bac873b0b915fe6c0679b103fea9afe25f70de3958cab8da3d8779d156a37"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(items, label):
    items = sorted(set(items))
    if len(items) != 1:
        raise RuntimeError(f"Expected one {label}, found: {items}")
    return items[0]


def discover_splits(root: Path):
    directories = [path for path in root.rglob("*") if path.is_dir()]

    lolv1_low = unique([
        path for path in directories
        if path.name.strip().lower() == "low"
        and "lol-v1" in str(path.parent).lower()
        and (path.parent / "high").is_dir()
    ], "LOL-v1 eval15 low folder")

    lolv2_syn_low = unique([
        path for path in directories
        if path.name.lower() == "low"
        and "synthetic" in str(path.parent).lower()
        and "test" in str(path.parent).lower()
        and (path.parent / "Normal").is_dir()
    ], "LOL-v2 Synthetic test low folder")

    huawei_low = unique([
        path for path in directories
        if path.name.lower() == "low" and path.parent.name.lower() == "huawei"
        and (path.parent / "high").is_dir()
    ], "LSRW Huawei low folder")

    nikon_low = unique([
        path for path in directories
        if path.name.lower() == "low" and path.parent.name.lower() == "nikon"
        and (path.parent / "high").is_dir()
    ], "LSRW Nikon low folder")

    ve_low = unique([
        path for path in directories
        if path.name.lower() == "ve-lol-l-syn-low_test"
    ], "VE-LOL Synthetic low folder")
    ve_high = unique([
        path for path in directories
        if path.name.lower() == "ve-lol-l-syn-normal_test"
    ], "VE-LOL Synthetic normal folder")

    return {
        "lol_v1_eval15": (lolv1_low, lolv1_low.parent / "high"),
        "lolv2_synthetic": (lolv2_syn_low, lolv2_syn_low.parent / "Normal"),
        "lsrw_huawei": (huawei_low, huawei_low.parent / "high"),
        "lsrw_nikon": (nikon_low, nikon_low.parent / "high"),
        "ve_lol_synthetic": (ve_low, ve_high),
    }


def paired_target(high_dir: Path, low_name: str):
    exact = high_dir / low_name
    if exact.is_file():
        return exact
    if low_name.startswith("low"):
        normal = high_dir / low_name.replace("low", "normal", 1)
        if normal.is_file():
            return normal
    return None


def audit_pairs(split, low_dir, high_dir):
    rows = []
    input_psnr = []
    input_ssim = []
    low_images = sorted(
        path for path in low_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for low_path in low_images:
        high_path = paired_target(high_dir, low_path.name)
        status = "valid"
        low_size = ""
        high_size = ""
        if high_path is None:
            status = "missing_target"
        else:
            with Image.open(low_path) as low_image, Image.open(high_path) as high_image:
                low_rgb = low_image.convert("RGB")
                high_rgb = high_image.convert("RGB")
                low_size = f"{low_rgb.width}x{low_rgb.height}"
                high_size = f"{high_rgb.width}x{high_rgb.height}"
                if low_rgb.size != high_rgb.size:
                    status = "shape_mismatch"
                else:
                    low_np = np.asarray(low_rgb)
                    high_np = np.asarray(high_rgb)
                    input_psnr.append(peak_signal_noise_ratio(high_np, low_np, data_range=255))
                    input_ssim.append(structural_similarity(
                        high_np, low_np, data_range=255, channel_axis=2
                    ))
        rows.append({
            "split": split,
            "low_image": low_path.name,
            "high_image": high_path.name if high_path else "",
            "low_size": low_size,
            "high_size": high_size,
            "status": status,
        })
    baseline = {
        "input_n": len(input_psnr),
        "input_psnr": float(np.mean(input_psnr)),
        "input_ssim": float(np.mean(input_ssim)),
        "discovered_low": len(low_images),
        "excluded": len(low_images) - len(input_psnr),
    }
    return rows, baseline


def run(command):
    print("\nRunning:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=SRC, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=99173)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Wrong checkpoint SHA-256: {checkpoint_hash}; expected {EXPECTED_CHECKPOINT_SHA256}"
        )

    splits = discover_splits(args.data_root.resolve())
    output_root = ROOT / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    pairing_rows = []
    summary_rows = []

    for split, (low_dir, high_dir) in splits.items():
        audit, baseline = audit_pairs(split, low_dir, high_dir)
        pairing_rows.extend(audit)
        split_output = output_root / split
        summary_path = split_output / "summary.csv"
        if not summary_path.is_file():
            run([
                sys.executable, "evaluation.py",
                "--splits", f"{split}:{low_dir}:{high_dir}",
                "--checkpoint", checkpoint,
                "--inference-steps", "5",
                "--sampler", "ddim",
                "--gate-alpha", "0.0",
                "--seed", args.seed,
                "--results-root", split_output,
            ])
        with summary_path.open(newline="", encoding="utf-8") as handle:
            result = next(csv.DictReader(handle))
        if int(result["n"]) != baseline["input_n"]:
            raise RuntimeError(
                f"Pair audit/evaluator count mismatch for {split}: "
                f"{baseline['input_n']} vs {result['n']}"
            )
        summary_rows.append({
            "split": split,
            **baseline,
            "output_psnr": float(result["psnr_mean"]),
            "output_ssim": float(result["ssim_mean"]),
            "output_lpips": float(result["lpips_mean"]),
            "delta_psnr": float(result["psnr_mean"]) - baseline["input_psnr"],
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_seed": args.seed,
            "inference_steps": 5,
            "sampler": "ddim",
            "gate_alpha": 0.0,
        })

    with (output_root / "pairing_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairing_rows[0]))
        writer.writeheader()
        writer.writerows(pairing_rows)
    with (output_root / "external_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("\nSummary:", output_root / "external_summary.csv")


if __name__ == "__main__":
    main()
