#!/usr/bin/env python3
"""Train and evaluate FSD hyperparameter ablations on validation only."""

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
EXPECTED_TEACHER_SHA256 = "a92c2988ced5cb0fb8e403fb143913d8fc2837fdb31dbeb1634aa3ca031697a2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command):
    print("\nRunning:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=SRC, check=True)


def dataset_root(path: Path) -> Path:
    candidates = [path, path / "LOL-v2", path / "LOL_v2"]
    for candidate in candidates:
        if (candidate / "Real_captured/Train/Low").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"LOL-v2 Real was not found under {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/lolv2_real_split_seed3407.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-seed", type=int, default=99173)
    args = parser.parse_args()

    data = dataset_root(args.dataset_root)
    teacher = args.teacher.resolve()
    manifest = args.manifest.resolve()
    if sha256(teacher) != EXPECTED_TEACHER_SHA256:
        raise ValueError("Wrong teacher_illum_ft.pth checkpoint")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    # The K=5, lambda_a=0.5 configuration is shared by both sweeps.
    configs = [
        ("K1_a050", 1, 0.50, "teacher_depth"),
        ("K3_a050", 3, 0.50, "teacher_depth"),
        ("K5_a050", 5, 0.50, "teacher_depth_and_anchor"),
        ("K10_a050", 10, 0.50, "teacher_depth"),
        ("K5_a025", 5, 0.25, "anchor_weight"),
        ("K5_a075", 5, 0.75, "anchor_weight"),
        ("K5_a100", 5, 1.00, "anchor_weight"),
    ]

    output_root = ROOT / "results"
    rows = []
    low = data / "Real_captured/Train/Low"
    high = data / "Real_captured/Train/Normal"

    for label, teacher_steps, anchor_weight, sweep in configs:
        checkpoint_dir = output_root / "checkpoints" / label
        checkpoint = checkpoint_dir / f"best_{label}.pth"
        if not checkpoint.is_file():
            run([
                sys.executable, "train_distill.py",
                "--layout", "lolv2_real",
                "--dataset-root", data,
                "--manifest", manifest,
                "--init-from", teacher,
                "--teacher-from", teacher,
                "--run-kind", "fsd",
                "--epochs", args.epochs,
                "--teacher-steps", teacher_steps,
                "--student-inference-steps", "5",
                "--lr", "1e-5",
                "--w-distill", "1.0",
                "--w-anchor", anchor_weight,
                "--batch-size", "4",
                "--crop-size", "256",
                "--val-every", "1",
                "--seed", args.seed,
                "--val-seed", args.eval_seed,
                "--output-dir", checkpoint_dir,
                "--tag", label,
            ])

        eval_dir = output_root / "validation" / label
        summary = eval_dir / "summary.csv"
        if not summary.is_file():
            run([
                sys.executable, "evaluation.py",
                "--splits", f"lolv2_real_val:{low}:{high}",
                "--checkpoint", checkpoint,
                "--inference-steps", "5",
                "--sampler", "ddim",
                "--gate-alpha", "0.0",
                "--seed", args.eval_seed,
                "--manifest", manifest,
                "--manifest-split", "val",
                "--results-root", eval_dir,
            ])

        with summary.open(newline="", encoding="utf-8") as handle:
            result = next(csv.DictReader(handle))
        metadata = torch.load(checkpoint, map_location="cpu")
        rows.append({
            "label": label,
            "sweep": sweep,
            "teacher_steps": teacher_steps,
            "anchor_weight": anchor_weight,
            "training_seed": args.seed,
            "evaluation_seed": args.eval_seed,
            "epochs_budget": args.epochs,
            "selected_epoch": int(metadata["epoch"]),
            "n": int(result["n"]),
            "psnr": float(result["psnr_mean"]),
            "ssim": float(result["ssim_mean"]),
            "lpips": float(result["lpips_mean"]),
            "checkpoint": str(checkpoint.resolve()),
        })

    output_root.mkdir(parents=True, exist_ok=True)
    aggregate = output_root / "validation_ablation_summary.csv"
    with aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nValidation-only ablation summary:", aggregate)


if __name__ == "__main__":
    main()
