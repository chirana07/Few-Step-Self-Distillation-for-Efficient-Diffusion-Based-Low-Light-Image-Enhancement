"""
ablate_method.py — method ablation: baseline vs +illumination-prior.

The "residual warm-start" change A is already present in the model (gated residual
head + residual-space diffusion), so the interesting axis left to ablate is the
illumination prior.

This trains two models:
    A. baseline          : use_illum_prior=False
    B. +illum_prior      : use_illum_prior=True

Then evaluates both and emits a single comparison table.

Usage on Kaggle:
    python ablate_method.py --layout lolv2_real --epochs 120 --crop-size 256
"""
import argparse
import csv
import os
import subprocess
import sys


def run(cmd):
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_variant(args, tag, use_prior):
    run([
        sys.executable, "train.py",
        "--layout", args.layout,
        "--epochs", str(args.epochs),
        "--crop-size", str(args.crop_size),
        "--batch-size", str(args.batch_size),
        "--use-illum-prior", str(int(use_prior)),
        "--tag", tag,
    ])


def eval_variant(args, tag, name):
    ckpt = f"./checkpoints/best_{tag}.pth"
    if not os.path.exists(ckpt):
        ckpt = f"./checkpoints/last_{tag}.pth"
    run([
        sys.executable, "evaluation.py",
        "--splits", *args.splits,
        "--inference-steps", str(args.inference_steps),
        "--sampler", "ddim",
        "--checkpoint", ckpt,
        "--results-root", "./eval_results/method_ablation",
        "--tag", name,
    ])
    summary_path = os.path.join(f"./eval_results/method_ablation_{name}", "summary.csv")
    rows = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for row in csv.DictReader(f):
                row["variant"] = name
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="lol_v1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--splits", nargs="+",
                        default=["eval15:./eval15/low:./eval15/high"])
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--out-csv", default="./eval_results/method_ablation.csv")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; use existing best_method_* checkpoints")
    args = parser.parse_args()

    rows = []

    if not args.skip_train:
        train_variant(args, tag="method_baseline", use_prior=False)
    rows += eval_variant(args, tag="method_baseline", name="baseline")

    if not args.skip_train:
        train_variant(args, tag="method_illum", use_prior=True)
    rows += eval_variant(args, tag="method_illum", name="plus_illum_prior")

    if not rows:
        print("No rows.")
        return
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n### Method ablation\n")
    print("| Variant | Split | PSNR | SSIM | LPIPS |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['split']} | {float(r['psnr_mean']):.3f} | "
              f"{float(r['ssim_mean']):.4f} | {r.get('lpips_mean') or '-'} |")


if __name__ == "__main__":
    main()
