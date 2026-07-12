"""
ablate_losses.py — train short variants each removing one loss term, then evaluate.

For a workshop-tier paper you don't need to run each variant to 300 epochs.
60-80 epochs is usually enough to see which term matters. Adjust --epochs
to what your Kaggle time budget allows.

Usage on Kaggle:
    python ablate_losses.py --layout lolv2_real --epochs 80 --crop-size 256
"""
import argparse
import csv
import os
import subprocess
import sys


# Variant name -> (w_char, w_ssim, w_perc, w_color, w_grad, w_tv)
VARIANTS = {
    "full":       (1.0, 0.5, 0.1, 0.05, 0.2, 0.02),
    "char_only":  (1.0, 0.0, 0.0, 0.00, 0.0, 0.00),
    "no_ssim":    (1.0, 0.0, 0.1, 0.05, 0.2, 0.02),
    "no_perc":    (1.0, 0.5, 0.0, 0.05, 0.2, 0.02),
    "no_color":   (1.0, 0.5, 0.1, 0.00, 0.2, 0.02),
    "no_grad":    (1.0, 0.5, 0.1, 0.05, 0.0, 0.02),
    "no_tv":      (1.0, 0.5, 0.1, 0.05, 0.2, 0.00),
}


def run(cmd):
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="lol_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-illum-prior", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--splits", nargs="+",
                        default=["eval15:./eval15/low:./eval15/high"],
                        help="Evaluation splits (passed through to evaluation.py)")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--out-csv", default="./eval_results/loss_ablation.csv")
    args = parser.parse_args()

    results = []
    for v in args.variants:
        if v not in VARIANTS:
            print(f"skipping unknown variant {v}")
            continue
        w_char, w_ssim, w_perc, w_color, w_grad, w_tv = VARIANTS[v]
        tag = f"loss_{v}"

        # Train
        train_cmd = [
            sys.executable, "train.py",
            "--layout", args.layout,
            "--epochs", str(args.epochs),
            "--crop-size", str(args.crop_size),
            "--batch-size", str(args.batch_size),
            "--use-illum-prior", str(args.use_illum_prior),
            "--tag", tag,
            "--w-char", str(w_char),
            "--w-ssim", str(w_ssim),
            "--w-perc", str(w_perc),
            "--w-color", str(w_color),
            "--w-grad", str(w_grad),
            "--w-tv", str(w_tv),
        ]
        run(train_cmd)

        # Eval — use best checkpoint for this variant
        best_ckpt = f"./checkpoints/best_{tag}.pth"
        if not os.path.exists(best_ckpt):
            best_ckpt = f"./checkpoints/last_{tag}.pth"
        eval_cmd = [
            sys.executable, "evaluation.py",
            "--splits", *args.splits,
            "--inference-steps", str(args.inference_steps),
            "--sampler", "ddim",
            "--checkpoint", best_ckpt,
            "--results-root", "./eval_results/loss_ablation",
            "--tag", v,
        ]
        run(eval_cmd)

        summary_path = os.path.join(f"./eval_results/loss_ablation_{v}", "summary.csv")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                for row in csv.DictReader(f):
                    row["variant"] = v
                    row["weights"] = VARIANTS[v]
                    results.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if not results:
        print("No results collected.")
        return
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n### Loss-component ablation\n")
    print("| Variant | Split | PSNR | SSIM | LPIPS |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['variant']} | {r['split']} | {float(r['psnr_mean']):.3f} | "
              f"{float(r['ssim_mean']):.4f} | {r.get('lpips_mean') or '-'} |")


if __name__ == "__main__":
    main()
