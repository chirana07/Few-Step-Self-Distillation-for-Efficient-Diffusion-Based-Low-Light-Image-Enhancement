"""
ablate_steps.py — sweep the sampling-step ablation across one or more test splits,
for both the DDIM and the legacy posterior samplers, and write a combined CSV
plus a quick matplotlib figure if matplotlib is available.

Run on Kaggle (after git pulling and copying LOL / LOL-v2 into ./):
    python ablate_steps.py \
        --splits eval15:./eval15/low:./eval15/high \
                 lolv2_real:./LOLv2/Real/Test/Low:./LOLv2/Real/Test/Normal \
        --steps 5 10 20 50 100 \
        --samplers ddim dpm_posterior
"""
import argparse
import csv
import os
import subprocess
import sys


def run_eval(splits, checkpoint, steps, sampler, results_root, tag):
    """Shell out to evaluation.py to keep the two scripts decoupled."""
    cmd = [
        sys.executable, "evaluation.py",
        "--splits", *splits,
        "--inference-steps", str(steps),
        "--sampler", sampler,
        "--results-root", results_root,
        "--tag", tag,
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_summary(results_root, tag):
    summary_path = os.path.join(f"{results_root}_{tag}", "summary.csv")
    if not os.path.exists(summary_path):
        return []
    with open(summary_path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["eval15:./eval15/low:./eval15/high"])
    parser.add_argument("--steps", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    parser.add_argument("--samplers", nargs="+", default=["ddim", "dpm_posterior"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-root", default="./eval_results/step_ablation")
    parser.add_argument("--out-csv", default="./eval_results/step_ablation/combined.csv")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()

    all_rows = []
    for sampler in args.samplers:
        for steps in args.steps:
            tag = f"{sampler}_s{steps}"
            # build command
            cmd = [
                sys.executable, "evaluation.py",
                "--splits", *args.splits,
                "--inference-steps", str(steps),
                "--sampler", sampler,
                "--results-root", args.results_root,
                "--tag", tag,
            ]
            if args.checkpoint:
                cmd += ["--checkpoint", args.checkpoint]
            if args.no_lpips:
                cmd += ["--no-lpips"]
            
            print("$ " + " ".join(cmd))
            subprocess.run(cmd, check=True)
            
            for row in collect_summary(args.results_root, tag):
                row["sampler_override"] = sampler
                row["steps_override"] = steps
                all_rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        if not all_rows:
            print("No results collected.")
            return
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCombined ablation CSV: {args.out_csv}\n")

    # Markdown summary
    print("### Step-ablation summary\n")
    print("| Split | Sampler | Steps | PSNR | SSIM | LPIPS | Latency/img (s) |")
    print("|---|---|---|---|---|---|---|")
    for r in all_rows:
        print(
            f"| {r['split']} | {r['sampler']} | {r['inference_steps']} | "
            f"{float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | "
            f"{r.get('lpips_mean') or '-'} | {float(r['runtime_mean']):.3f} |"
        )

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot")
            return
        # Group rows by (split, sampler) and plot PSNR vs steps
        groups = {}
        for r in all_rows:
            key = (r["split"], r["sampler"])
            groups.setdefault(key, []).append(r)
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, rows in groups.items():
            rows = sorted(rows, key=lambda x: int(x["inference_steps"]))
            xs = [int(r["inference_steps"]) for r in rows]
            ys = [float(r["psnr_mean"]) for r in rows]
            ax.plot(xs, ys, marker="o", label=f"{key[0]} / {key[1]}")
        ax.set_xlabel("Sampling steps")
        ax.set_ylabel("PSNR (dB)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Quality vs. sampling steps")
        plot_path = os.path.join(os.path.dirname(args.out_csv), "psnr_vs_steps.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
