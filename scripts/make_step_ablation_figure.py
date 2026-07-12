"""
make_step_ablation_figure.py — Figure 3 of the paper.

Reads the per-tag summary.csv files written by evaluation.py and renders a clean
PSNR-vs-steps plot. Two panels (one per split: eval15 and LOL-v2 Real), two lines
each (DDIM vs the legacy DPM-posterior sampler), with markers and value labels.

This is the *core* figure of the paper — the visual that justifies the whole
"5-step is enough" claim. Don't skip it, don't make it ugly.

Usage (locally on M4 or on Kaggle, no GPU needed):

    python make_step_ablation_figure.py \
        --eval-root ./eval_results \
        --out ./eval_results/figure3_step_ablation.pdf \
        --out-png ./eval_results/figure3_step_ablation.png

It auto-discovers folders named:
    step_ablation_<SPLIT>_<SAMPLER>_s<STEPS>/summary.csv
    step_ablation_full_<SAMPLER>_s<STEPS>/summary.csv   (these contain ALL splits)

If a (split, sampler, steps) combination isn't on disk, it's silently skipped.
"""
import argparse
import csv
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_TITLES = {
    "eval15": "LOL eval15 (n=15)",
    "lolv2_real": "LOL-v2 Real (n=100)",
    "lolv2_syn": "LOL-v2 Synthetic (n=100)",
}

SAMPLER_LABELS = {
    "ddim": "DDIM (ours)",
    "dpm_posterior": "DPM posterior (legacy)",
}

SAMPLER_STYLE = {
    "ddim": dict(color="#0b5fff", marker="o", linestyle="-", linewidth=2.0,
                 markersize=7, zorder=3),
    "dpm_posterior": dict(color="#d6336c", marker="s", linestyle="--",
                          linewidth=1.6, markersize=6, zorder=2),
}


def discover_runs(eval_root):
    """Walk eval_root and return list of (split, sampler, steps, summary_path)."""
    rows = []
    if not os.path.isdir(eval_root):
        return rows

    pat_per_split = re.compile(
        r"^step_ablation_([a-z0-9_]+)_(ddim|dpm_posterior)_s(\d+)$"
    )
    pat_full = re.compile(r"^step_ablation_full_(ddim|dpm_posterior)_s(\d+)$")

    for d in sorted(os.listdir(eval_root)):
        full = os.path.join(eval_root, d)
        if not os.path.isdir(full):
            continue
        summary = os.path.join(full, "summary.csv")
        if not os.path.exists(summary):
            continue

        m = pat_full.match(d)
        if m:
            sampler, steps = m.group(1), int(m.group(2))
            with open(summary) as f:
                for row in csv.DictReader(f):
                    rows.append((row["split"], sampler, steps, row))
            continue

        m = pat_per_split.match(d)
        if m:
            split_in_name, sampler, steps = m.group(1), m.group(2), int(m.group(3))
            with open(summary) as f:
                for row in csv.DictReader(f):
                    rows.append((row["split"], sampler, steps, row))
    return rows


def aggregate(rows, metric):
    """rows: list of (split, sampler, steps, csv_row).
    Returns nested dict: {split: {sampler: {steps: value}}}
    Picks the latest value if duplicates appear (later listdir entry wins, fine in practice).
    """
    out = {}
    for split, sampler, steps, row in rows:
        v = row.get(metric)
        if v in (None, ""):
            continue
        try:
            v = float(v)
        except ValueError:
            continue
        out.setdefault(split, {}).setdefault(sampler, {})[steps] = v
    return out


def plot_psnr_vs_steps(agg, splits, out_path, out_png=None, annotate=True):
    fig, axes = plt.subplots(
        1, len(splits), figsize=(4.6 * len(splits), 3.6), sharey=False
    )
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        if split not in agg:
            ax.set_title(f"{SPLIT_TITLES.get(split, split)}\n(no data)",
                         fontsize=11)
            ax.axis("off")
            continue

        for sampler, by_steps in agg[split].items():
            xs = sorted(by_steps.keys())
            ys = [by_steps[x] for x in xs]
            style = SAMPLER_STYLE.get(sampler, dict(marker="x"))
            ax.plot(xs, ys, label=SAMPLER_LABELS.get(sampler, sampler), **style)

            if annotate:
                for x, y in zip(xs, ys):
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                xytext=(6, 4), fontsize=8,
                                color=style.get("color", "black"))

        ax.set_title(SPLIT_TITLES.get(split, split), fontsize=11)
        ax.set_xlabel("Sampling steps")
        ax.set_ylabel("PSNR (dB)")
        ax.set_xscale("log")
        # show clean tick labels at the actual step counts
        all_steps = sorted({s for sm in agg[split].values() for s in sm.keys()})
        if all_steps:
            ax.set_xticks(all_steps)
            ax.set_xticklabels([str(s) for s in all_steps])
        ax.grid(True, which="both", alpha=0.25, linestyle=":")
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.suptitle("Quality vs. sampling steps", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_png}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="./eval_results")
    ap.add_argument("--out", default="./eval_results/figure3_step_ablation.pdf")
    ap.add_argument("--out-png", default="./eval_results/figure3_step_ablation.png")
    ap.add_argument("--splits", nargs="+",
                    default=["eval15", "lolv2_real"],
                    help="Which splits to plot (must exist in eval_root)")
    ap.add_argument("--metric", default="psnr_mean",
                    help="psnr_mean | ssim_mean | lpips_mean")
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    rows = discover_runs(args.eval_root)
    if not rows:
        print(f"No step-ablation runs found under {args.eval_root}.")
        print("Expected folders matching: step_ablation_full_<sampler>_s<steps>/summary.csv")
        sys.exit(1)

    agg = aggregate(rows, args.metric)
    found = sorted({(s, sm, st) for (s, sm, st, _) in rows})
    print(f"Discovered {len(found)} (split, sampler, steps) combos:")
    for s, sm, st in found:
        v = agg.get(s, {}).get(sm, {}).get(st)
        print(f"   {s:<12s}  {sm:<14s}  steps={st:<3d}  {args.metric}={v}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_psnr_vs_steps(
        agg, args.splits, args.out,
        out_png=args.out_png, annotate=not args.no_annotate,
    )


if __name__ == "__main__":
    main()
