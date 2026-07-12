"""
make_teaser_figure.py — Figure 1 of the paper.

Picks one carefully chosen LOL-v2 Real example and renders the 4-panel teaser:
    [ low input | ours-5step | ours-20step | GT ]
with PSNR/SSIM annotated under each prediction.

The rule for picking the example: NOT the highest-PSNR row. Pick a row where
(a) the low input is dramatically dark (so the lift is visible to a reader skimming),
and (b) 5-step PSNR is at or above the dataset mean (so the example is honest, not
cherry-picked from the easy tail).

Usage:
    python make_teaser_figure.py \
        --pred-s5  ./eval_results/step_ablation_full_ddim_s5/lolv2_real \
        --pred-s20 ./eval_results/step_ablation_full_ddim_s20/lolv2_real \
        --low      ./LOL-v2/Real_captured/Test/Low \
        --gt       ./LOL-v2/Real_captured/Test/Normal \
        --per-image-csv     ./eval_results/step_ablation_full_ddim_s5/per_image.csv \
        --per-image-csv-s20 ./eval_results/step_ablation_full_ddim_s20/per_image.csv \
        --split lolv2_real \
        --out  ./eval_results/figure1_teaser.pdf

Or specify the image manually with --image low00712.png.
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_per_image(csv_path, split):
    out = {}
    if not csv_path or not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            try:
                out[row["image"]] = (float(row["psnr"]), float(row["ssim"]))
            except (ValueError, KeyError):
                continue
    return out


def _resolve_pred(pred_dir, split, basename):
    for c in (basename, f"{split}_{basename}"):
        p = os.path.join(pred_dir, c)
        if os.path.exists(p):
            return p
    return None


def _resolve_gt(gt_dir, basename):
    for c in (basename,
              basename.replace("low", "normal"),
              basename.replace("low", "high"),
              basename.replace("Low", "Normal"),
              basename.replace("Low", "High")):
        p = os.path.join(gt_dir, c)
        if os.path.exists(p):
            return p
    return None


def _mean_brightness(path):
    """Lower = darker low-light input. Used for picking dramatic examples."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return float(arr.mean())


def pick_image(s5_scores, low_dir, dataset_psnr_mean, prefer_dark=True):
    """Return basename of the chosen teaser image."""
    candidates = []
    for basename, (psnr, ssim) in s5_scores.items():
        if psnr < dataset_psnr_mean - 0.5:  # below average -> reject
            continue
        if psnr > dataset_psnr_mean + 4.0:  # cherry-pick territory -> reject
            continue
        low_path = os.path.join(low_dir, basename)
        if not os.path.exists(low_path):
            continue
        bright = _mean_brightness(low_path)
        candidates.append((basename, psnr, bright))

    if not candidates:
        # fall back: just pick a median-PSNR row
        sorted_by_psnr = sorted(s5_scores.items(), key=lambda x: x[1][0])
        return sorted_by_psnr[len(sorted_by_psnr) // 2][0]

    # rank by darkness ascending if prefer_dark, else random middle
    candidates.sort(key=lambda x: x[2] if prefer_dark else -x[2])
    return candidates[0][0]


def render(low_path, s5_path, s20_path, gt_path, s5_metrics, s20_metrics,
           gt_metrics, basename, out_pdf, out_png=None, dpi=200):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    titles = [
        "Low-light input",
        f"Ours, 5 DDIM steps" + (
            f"\nPSNR {s5_metrics[0]:.2f} / SSIM {s5_metrics[1]:.3f}" if s5_metrics else ""),
        f"Ours, 20 DDIM steps" + (
            f"\nPSNR {s20_metrics[0]:.2f} / SSIM {s20_metrics[1]:.3f}" if s20_metrics else ""),
        "Ground truth",
    ]
    paths = [low_path, s5_path, s20_path, gt_path]
    for ax, title, p in zip(axes, titles, paths):
        img = Image.open(p).convert("RGB")
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"LOL-v2 Real, image {basename} — 5-step output is visually indistinguishable from 20-step",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out_png}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-s5", required=True)
    ap.add_argument("--pred-s20", required=True)
    ap.add_argument("--low", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", default="lolv2_real")
    ap.add_argument("--per-image-csv", default=None)
    ap.add_argument("--per-image-csv-s20", default=None)
    ap.add_argument("--image", default=None,
                    help="Force a specific basename, e.g. low00712.png")
    ap.add_argument("--out", default="./eval_results/figure1_teaser.pdf")
    ap.add_argument("--out-png", default="./eval_results/figure1_teaser.png")
    args = ap.parse_args()

    s5 = _load_per_image(args.per_image_csv, args.split)
    s20 = _load_per_image(args.per_image_csv_s20, args.split)
    if not s5:
        print("No s5 per-image CSV available; pass --image manually if you want a specific one.")
        if not args.image:
            sys.exit(1)

    if args.image:
        basename = args.image
    else:
        mean_psnr = float(np.mean([p for p, _ in s5.values()]))
        basename = pick_image(s5, args.low, mean_psnr, prefer_dark=True)
        print(f"Auto-picked: {basename}  (dataset mean PSNR={mean_psnr:.2f})")

    low_path = os.path.join(args.low, basename)
    gt_path = _resolve_gt(args.gt, basename)
    s5_path = _resolve_pred(args.pred_s5, args.split, basename)
    s20_path = _resolve_pred(args.pred_s20, args.split, basename)
    missing = [(n, p) for n, p in [("low", low_path), ("gt", gt_path),
                                   ("s5", s5_path), ("s20", s20_path)]
               if not p or not os.path.exists(p)]
    if missing:
        print(f"Missing: {missing}")
        sys.exit(1)

    s5_metrics = s5.get(basename)
    s20_metrics = s20.get(basename)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render(low_path, s5_path, s20_path, gt_path,
           s5_metrics, s20_metrics, None,
           basename, args.out, args.out_png)


if __name__ == "__main__":
    main()
