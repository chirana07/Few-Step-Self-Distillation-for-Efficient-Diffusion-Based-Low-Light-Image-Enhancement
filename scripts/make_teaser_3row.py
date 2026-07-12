"""
make_teaser_3row.py — Figure 1 of the paper (3-row version).

Renders a 3-row × 3-column teaser grid:
    [ low input | ours (FSD+ARR, 5 steps) | GT ]
for three selected LOL-v2 Real images, with PSNR/SSIM annotated.

Usage:
    python make_teaser_3row.py \
        --pred-dir ./eval_results_distill/headline_student_s5/lolv2_real \
        --low-dir  ./LOL-v2/Real_captured/Test/Low \
        --gt-dir   ./LOL-v2/Real_captured/Test/Normal \
        --per-image-csv ./eval_results_distill/headline_student_s5/per_image.csv \
        --images low00742.png low00690.png low00738.png \
        --out ./PAPER/figure1_teaser_3row.pdf
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
import matplotlib.gridspec as gridspec


def load_metrics(csv_path, split="lolv2_real"):
    """Return {basename: (psnr, ssim)} for the given split."""
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


def resolve_pred(pred_dir, basename):
    """Try multiple naming conventions for predicted images."""
    for candidate in (basename, f"lolv2_real_{basename}"):
        p = os.path.join(pred_dir, candidate)
        if os.path.exists(p):
            return p
    return None


def resolve_gt(gt_dir, basename):
    """Try multiple naming conventions for GT images."""
    for candidate in (basename,
                      basename.replace("low", "normal"),
                      basename.replace("low", "high"),
                      basename.replace("Low", "Normal")):
        p = os.path.join(gt_dir, candidate)
        if os.path.exists(p):
            return p
    return None


def render_3row(images_data, out_pdf, out_png=None, dpi=300):
    """
    images_data: list of 3 dicts, each with keys:
        low_path, pred_path, gt_path, basename, psnr, ssim
    """
    fig = plt.figure(figsize=(10, 10.5))
    gs = gridspec.GridSpec(3, 3, wspace=0.04, hspace=0.12,
                           left=0.02, right=0.98, top=0.95, bottom=0.02)

    col_titles = ["Low-light Input", "Ours (FSD + ARR, 5 steps)", "Ground Truth"]

    for row_idx, data in enumerate(images_data):
        paths = [data["low_path"], data["pred_path"], data["gt_path"]]

        for col_idx, (path, title) in enumerate(zip(paths, col_titles)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(0.5)

            # Column titles on the first row only
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

            # Annotate PSNR/SSIM on prediction column
            if col_idx == 1 and data.get("psnr") is not None:
                label = "PSNR {:.2f} dB / SSIM {:.4f}".format(
                    data["psnr"], data["ssim"])
                ax.text(0.5, -0.06, label,
                        transform=ax.transAxes, fontsize=9,
                        ha="center", va="top",
                        fontweight="medium",
                        color="#222222",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="#cccccc",
                                  alpha=0.85))

    fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
    print(f"Wrote {out_pdf}")
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out_png}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="Directory with predicted images")
    ap.add_argument("--low-dir", required=True,
                    help="Directory with low-light inputs")
    ap.add_argument("--gt-dir", required=True,
                    help="Directory with ground truth images")
    ap.add_argument("--per-image-csv", required=True,
                    help="CSV with per-image PSNR/SSIM")
    ap.add_argument("--split", default="lolv2_real")
    ap.add_argument("--images", nargs=3, required=True,
                    help="Three image basenames, e.g. low00742.png low00690.png low00738.png")
    ap.add_argument("--out", default="./PAPER/figure1_teaser_3row.pdf")
    ap.add_argument("--out-png", default=None,
                    help="Also save PNG (auto-derived from --out if not specified)")
    args = ap.parse_args()

    if args.out_png is None:
        args.out_png = args.out.replace(".pdf", ".png")

    metrics = load_metrics(args.per_image_csv, args.split)

    images_data = []
    for basename in args.images:
        low_path = os.path.join(args.low_dir, basename)
        pred_path = resolve_pred(args.pred_dir, basename)
        gt_path = resolve_gt(args.gt_dir, basename)

        missing = []
        if not os.path.exists(low_path):
            missing.append(("low", low_path))
        if not pred_path or not os.path.exists(pred_path):
            missing.append(("pred", pred_path))
        if not gt_path or not os.path.exists(gt_path):
            missing.append(("gt", gt_path))

        if missing:
            print(f"ERROR: Missing files for {basename}: {missing}")
            sys.exit(1)

        m = metrics.get(basename, (None, None))
        images_data.append({
            "basename": basename,
            "low_path": low_path,
            "pred_path": pred_path,
            "gt_path": gt_path,
            "psnr": m[0],
            "ssim": m[1],
        })
        print(f"  {basename}: PSNR={m[0]:.4f}, SSIM={m[1]:.4f}" if m[0] else f"  {basename}: no metrics")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render_3row(images_data, args.out, args.out_png)


if __name__ == "__main__":
    main()
