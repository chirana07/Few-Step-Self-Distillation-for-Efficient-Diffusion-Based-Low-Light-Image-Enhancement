"""
make_failure_cases_figure.py — paper Figure 5: failure cases (Limitations section).

Picks the worst-N images from a per_image.csv (sorted by PSNR ascending) and
renders a grid: low input | predicted | GT, with PSNR/SSIM annotations.

Addresses Reviewer H2EM's "more discussion on failure cases" concern.

Usage (locally on macOS):

    python make_failure_cases_figure.py \\
        --pred-dir   ./phase3_day1_outputs/headline_s5/lolv2_real \\
        --low-dir    ./LOL-v2/Real_captured/Test/Low \\
        --gt-dir     ./LOL-v2/Real_captured/Test/Normal \\
        --per-image-csv ./phase3_day1_outputs/headline_s5/per_image.csv \\
        --split lolv2_real \\
        --top-n 3 \\
        --out  ./figures/figure5_failure_cases.pdf
"""
import argparse
import csv
import os
import sys

from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_per_image(csv_path, split):
    """Return [(basename, psnr, ssim), ...] for the requested split."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            try:
                rows.append((row["image"], float(row["psnr"]), float(row["ssim"])))
            except (ValueError, KeyError):
                continue
    return rows


def resolve_pred_path(pred_dir, split, basename):
    for cand in (basename, f"{split}_{basename}"):
        p = os.path.join(pred_dir, cand)
        if os.path.exists(p):
            return p
    return None


def resolve_gt_path(gt_dir, basename):
    for cand in (basename,
                  basename.replace("low", "normal"),
                  basename.replace("low", "high"),
                  basename.replace("Low", "Normal"),
                  basename.replace("Low", "High")):
        p = os.path.join(gt_dir, cand)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--low-dir",  required=True)
    ap.add_argument("--gt-dir",   required=True)
    ap.add_argument("--per-image-csv", required=True)
    ap.add_argument("--split", default="lolv2_real")
    ap.add_argument("--top-n", type=int, default=3, help="number of worst cases to show")
    ap.add_argument("--images", nargs="+", default=None,
                    help="Manual list of images to show (e.g. low00754.png low00756.png)")
    ap.add_argument("--out", default="./figures/figure5_failure_cases.pdf")
    ap.add_argument("--out-png", default="./figures/figure5_failure_cases.png")
    args = ap.parse_args()

    rows = load_per_image(args.per_image_csv, args.split)
    if not rows:
        sys.exit(f"No rows found for split '{args.split}' in {args.per_image_csv}")

    if args.images:
        # Filter for the requested images and maintain order
        row_dict = {r[0]: r for r in rows}
        failures = [row_dict[img] for img in args.images if img in row_dict]
        if not failures:
            sys.exit(f"None of the requested images {args.images} found in CSV.")
    else:
        # Sort ascending by PSNR — worst cases first
        rows.sort(key=lambda x: x[1])
        failures = rows[: args.top_n]

    # Build the figure: 3 columns × N rows
    num_rows = len(failures)
    fig, axes = plt.subplots(num_rows, 3,
                             figsize=(11, 3.4 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Low-light input", "Ours (5-step DDIM)", "Ground truth"]

    for r, (basename, psnr, ssim) in enumerate(failures):
        low_path  = os.path.join(args.low_dir, basename)
        gt_path   = resolve_gt_path(args.gt_dir, basename)
        pred_path = resolve_pred_path(args.pred_dir, args.split, basename)
        if not all([os.path.exists(low_path), gt_path, pred_path]):
            print(f"  skipping {basename}: missing one of low/gt/pred")
            continue

        for c, p in enumerate([low_path, pred_path, gt_path]):
            img = Image.open(p).convert("RGB")
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11, fontweight="bold")

        # Row label on the left side
        axes[r, 0].set_ylabel(
            f"{basename}\nPSNR {psnr:.2f}  SSIM {ssim:.3f}",
            rotation=0, ha="right", va="center",
            fontsize=9, fontweight="bold", labelpad=80,
        )

    fig.suptitle(
        f"Failure cases (worst {args.top_n} on {args.split} by PSNR) — "
        "see Section 5 (Limitations)",
        fontsize=11, y=0.98, fontweight="bold",
    )
    plt.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.02,
                        hspace=0.05, wspace=0.04)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {args.out}")
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
        print(f"Wrote {args.out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
