"""
make_comparison_grid.py — visual sanity-check helper.

Builds side-by-side comparison grids so you can actually *see* whether the 5-step
DDIM output is competitive with the 20-step output, instead of staring at folders
of single PNGs.

For each test image it produces a panel:

    [ low input | 5-step output | 20-step output | GT | (5↔20 diff heatmap) ]

with PSNR/SSIM annotated under each prediction.

Usage (from the project root, on the M4 / locally — no GPU needed, just PIL/numpy):

    # LOL eval15 (15 images, 1 grid file)
    python make_comparison_grid.py \
        --pred-s5  ./eval_results/step_ablation_full_ddim_s5/eval15 \
        --pred-s20 ./eval_results/step_ablation_full_ddim_s20/eval15 \
        --low      ./eval15/low \
        --gt       ./eval15/high \
        --per-image-csv ./eval_results/step_ablation_full_ddim_s5/per_image.csv \
        --per-image-csv-s20 ./eval_results/step_ablation_full_ddim_s20/per_image.csv \
        --split eval15 \
        --out  ./eval_results/visual_check_eval15.png

    # LOL-v2 Real (sample 8 images by default; use --max 16 for more)
    python make_comparison_grid.py \
        --pred-s5  ./eval_results/step_ablation_full_ddim_s5/lolv2_real \
        --pred-s20 ./eval_results/step_ablation_full_ddim_s20/lolv2_real \
        --low      ./LOL-v2/Real_captured/Test/Low \
        --gt       ./LOL-v2/Real_captured/Test/Normal \
        --per-image-csv     ./eval_results/step_ablation_full_ddim_s5/per_image.csv \
        --per-image-csv-s20 ./eval_results/step_ablation_full_ddim_s20/per_image.csv \
        --split lolv2_real \
        --pick worst,median,best \
        --max 9 \
        --out  ./eval_results/visual_check_lolv2_real.png

The --pick flag controls *which* images you see:
    all         — every image (good for the 15-image eval15)
    worst,median,best  — bottom-3 / middle-3 / top-3 by PSNR (good for big sets)

Naming assumption: the prediction file basename matches the GT/low basename, with
optional prefix `eval15_` or `lolv2_real_` (matches what evaluation.py writes).
"""
import argparse
import csv
import os
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def _load_per_image_psnr(csv_path, split):
    """Return {image_basename: (psnr, ssim)} for the requested split."""
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


def _resolve_pred_path(pred_dir, split, basename):
    """evaluation.py writes either `<basename>` or `<split>_<basename>`."""
    candidates = [
        os.path.join(pred_dir, basename),
        os.path.join(pred_dir, f"{split}_{basename}"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _resolve_gt_path(gt_dir, basename):
    """Per-image CSV stores the *low* image name. LOL-v2 GT is named normal*.png.
    Try the literal name, then a few common low/high naming swaps."""
    candidates = [
        basename,
        basename.replace("low", "normal"),    # LOL-v2 Real
        basename.replace("low", "high"),       # LOL-v1
        basename.replace("Low", "Normal"),
        basename.replace("Low", "High"),
    ]
    for c in candidates:
        p = os.path.join(gt_dir, c)
        if os.path.exists(p):
            return p
    return None


def _resize_to_height(img, target_h):
    w, h = img.size
    if h == target_h:
        return img
    scale = target_h / h
    return img.resize((max(1, int(round(w * scale))), target_h), Image.BICUBIC)


def _abs_diff_heatmap(img_a, img_b, vmax=40):
    """Per-pixel L1 distance scaled to a hot-style colormap (no matplotlib needed)."""
    a = np.asarray(img_a.convert("RGB"), dtype=np.float32)
    b = np.asarray(img_b.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        b = np.asarray(img_b.convert("RGB").resize(img_a.size, Image.BICUBIC),
                       dtype=np.float32)
    d = np.abs(a - b).mean(axis=2)
    d = np.clip(d / vmax, 0.0, 1.0)
    # cheap "hot" colormap: black -> red -> yellow -> white
    r = np.clip(d * 3, 0, 1)
    g = np.clip(d * 3 - 1, 0, 1)
    bch = np.clip(d * 3 - 2, 0, 1)
    rgb = (np.stack([r, g, bch], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _label(img, text, font):
    """Return a new image with a label strip on top."""
    pad = 28
    out = Image.new("RGB", (img.width, img.height + pad), color=(245, 245, 245))
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((6, 4), text, fill=(20, 20, 20), font=font)
    return out


def _pick_subset(scores, mode, max_n):
    """scores: list of (basename, psnr). Returns chosen subset preserving order."""
    if mode == "all":
        chosen = [b for b, _ in scores]
        return chosen[:max_n] if max_n else chosen
    parts = [p.strip() for p in mode.split(",")]
    sorted_asc = sorted(scores, key=lambda x: x[1])
    n = len(sorted_asc)
    k = max(1, max_n // max(1, len(parts)))
    chunks = []
    if "worst" in parts:
        chunks += sorted_asc[:k]
    if "median" in parts:
        mid = n // 2
        half = k // 2
        chunks += sorted_asc[max(0, mid - half): max(0, mid - half) + k]
    if "best" in parts:
        chunks += sorted_asc[-k:]
    seen, chosen = set(), []
    for b, _ in chunks:
        if b not in seen:
            seen.add(b)
            chosen.append(b)
    return chosen[:max_n] if max_n else chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-s5", required=True)
    ap.add_argument("--pred-s20", required=True)
    ap.add_argument("--low", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", required=True,
                    help="split name as it appears in per_image.csv (eval15, lolv2_real, lolv2_syn)")
    ap.add_argument("--per-image-csv", default=None,
                    help="per_image.csv from the s5 run (used both for picking and labels)")
    ap.add_argument("--per-image-csv-s20", default=None,
                    help="per_image.csv from the s20 run (for s20 PSNR labels)")
    ap.add_argument("--pick", default="all",
                    help="'all' or comma list of {worst,median,best}")
    ap.add_argument("--max", type=int, default=0,
                    help="max images to include (0 = no cap)")
    ap.add_argument("--row-height", type=int, default=240)
    ap.add_argument("--diff", action="store_true",
                    help="also include a 5↔20 abs-diff heatmap column")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s5_scores = _load_per_image_psnr(args.per_image_csv, args.split)
    s20_scores = _load_per_image_psnr(args.per_image_csv_s20, args.split)

    if not s5_scores:
        # fall back to whatever predictions are on disk
        files = [f for f in sorted(os.listdir(args.pred_s5))
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        scored = [(f.replace(f"{args.split}_", ""), 0.0) for f in files]
    else:
        scored = sorted(s5_scores.items(), key=lambda x: x[1][0])
        scored = [(b, p) for b, (p, _s) in scored]

    chosen = _pick_subset(scored, args.pick, args.max)
    if not chosen:
        print("No images chosen. Check --split / --per-image-csv match what's on disk.")
        sys.exit(1)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    cols = ["low input", "5-step DDIM", "20-step DDIM", "GT"]
    if args.diff:
        cols.append("|5 - 20|")

    rows = []
    for basename in chosen:
        low_path = os.path.join(args.low, basename)
        gt_path = _resolve_gt_path(args.gt, basename)
        s5_path = _resolve_pred_path(args.pred_s5, args.split, basename)
        s20_path = _resolve_pred_path(args.pred_s20, args.split, basename)
        missing = [n for n, p in [("low", low_path), ("gt", gt_path),
                                  ("s5", s5_path), ("s20", s20_path)]
                   if not p or not os.path.exists(p)]
        if missing:
            print(f"  skip {basename}: missing {missing}")
            continue

        low_img = _resize_to_height(Image.open(low_path).convert("RGB"), args.row_height)
        gt_img = _resize_to_height(Image.open(gt_path).convert("RGB"), args.row_height)
        s5_img = _resize_to_height(Image.open(s5_path).convert("RGB"), args.row_height)
        s20_img = _resize_to_height(Image.open(s20_path).convert("RGB"), args.row_height)

        s5_psnr, s5_ssim = s5_scores.get(basename, (None, None))
        s20_psnr, s20_ssim = s20_scores.get(basename, (None, None))
        s5_lab = "5-step"
        s20_lab = "20-step"
        if s5_psnr is not None:
            s5_lab += f"  PSNR {s5_psnr:.2f} / SSIM {s5_ssim:.3f}"
        if s20_psnr is not None:
            s20_lab += f"  PSNR {s20_psnr:.2f} / SSIM {s20_ssim:.3f}"

        panels = [
            _label(low_img,  f"{basename} — low input", font),
            _label(s5_img,   s5_lab, font),
            _label(s20_img,  s20_lab, font),
            _label(gt_img,   "ground truth", font),
        ]
        if args.diff:
            diff_img = _abs_diff_heatmap(s5_img, s20_img)
            panels.append(_label(diff_img, "|s5 - s20|  (red=hot)", font))

        # equalize widths to the max in this row, pad on right
        max_w = max(p.width for p in panels)
        padded = []
        for p in panels:
            if p.width < max_w:
                bg = Image.new("RGB", (max_w, p.height), (245, 245, 245))
                bg.paste(p, ((max_w - p.width) // 2, 0))
                padded.append(bg)
            else:
                padded.append(p)

        row_w = sum(p.width for p in padded) + (len(padded) - 1) * 6
        row_h = padded[0].height
        row = Image.new("RGB", (row_w, row_h), (255, 255, 255))
        x = 0
        for p in padded:
            row.paste(p, (x, 0))
            x += p.width + 6
        rows.append(row)

    if not rows:
        print("Nothing to render.")
        sys.exit(1)

    grid_w = max(r.width for r in rows)
    header_h = 30
    grid_h = header_h + sum(r.height for r in rows) + 8 * (len(rows) - 1) + 16
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    draw.text((8, 8), f"split={args.split}   pick={args.pick}   "
                       f"5-step vs 20-step DDIM   ←better PSNR ===> worse",
              fill=(20, 20, 20), font=font)
    y = header_h
    for r in rows:
        grid.paste(r, (0, y))
        y += r.height + 8

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    grid.save(args.out)
    print(f"Wrote {args.out}  ({grid.width}x{grid.height}, {len(rows)} rows)")


if __name__ == "__main__":
    main()
