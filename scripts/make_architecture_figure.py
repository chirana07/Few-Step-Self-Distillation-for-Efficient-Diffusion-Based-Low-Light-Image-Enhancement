"""
make_architecture_figure.py — paper Figure 2: model architecture.

Draws the residual-conditioned U-Net with Retinex illumination prior and SFT
conditioning, in a clean vector-graphics style suitable for paper inclusion.

Runs locally on macOS / Linux with just matplotlib (no GPU, no torch, no model
loading). Outputs PDF (vector) + PNG (preview).

Usage:
    python make_architecture_figure.py
    python make_architecture_figure.py --out ./figures/figure2_architecture.pdf

Layout:
    Row 1 (top):    Low-light y  -->  Illumination Prior L(y)  -->  Condition Encoder
                                                                          |
                                                                       SFT γ,β
                                                                          v
    Row 2 (middle): x_t + y + L(y)  -->  Head conv  -->  U-Net w/ SFT  -->  GN+Swish
                                                                          |
    Row 3 (bottom):  delta head (tanh) + gate head (sigmoid)  -->  g·δ = r-hat
                                                                          |
                                                                          v
                                                                  y + r-hat = output

Designed to render at ~7.5" wide, paper-quality.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- visual style constants ----
COLORS = {
    "input":     "#dbeafe",  # blue-100
    "input_b":   "#1d4ed8",
    "encoder":   "#ffedd5",  # orange-100
    "encoder_b": "#c2410c",
    "unet":      "#dcfce7",  # green-100
    "unet_b":    "#15803d",
    "head":      "#f3e8ff",  # purple-100
    "head_b":    "#6b21a8",
    "output":    "#fee2e2",  # red-100
    "output_b":  "#b91c1c",
    "concat":    "#f3f4f6",  # gray-100
    "concat_b":  "#4b5563",
    "arrow":     "#374151",
    "label":     "#111827",
    "subtle":    "#6b7280",
}

FONT_FAMILY = "DejaVu Sans"


def box(ax, x, y, w, h, fill, edge, label_text, fontsize=10,
        sub=None, sub_fontsize=8.0, rounded=0.05, label_weight="bold",
        label_color=None):
    """Rounded rectangle with a primary label and optional italic sub-label."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={rounded}",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(patch)
    if sub:
        # primary slightly above center, sub below
        ax.text(x + w / 2, y + h * 0.65, label_text, ha="center", va="center",
                fontsize=fontsize, color=label_color or COLORS["label"],
                fontweight=label_weight, family=FONT_FAMILY, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=sub_fontsize, color=COLORS["subtle"],
                style="italic", family=FONT_FAMILY, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label_text, ha="center", va="center",
                fontsize=fontsize, color=label_color or COLORS["label"],
                fontweight=label_weight, family=FONT_FAMILY, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.3, style="-|>", curve=0.0,
          arrowsize=12):
    color = color or COLORS["arrow"]
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=arrowsize,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={curve}",
        zorder=4,
    )
    ax.add_patch(a)


def caption(ax, x, y, text, fontsize=8.0, color=None, ha="center", va="center",
            italic=True):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            color=color or COLORS["subtle"], family=FONT_FAMILY,
            style="italic" if italic else "normal", zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./figures/figure2_architecture.pdf")
    ap.add_argument("--out-png", default="./figures/figure2_architecture.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ===== canvas =====
    fig_w, fig_h = 11, 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Box dimensions (consistent)
    BW = 2.0   # standard box width
    BH = 1.1   # standard box height

    # ----- ROW 1: input pathway and condition encoder -----
    R1_Y = 8.2

    # Low-light input
    box(ax, 0.5, R1_Y, BW, BH, COLORS["input"], COLORS["input_b"],
        "Low-light input", sub="$y \\in \\mathbb{R}^{H \\times W \\times 3}$",
        fontsize=10.5)

    # Illumination prior
    box(ax, 3.6, R1_Y, BW + 0.5, BH, COLORS["input"], COLORS["input_b"],
        "Illumination Prior",
        sub="$L(y) = \\mathrm{blur}(\\max_c y)$",
        fontsize=10.5)
    arrow(ax, 2.5, R1_Y + BH/2, 3.6, R1_Y + BH/2)

    # Concat
    box(ax, 7.0, R1_Y + 0.18, 1.6, 0.74, COLORS["concat"], COLORS["concat_b"],
        "$[y \\| L(y)]$", fontsize=10.5, rounded=0.04, label_weight="normal")
    arrow(ax, 6.1, R1_Y + BH/2, 7.0, R1_Y + 0.55)
    # skip arrow from y to concat
    arrow(ax, 1.5, R1_Y + 0.15, 7.0, R1_Y + 0.35,
          lw=0.9, curve=-0.18, color=COLORS["subtle"])

    # Condition encoder
    box(ax, 9.4, R1_Y, BW + 0.6, BH, COLORS["encoder"], COLORS["encoder_b"],
        "Condition Encoder",
        sub="multi-scale features for SFT",
        fontsize=10.5)
    arrow(ax, 8.6, R1_Y + 0.55, 9.4, R1_Y + BH/2)

    # SFT modulation arrow down to U-Net
    arrow(ax, 10.7, R1_Y, 10.7, 6.1, lw=1.7, color=COLORS["encoder_b"], arrowsize=14)
    caption(ax, 11.7, 7.3, "SFT  ($\\gamma, \\beta$)", color=COLORS["encoder_b"],
            fontsize=9.5, italic=False)

    # ----- ROW 2: noisy residual -> head conv -> U-Net -----
    R2_Y = 5.0

    # Noisy residual input
    box(ax, 0.5, R2_Y, BW, BH, COLORS["input"], COLORS["input_b"],
        "Noisy residual",
        sub="$x_t$,  $t \\sim \\mathcal{T}$",
        fontsize=10.5)

    # Concat
    box(ax, 2.85, R2_Y + 0.18, 1.7, 0.74, COLORS["concat"], COLORS["concat_b"],
        "$[x_t \\| y \\| L(y)]$", fontsize=10.0,
        rounded=0.04, label_weight="normal")
    arrow(ax, 2.5, R2_Y + 0.55, 2.85, R2_Y + 0.55)

    # Head conv
    box(ax, 4.95, R2_Y, BW, BH, COLORS["unet"], COLORS["unet_b"],
        "Head conv 3×3",
        sub="(7 → C) channels",
        fontsize=10.5)
    arrow(ax, 4.55, R2_Y + 0.55, 4.95, R2_Y + 0.55)

    # U-Net body (wider box)
    box(ax, 7.4, R2_Y - 0.05, 5.6, 1.25, COLORS["unet"], COLORS["unet_b"],
        "U-Net denoiser  (SFT-conditioned)",
        sub="ResBlocks + Self-Attention bottleneck;  time embedding $t$",
        fontsize=11)
    arrow(ax, 6.95, R2_Y + 0.55, 7.4, R2_Y + 0.55)

    # GN+Swish (post-U-Net)
    box(ax, 13.4, R2_Y, BW + 0.2, BH, COLORS["unet"], COLORS["unet_b"],
        "GroupNorm\n+ Swish",
        fontsize=10, rounded=0.04)
    arrow(ax, 13.0, R2_Y + 0.55, 13.4, R2_Y + 0.55)

    # SFT lateral feed already pointing down from row 1 to (10.7, 6.1) — the U-Net top edge
    # SFT info also conditions encoder branch; use a small caption arrow into U-Net
    caption(ax, 10.0, 4.3,
            "low-light input  $y$  also conditions every block via SFT",
            fontsize=8.5, color=COLORS["subtle"])

    # Vertical arrow from U-Net output down toward row 3 split
    arrow(ax, 14.5, R2_Y, 14.5, 3.0, lw=1.5, arrowsize=14)

    # ----- ROW 3: heads + residual fusion + output -----
    R3_Y = 1.7

    # Delta head
    box(ax, 9.5, R3_Y, 1.6, 0.95, COLORS["head"], COLORS["head_b"],
        "$\\delta$-head",
        sub="3×3 conv → tanh",
        fontsize=10.5, rounded=0.05)

    # Gate head
    box(ax, 11.4, R3_Y, 1.6, 0.95, COLORS["head"], COLORS["head_b"],
        "$g$-head",
        sub="3×3 conv → sigmoid",
        fontsize=10.5, rounded=0.05)

    # Feed from GN+Swish (above row 3) down into both heads
    arrow(ax, 14.3, 3.0, 10.3, R3_Y + 0.95, lw=1.2, curve=0.18)
    arrow(ax, 14.3, 3.0, 12.2, R3_Y + 0.95, lw=1.2, curve=0.10)

    # Fusion box
    box(ax, 13.4, R3_Y, 2.2, 0.95, "white", COLORS["head_b"],
        "$\\hat r = g \\odot \\delta$",
        sub="residual prediction",
        fontsize=10.5, rounded=0.05, label_color=COLORS["head_b"])
    arrow(ax, 11.1, R3_Y + 0.5, 13.4, R3_Y + 0.5, curve=0.0)
    arrow(ax, 13.0, R3_Y + 0.5, 13.4, R3_Y + 0.5)

    # Final output
    box(ax, 13.4, 0.2, 2.2, 0.95, COLORS["output"], COLORS["output_b"],
        "$\\tilde x = y + \\hat r$",
        sub="(clamped to [-1, 1])",
        fontsize=10.5, rounded=0.05)
    arrow(ax, 14.5, R3_Y, 14.5, 1.15, lw=1.5)

    # ----- title and legend -----
    fig.suptitle(
        "Architecture: Residual-Space Diffusion U-Net with Retinex Illumination Prior and SFT Conditioning",
        fontsize=12, y=0.97, fontweight="bold", family=FONT_FAMILY,
    )

    legend_lines = [
        Line2D([0], [0], color=COLORS["input_b"],   lw=8, label="Inputs / illumination"),
        Line2D([0], [0], color=COLORS["encoder_b"], lw=8, label="Condition encoder (SFT)"),
        Line2D([0], [0], color=COLORS["unet_b"],    lw=8, label="U-Net denoiser"),
        Line2D([0], [0], color=COLORS["head_b"],    lw=8, label="Residual fusion heads"),
        Line2D([0], [0], color=COLORS["output_b"],  lw=8, label="Final image"),
    ]
    fig.legend(
        handles=legend_lines, loc="lower center", ncol=5, fontsize=9.5,
        frameon=False, bbox_to_anchor=(0.5, 0.01),
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {args.out}")
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, dpi=220, bbox_inches="tight", pad_inches=0.1)
        print(f"Wrote {args.out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
