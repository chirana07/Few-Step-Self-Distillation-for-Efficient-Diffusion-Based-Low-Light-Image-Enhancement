"""
make_fsd_pipeline_figure.py — paper Figure 4: Few-Step Self-Distillation pipeline.

Visualizes the training procedure:
  - Frozen teacher runs K=5 DDIM substeps from x_t down to t=0  →  refined target
  - Trainable student does a single forward pass at t           →  prediction
  - Loss = lambda_d * ||student - teacher||  +  lambda_a * ||student - GT||

Outputs PDF + PNG. Vector graphics, paper-quality.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


COLORS = {
    "data":    "#dbeafe",
    "data_b":  "#1d4ed8",
    "teacher": "#fee2e2",  # frozen — red border
    "teacher_b":"#b91c1c",
    "student": "#dcfce7",  # trainable — green border
    "student_b":"#15803d",
    "loss":    "#fef3c7",  # yellow
    "loss_b":  "#b45309",
    "gt":      "#f3e8ff",
    "gt_b":    "#6b21a8",
    "arrow":   "#374151",
    "label":   "#111827",
    "subtle":  "#6b7280",
}

FONT_FAMILY = "DejaVu Sans"


def box(ax, x, y, w, h, fill, edge, label, fontsize=10.5,
        sub=None, sub_fontsize=8.0, rounded=0.05, weight="bold",
        label_color=None):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={rounded}",
        linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.65, label, ha="center", va="center",
                fontsize=fontsize, color=label_color or COLORS["label"],
                fontweight=weight, family=FONT_FAMILY, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sub_fontsize, color=COLORS["subtle"],
                style="italic", family=FONT_FAMILY, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=label_color or COLORS["label"],
                fontweight=weight, family=FONT_FAMILY, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.4, style="-|>", curve=0.0,
          arrowsize=12, dashed=False):
    color = color or COLORS["arrow"]
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=arrowsize,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={curve}",
        linestyle="--" if dashed else "-",
        zorder=4,
    )
    ax.add_patch(a)


def caption(ax, x, y, text, fontsize=8.5, color=None, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            color=color or COLORS["subtle"], family=FONT_FAMILY,
            style="italic", zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./figures/figure4_fsd_pipeline.pdf")
    ap.add_argument("--out-png", default="./figures/figure4_fsd_pipeline.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ===== canvas =====
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- Inputs (top) -----
    # Ground-truth pair
    box(ax, 0.4, 7.5, 2.2, 1.0, COLORS["data"], COLORS["data_b"],
        "Training pair", sub="$y$ (low),  $x$ (normal)",
        fontsize=10.5)

    # Compute residual
    box(ax, 3.4, 7.5, 2.5, 1.0, COLORS["data"], COLORS["data_b"],
        "Clean residual",
        sub="$r = \\mathrm{clip}(x - y, -1, 1)$",
        fontsize=10.5)
    arrow(ax, 2.6, 8.0, 3.4, 8.0)

    # Sample t and forward-diffuse
    box(ax, 6.7, 7.5, 3.0, 1.0, COLORS["data"], COLORS["data_b"],
        "Forward diffuse",
        sub="$x_t = \\sqrt{\\bar\\alpha_t}\\,r + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$",
        fontsize=10.5)
    arrow(ax, 5.9, 8.0, 6.7, 8.0)
    caption(ax, 8.2, 7.20,
            "$t \\in \\{99, 74, 49, 24, 0\\}$  (the fixed 5-step DDIM schedule)",
            fontsize=8.5)

    # x_t branches into both teacher and student paths
    arrow(ax, 9.7, 8.0, 11.5, 6.0, lw=1.4, curve=0.05)
    arrow(ax, 9.7, 8.0, 11.5, 3.0, lw=1.4, curve=-0.05)

    # ----- Teacher branch (top-right) -----
    # Teacher box
    box(ax, 11.5, 5.4, 4.0, 1.2, COLORS["teacher"], COLORS["teacher_b"],
        "[ frozen ]  Teacher  $f_{\\theta_0}$",
        sub="K=5 DDIM substeps from $t$ → 0",
        fontsize=11)

    arrow(ax, 13.5, 5.4, 13.5, 4.4, lw=1.4)

    # Teacher output
    box(ax, 11.5, 3.4, 4.0, 1.0, "white", COLORS["teacher_b"],
        "$\\hat r_T$  (refined)",
        sub="multi-step refinement target",
        fontsize=10.5,
        label_color=COLORS["teacher_b"])

    # Annotation: frozen
    caption(ax, 13.5, 6.85, "frozen weights, no gradient",
            fontsize=8.5, color=COLORS["teacher_b"])

    # ----- Student branch (bottom-right) -----
    # Student box
    box(ax, 11.5, 2.0, 4.0, 1.2, COLORS["student"], COLORS["student_b"],
        "Student  $f_{\\theta_s}$  (training)",
        sub="single forward pass at  $t$",
        fontsize=11)

    arrow(ax, 13.5, 2.0, 13.5, 1.0, lw=1.4)

    # Student output
    box(ax, 11.5, 0.0, 4.0, 1.0, "white", COLORS["student_b"],
        "$\\hat r_S$  (predicted)",
        sub="single-shot prediction",
        fontsize=10.5,
        label_color=COLORS["student_b"])

    caption(ax, 13.5, 3.45, "trainable weights",
            fontsize=8.5, color=COLORS["student_b"])

    # ----- Loss block (left-center) -----
    # Distillation loss (between r_T and r_S)
    box(ax, 6.7, 1.6, 3.6, 1.1, COLORS["loss"], COLORS["loss_b"],
        "$\\mathcal{L}_{\\mathrm{distill}}$",
        sub="Charbonnier($\\hat r_S, \\hat r_T$)",
        fontsize=11)

    # arrows from teacher_out and student_out into distill loss
    arrow(ax, 11.5, 3.9, 10.3, 2.5, lw=1.4, curve=-0.18, color=COLORS["teacher_b"])
    arrow(ax, 11.5, 0.5, 10.3, 1.9, lw=1.4, curve=0.18, color=COLORS["student_b"])

    # Anchor loss (between r and r_S — uses GT residual directly)
    box(ax, 2.4, 1.6, 3.6, 1.1, COLORS["loss"], COLORS["loss_b"],
        "$\\mathcal{L}_{\\mathrm{anchor}}$",
        sub="Charbonnier($\\hat r_S$, $r$)",
        fontsize=11)
    arrow(ax, 4.6, 7.5, 4.2, 2.7, lw=1.0, dashed=True, curve=-0.15,
          color=COLORS["data_b"])
    arrow(ax, 11.5, 0.5, 6.0, 2.0, lw=1.0, dashed=True, curve=0.15,
          color=COLORS["student_b"])

    # Total loss arrow from both into a final marker (just a label for now)
    box(ax, 4.5, 0.05, 3.4, 1.0, "white", COLORS["loss_b"],
        "$\\mathcal{L} = \\lambda_d \\mathcal{L}_{\\mathrm{distill}} + \\lambda_a \\mathcal{L}_{\\mathrm{anchor}}$",
        sub="(only updates student parameters)",
        fontsize=10.5,
        label_color=COLORS["loss_b"])
    arrow(ax, 4.2, 1.6, 6.2, 1.05, lw=1.4)
    arrow(ax, 8.5, 1.6, 6.2, 1.05, lw=1.4)

    # ----- title + legend -----
    fig.suptitle(
        "Few-Step Self-Distillation (FSD) Training Pipeline",
        fontsize=12.5, y=0.96, fontweight="bold", family=FONT_FAMILY,
    )

    legend_lines = [
        Line2D([0], [0], color=COLORS["data_b"],    lw=8, label="Data flow"),
        Line2D([0], [0], color=COLORS["teacher_b"], lw=8, label="Teacher (frozen)"),
        Line2D([0], [0], color=COLORS["student_b"], lw=8, label="Student (trainable)"),
        Line2D([0], [0], color=COLORS["loss_b"],    lw=8, label="Loss"),
    ]
    fig.legend(
        handles=legend_lines, loc="lower center", ncol=4, fontsize=10,
        frameon=False, bbox_to_anchor=(0.5, 0.01),
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {args.out}")
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, dpi=220, bbox_inches="tight", pad_inches=0.1)
        print(f"Wrote {args.out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
