"""
Regenerate the framework architecture figure.

The previous figure was exported from a drawing tool and fell out of step with
the manuscript in four places: it described the classifier as "DGI
pre-training + GAT fine-tuning", which is not what the code does; it showed a
shared 169-dimensional feature matrix, although two of those features were
withdrawn; it described Phase 2 as fine-tuning; and it advertised five-seed
confidence intervals, since superseded by twenty seeds and bootstrap
intervals.

Building it from a script rather than a drawing keeps it in step: the labels
now come from the same constants the experiments use, so a figure that
disagrees with the text becomes a code change rather than an oversight.

Output: figures/fig_architecture.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import FIGURES

C_DATA = "#1b4965"
C_MOD = "#5f4b8b"
C_EVAL = "#4a5568"
C_VALID = "#2a9d8f"
C_TEXT = "#1a202c"
C_MUTED = "#6b7280"


def box(ax, x, y, w, h, fc, ec, lw=1.2, ls="solid", r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=1))


def arrow(ax, x1, y1, x2, y2, color=C_MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=color, linewidth=1.1, zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- tier 1: data and features ---------------------------------------
    box(ax, 2, 80, 96, 17, "#eef4fa", C_DATA)
    ax.text(50, 94.2, "1.  Data sources and feature engineering",
            ha="center", fontsize=11.5, weight="bold", color=C_DATA)
    for i, (t, sub) in enumerate([
            ("Elliptic Data Set", "203,769 tx · 49 steps"),
            ("Author corpus", "5,884,387 tx · 2022–2024"),
            ("Block explorers", "per-output values")]):
        x = 6 + i * 23
        ax.text(x + 10, 89.0, t, ha="center", fontsize=9.2, weight="bold",
                color=C_TEXT)
        ax.text(x + 10, 86.2, sub, ha="center", fontsize=7.8, color=C_MUTED)
    box(ax, 78, 82.5, 18, 10, "#ffffff", C_DATA, lw=1.0)
    ax.text(87, 88.6, "165 distributed", ha="center", fontsize=8.6,
            weight="bold", color=C_DATA)
    ax.text(87, 85.9, "Elliptic features", ha="center", fontsize=8.0,
            color=C_MUTED)
    ax.text(87, 83.6, "two withdrawn (§3.2)", ha="center", fontsize=7.2,
            style="italic", color=C_MUTED)

    # ---- tier 2: the three modules ---------------------------------------
    box(ax, 2, 40, 96, 35, "#ffffff", C_MOD, ls=(0, (5, 3)))
    ax.text(50, 72.0, "2.  Three modules, evaluated independently",
            ha="center", fontsize=11.5, weight="bold", color=C_MOD)
    ax.text(50, 69.0, "no information flows between modules and none is claimed",
            ha="center", fontsize=7.8, style="italic", color=C_MUTED)

    mods = [
        ("2.1  Structural screening",
         "Density-based clustering",
         ["Target is a disclosed heuristic:",
          "at least four inputs and outputs",
          "Reported as a gradient across",
          "feature regimes, not one score"]),
        ("2.2  Post-mix linkage",
         "Multi-dimensional Chamfer distance",
         ["Four-dimensional extension of",
          "the temporal baseline",
          "Null result, reported as a",
          "bounded negative finding"]),
        ("2.3  Illicit classification",
         "Frozen self-supervised + supervised",
         ["DGI encoder trained, then frozen",
          "GAT trained separately, supervised",
          "Representations concatenated;",
          "no encoder weight is fine-tuned"]),
    ]
    for i, (title, sub, bullets) in enumerate(mods):
        x = 5 + i * 31
        box(ax, x, 43, 28, 24, "#f7f5fb", C_MOD, lw=1.0)
        ax.text(x + 14, 64.0, title, ha="center", fontsize=9.0, weight="bold",
                color=C_MOD)
        ax.text(x + 14, 61.3, sub, ha="center", fontsize=7.8, color=C_TEXT)
        ax.plot([x + 3, x + 25], [59.6, 59.6], color=C_MOD, lw=0.7, alpha=0.5)
        for j, b in enumerate(bullets):
            ax.text(x + 14, 56.8 - j * 2.9, b, ha="center", fontsize=7.2,
                    color=C_MUTED)

    # ---- tier 3: validation ----------------------------------------------
    box(ax, 2, 24, 96, 12, "#eefaf7", C_VALID)
    ax.text(50, 33.6, "3.  External validation of CoinJoin status",
            ha="center", fontsize=10.5, weight="bold", color=C_VALID)
    for i, (t1, t2) in enumerate([
            ("Published protocol rules applied", "to per-output values"),
            ("6,209 candidates verified", "4,491 confirmed CoinJoins"),
            ("Screening label precision 4.07 %", "at near-total recall")]):
        x = 18 + i * 32
        ax.text(x, 29.9, t1, ha="center", fontsize=7.6, color=C_TEXT)
        ax.text(x, 27.2, t2, ha="center", fontsize=7.6, color=C_TEXT)

    # ---- tier 4: evaluation ----------------------------------------------
    box(ax, 2, 3, 96, 18, "#f5f6f8", C_EVAL)
    ax.text(50, 18.4, "4.  Evaluation protocol, uniform across every module",
            ha="center", fontsize=10.5, weight="bold", color=C_EVAL)
    evals = [
        ("Chronological splits", "no future data in fitting"),
        ("Thresholds frozen", "selected on validation only"),
        ("Twenty seeds", "plus test-set bootstrap"),
        ("Per-period metrics", "pooled figures hide drift"),
    ]
    for i, (t, sub) in enumerate(evals):
        x = 4 + i * 24
        box(ax, x, 6, 21, 9, "#ffffff", C_EVAL, lw=0.9)
        ax.text(x + 10.5, 11.9, t, ha="center", fontsize=8.4, weight="bold",
                color=C_EVAL)
        ax.text(x + 10.5, 8.6, sub, ha="center", fontsize=7.0, color=C_MUTED)

    for x in (25, 50, 75):
        arrow(ax, x, 80, x, 75.4)
    arrow(ax, 19, 43, 19, 36.2)
    arrow(ax, 50, 24, 50, 21.2)

    fig.savefig(FIGURES / "fig_architecture.png", dpi=320, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", FIGURES / "fig_architecture.png")


if __name__ == "__main__":
    main()
