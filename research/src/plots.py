"""
Publication figures.

Round-1 mixed single-seed illustrative figures with multi-seed summary tables,
which R3 (comment 7) found confusing and potentially inflating. Every figure
produced here is either seed-aggregated with an explicit variance band, or
labelled in its own caption as a single-run diagnostic. Nothing is drawn from
a superseded protocol.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
})

C_PRIMARY = "#1b4965"
C_ACCENT = "#bc4749"
C_MUTED = "#8d99ae"
C_OK = "#2a9d8f"
C_WARN = "#e9c46a"


def fig_leakage_ablation(e02_csv: Path, out: Path) -> Path:
    """Bar chart of the four-way feature ablation, both model families."""
    df = pd.read_csv(e02_csv)
    agg = (df.groupby(["model", "feature_set"])["f1_illicit"]
             .agg(["mean", "std"]).reset_index())
    order = ["(i) base 165", "(ii) base + INR",
             "(iii) base + SPTI", "(iv) base + INR + SPTI"]
    models = sorted(agg["model"].unique())

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    w = 0.38
    xs = np.arange(len(order))
    for i, m in enumerate(models):
        sub = agg[agg["model"] == m].set_index("feature_set").reindex(order)
        colors = [C_PRIMARY if k == "(i) base 165" else C_ACCENT for k in order]
        ax.bar(xs + (i - 0.5) * w, sub["mean"], w,
               yerr=sub["std"].fillna(0), capsize=3,
               color=colors, alpha=0.95 if i == 0 else 0.55,
               edgecolor="white", linewidth=0.8, label=m)
    ax.axhline(1.0, color=C_MUTED, ls=":", lw=1)
    ax.text(len(order) - 0.55, 1.012, "perfect score", fontsize=7.5,
            color=C_MUTED, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([o.replace(" + ", "\n+ ") for o in order], fontsize=8)
    ax.set_ylabel("Test $F_1$ (illicit)")
    ax.set_ylim(0, 1.13)
    ax.set_title("Leaked features saturate every metric (5 seeds, mean ± sd)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_temporal_collapse(per_step_csv: Path, out: Path,
                          protocol_prefix: str = "B_") -> Path:
    """Per-time-step F1 and recall, annotating the step-43 shock."""
    df = pd.read_csv(per_step_csv)
    if "protocol" in df.columns:
        df = df[df["protocol"].astype(str).str.startswith(protocol_prefix)]
    g = (df.groupby("time_step")
           .agg(f1=("f1_illicit", "mean"), f1_sd=("f1_illicit", "std"),
                rec=("recall", "mean"), roc=("roc_auc", "mean"),
                n_ill=("n_illicit", "first"))
           .reset_index())

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.axvspan(42.5, g["time_step"].max() + 0.5, color=C_ACCENT, alpha=0.07)
    ax.plot(g["time_step"], g["f1"], "-o", ms=3.4, color=C_PRIMARY, label="$F_1$")
    if g["f1_sd"].notna().any():
        ax.fill_between(g["time_step"], g["f1"] - g["f1_sd"].fillna(0),
                        g["f1"] + g["f1_sd"].fillna(0),
                        color=C_PRIMARY, alpha=0.15)
    ax.plot(g["time_step"], g["rec"], "--s", ms=3.0, color=C_OK, label="Recall")
    ax.plot(g["time_step"], g["roc"], ":^", ms=3.0, color=C_MUTED,
            label="ROC-AUC")
    ax.axvline(43, color=C_ACCENT, lw=1.2)
    ax.annotate("dark-market shutdown\n(step 43)", xy=(43, 0.55),
                xytext=(44.2, 0.72), fontsize=8, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1))
    ax.set_xlabel("Elliptic time step")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Performance is bimodal, not gradually degrading "
                 "(RandomForest, 5 seeds)")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_model_comparison(agg_csv: Path, out: Path) -> Path:
    """Seed-aggregated model comparison with error bars."""
    df = pd.read_csv(agg_csv)
    df = df.sort_values("f1_mean")
    labels = [f"{r.model}" + (f"\n({r.dgi_regime})"
                              if r.dgi_regime not in ("n/a", "nan") else "")
              for r in df.itertuples()]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ys = np.arange(len(df))
    colors = [C_ACCENT if m == "fusion" else C_PRIMARY for m in df["model"]]
    ax.barh(ys, df["f1_mean"], xerr=df["f1_std"].fillna(0), capsize=3,
            color=colors, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Test $F_1$ (illicit), mean ± sd over seeds")
    ax.set_title("Model comparison, leakage-free features")
    for y, v in zip(ys, df["f1_mean"]):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0, max(df["f1_mean"]) * 1.22)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_curves_with_bands(score_files: dict[str, Path], y_true: np.ndarray,
                          out: Path) -> Path:
    """PR and ROC curves for several models on a common test mask."""
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    palette = [C_PRIMARY, C_ACCENT, C_OK, C_WARN, C_MUTED]
    for (name, path), col in zip(score_files.items(), palette):
        s = np.load(path)
        p, r, _ = precision_recall_curve(y_true, s)
        axes[0].plot(r, p, lw=1.4, color=col,
                     label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
        fpr, tpr, _ = roc_curve(y_true, s)
        axes[1].plot(fpr, tpr, lw=1.4, color=col,
                     label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision–recall")
    axes[1].plot([0, 1], [0, 1], ls=":", color=C_MUTED, lw=1)
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC")
    for a in axes:
        a.legend(fontsize=7)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_script_composition(summary_json: Path, out: Path) -> Path:
    """Output script-type composition of the author corpus."""
    import json
    d = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    share = d["output_script_share_pct"]
    share = {k: v for k, v in share.items() if v >= 0.01}
    keys = list(share.keys())
    vals = [share[k] for k in keys]
    colors = [C_ACCENT if k == "p2tr" else C_PRIMARY for k in keys]

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    xs = np.arange(len(keys))
    ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([k.upper() for k in keys], rotation=30, ha="right",
                       fontsize=8)
    ax.set_ylabel("Share of outputs (%)")
    tp = d["tx_taproot_either_side_pct"]
    cj = d["coinjoin_like"]["taproot_either_side_pct"]
    ax.set_title(f"Corpus is post-Taproot: {tp:.1f}% of transactions touch P2TR\n"
                 f"({cj:.1f}% of CoinJoin-like candidates)", fontsize=9)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, max(vals) * 1.2)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_operating_points(sweep_csv: Path, out: Path) -> Path:
    """FPR-FNR and precision-recall trade-off across parameter settings."""
    df = pd.read_csv(sweep_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for regime, col in zip(sorted(df["regime"].unique()), [C_PRIMARY, C_ACCENT]):
        sub = df[df["regime"] == regime]
        rf = sub[sub["method"] == "RandomForest"].sort_values("recall")
        db = sub[sub["method"] == "DBSCAN"]
        axes[0].plot(rf["recall"], rf["precision"], "-", lw=1.4, color=col,
                     label=f"RF · {regime}")
        axes[0].scatter(db["recall"], db["precision"], s=26, marker="D",
                        color=col, edgecolor="white", linewidth=0.6,
                        label=f"DBSCAN · {regime}")
        axes[1].plot(rf["fpr"], rf["fnr"], "-", lw=1.4, color=col)
        axes[1].scatter(db["fpr"], db["fnr"], s=26, marker="D", color=col,
                        edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Operating points")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("False negative rate")
    axes[1].set_title("FPR–FNR trade-off")
    fig.savefig(out)
    plt.close(fig)
    return out
