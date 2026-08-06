"""
14_visualizer.py  |  MixTrace Pipeline — Step 14: Publication-Quality Figures
===============================================================================
Generates all figures for the SCI journal paper from the results produced
by Steps 4–13.  Figures are saved as high-resolution PDF (vector) and PNG
(300 DPI raster) for maximum compatibility with journal submission systems.

Figures Generated
-----------------
  Fig 1  — Dataset class distribution (Elliptic)
  Fig 2  — DGI training loss curve
  Fig 3  — GAT training curves (F1 on train/val)
  Fig 4  — Precision-Recall curves: all classifiers
  Fig 5  — ROC curves: all classifiers
  Fig 6  — Confusion matrices: RF vs. MixTrace (side-by-side)
  Fig 7  — Random Forest feature importance (top 20)
  Fig 8  — t-SNE of learned GAT node embeddings (test set)
  Fig 9  — Chamfer distance comparison: 1D vs. MD-CD
  Fig 10 — Taproot FPR comparison: Naive CIOH vs. DBSCAN
  Fig 11 — Method comparison bar chart (F1 illicit, PR-AUC)

Run
---
    python 14_visualizer.py

Reads   : results/  (Steps 4–13 outputs)
          models/   (GAT model, DGI loss curve)
Outputs : figures/  (PDF + PNG for each figure)
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Agg")   # non-interactive backend for server/headless use
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
MODELS_DIR  = HERE / "models"
DATA_PROC   = HERE / "data" / "processed"
FIGURES_DIR = HERE / "figures"

# ── Plot Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "lines.linewidth":  1.8,
})
PALETTE = ["#2B6CB0", "#E53E3E", "#38A169", "#D69E2E", "#805AD5", "#319795"]

METHOD_COLORS = {
    "Random Forest":          PALETTE[0],
    "KNN":                    PALETTE[1],
    "K-Means":                PALETTE[2],
    "GAT (no DGI)":           PALETTE[3],
    "MixTrace (DGI+GAT) ★":   PALETTE[4],
    "1D-Chamfer":             PALETTE[5],
    "MD-Chamfer":             "#C05621",
}


def save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → figures/{name}.pdf  |  .png")


def safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Fig 1: Dataset Class Distribution ────────────────────────────────────────

def fig_class_distribution() -> None:
    stats_path = RESULTS_DIR / "01_dataset_stats.txt"
    if not stats_path.exists():
        print("  ⚠  Skipping Fig 1 — run Step 1 first")
        return

    labels = ["Illicit", "Licit", "Unknown"]
    counts = [4545, 42019, 157205]   # Elliptic ground-truth values
    colors = [PALETTE[0], PALETTE[2], "#CBD5E0"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Elliptic
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[0].set_title("Elliptic Dataset — Node Label Distribution")

    # Bar chart (labeled only)
    axes[1].bar(["Illicit", "Licit"], [4545, 42019],
                color=[PALETTE[0], PALETTE[2]], edgecolor="white", linewidth=0.8)
    axes[1].set_title("Labeled Nodes (Elliptic)")
    axes[1].set_ylabel("Count")
    for i, v in enumerate([4545, 42019]):
        axes[1].text(i, v + 300, f"{v:,}", ha="center", fontsize=10)

    fig.suptitle("Figure 1 — Dataset Class Distribution", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig01_class_distribution")


# ── Fig 2: DGI Training Loss ──────────────────────────────────────────────────

def fig_dgi_loss() -> None:
    loss_path = RESULTS_DIR / "07_dgi_training_loss.csv"
    if not loss_path.exists():
        print("  ⚠  Skipping Fig 2 — run Step 7 first")
        return

    df  = pd.read_csv(loss_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["loss"], color=PALETTE[0], linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DGI Loss (BCE)")
    ax.set_title("Figure 2 — Deep Graph Infomax Pre-training Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, "fig02_dgi_loss")


# ── Fig 3: GAT Training Curves ────────────────────────────────────────────────

def fig_training_curves() -> None:
    curve_path = RESULTS_DIR / "09_training_curves.csv"
    if not curve_path.exists():
        print("  ⚠  Skipping Fig 3 — run Step 9 first")
        return

    df  = pd.read_csv(curve_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss",
                 color=PALETTE[0])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_f1"], label="Train F1", color=PALETTE[0])
    axes[1].plot(df["epoch"], df["val_f1"],   label="Val F1",   color=PALETTE[1],
                 linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1 (illicit class)")
    axes[1].set_title("F1 on Train / Validation")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle("Figure 3 — MixTrace GAT Training Curves", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig03_training_curves")


# ── Fig 4: Precision-Recall Curves ───────────────────────────────────────────

def fig_pr_curves() -> None:
    pred_path = RESULTS_DIR / "12_test_predictions.csv"
    if not pred_path.exists():
        print("  ⚠  Skipping Fig 4 — run Step 12 first")
        return

    pred_df = pd.read_csv(pred_path)
    test    = pred_df[pred_df["in_test_set"] & (pred_df["label_true"] >= 0)]
    y_true  = test["label_true"].values

    fig, ax = plt.subplots(figsize=(7, 6))

    # MixTrace DGI+GAT
    prec, rec, _ = precision_recall_curve(y_true, test["prob_illicit"].values)
    ax.plot(rec, prec, label="MixTrace (DGI+GAT)", color=METHOD_COLORS["MixTrace (DGI+GAT) ★"],
            linewidth=2.2)

    # GAT without DGI
    prec_nd, rec_nd, _ = precision_recall_curve(y_true, test["prob_no_dgi"].values)
    ax.plot(rec_nd, prec_nd, label="GAT (no DGI)", color=METHOD_COLORS["GAT (no DGI)"],
            linestyle="--")

    # Load RF predictions if available
    rf_model_path = MODELS_DIR / "rf_baseline.pkl"
    if rf_model_path.exists():
        X_te = np.load(DATA_PROC / "elliptic_X_all.npy")
        mask = pred_df["in_test_set"].values & (pred_df["label_true"].values >= 0)
        rf   = joblib.load(rf_model_path)
        rf_proba = rf.predict_proba(X_te[mask])[:, 1]
        prec_rf, rec_rf, _ = precision_recall_curve(y_true, rf_proba)
        ax.plot(rec_rf, prec_rf, label="Random Forest", color=METHOD_COLORS["Random Forest"],
                linestyle="-.")

    # Chance line
    pos_rate = y_true.mean()
    ax.axhline(pos_rate, color="grey", linestyle=":", linewidth=1.2, label="Chance")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Figure 4 — Precision-Recall Curves (Test Set, illicit class)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig04_pr_curves")


# ── Fig 5: ROC Curves ─────────────────────────────────────────────────────────

def fig_roc_curves() -> None:
    pred_path = RESULTS_DIR / "12_test_predictions.csv"
    if not pred_path.exists():
        print("  ⚠  Skipping Fig 5 — run Step 12 first")
        return

    pred_df = pd.read_csv(pred_path)
    test    = pred_df[pred_df["in_test_set"] & (pred_df["label_true"] >= 0)]
    y_true  = test["label_true"].values

    fig, ax = plt.subplots(figsize=(7, 6))

    for prob_col, label, color, ls in [
        ("prob_illicit", "MixTrace (DGI+GAT)", METHOD_COLORS["MixTrace (DGI+GAT) ★"], "-"),
        ("prob_no_dgi",  "GAT (no DGI)",       METHOD_COLORS["GAT (no DGI)"],          "--"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, test[prob_col].values)
        from sklearn.metrics import auc as sk_auc
        auc_val = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label}  (AUC={auc_val:.3f})", color=color, linestyle=ls)

    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 5 — ROC Curves (Test Set, illicit class)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig05_roc_curves")


# ── Fig 6: Confusion Matrices ─────────────────────────────────────────────────

def fig_confusion_matrices() -> None:
    pred_path = RESULTS_DIR / "12_test_predictions.csv"
    if not pred_path.exists():
        print("  ⚠  Skipping Fig 6 — run Step 12 first")
        return

    pred_df  = pd.read_csv(pred_path)
    test_lab = pred_df[pred_df["in_test_set"] & (pred_df["label_true"] >= 0)]
    y_true   = test_lab["label_true"].values
    y_pred   = test_lab["pred_illicit"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MixTrace
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Licit", "Illicit"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("MixTrace (DGI+GAT)")

    # RF baseline (if model saved)
    rf_model_path = MODELS_DIR / "rf_baseline.pkl"
    if rf_model_path.exists():
        X_te = np.load(DATA_PROC / "elliptic_X_all.npy")
        mask = pred_df["in_test_set"].values & (pred_df["label_true"].values >= 0)
        rf   = joblib.load(rf_model_path)
        y_pred_rf = rf.predict(X_te[mask])
        cm_rf = confusion_matrix(y_true, y_pred_rf)
        disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=["Licit", "Illicit"])
        disp_rf.plot(ax=axes[1], colorbar=False, cmap="Oranges")
        axes[1].set_title("Random Forest Baseline")
    else:
        axes[1].set_visible(False)

    fig.suptitle("Figure 6 — Confusion Matrices (Test Set)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig06_confusion_matrices")


# ── Fig 7: Feature Importance ─────────────────────────────────────────────────

def fig_feature_importance() -> None:
    rf_model_path = MODELS_DIR / "rf_baseline.pkl"
    feat_path     = DATA_PROC / "author_feature_names.txt"
    elliptic_path = DATA_PROC / "elliptic_X_all.npy"

    if not rf_model_path.exists():
        print("  ⚠  Skipping Fig 7 — run Step 5 first")
        return

    rf = joblib.load(rf_model_path)
    importances = rf.feature_importances_
    n_feats = len(importances)

    # Use generic feature names if the names file is for Author Dataset
    feat_names = [f"feat_{i}" for i in range(n_feats)]

    indices = np.argsort(importances)[::-1][:20]
    top_names  = [feat_names[i]   for i in indices]
    top_imps   = [importances[i]  for i in indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_names)), top_imps[::-1],
            color=PALETTE[0], edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Figure 7 — Random Forest Feature Importance (Top 20)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, "fig07_feature_importance")


# ── Fig 8: t-SNE of GAT Embeddings ────────────────────────────────────────────

def fig_tsne_embeddings() -> None:
    pred_path = RESULTS_DIR / "12_test_predictions.csv"
    if not pred_path.exists():
        print("  ⚠  Skipping Fig 8 — run Step 12 first")
        return

    try:
        import torch
        import torch.nn.functional as F
        from sklearn.manifold import TSNE
    except ImportError:
        print("  ⚠  Skipping Fig 8 — torch or sklearn not available")
        return

    graph_path = DATA_PROC / "elliptic_graph.pt"
    gat_path   = MODELS_DIR / "mixtrace_gat_best.pt"
    if not graph_path.exists() or not gat_path.exists():
        print("  ⚠  Skipping Fig 8 — graph or model not found")
        return

    device = torch.device("cpu")
    data   = torch.load(graph_path, map_location=device, weights_only=False)
    ckpt   = torch.load(gat_path,   map_location=device, weights_only=False)

    pred_df  = pd.read_csv(pred_path)
    test_labeled = pred_df["in_test_set"].values & (pred_df["label_true"].values >= 0)
    y_true   = data.y.numpy()[test_labeled]

    # Simplified embedding extraction using the first two GAT layers only
    from torch_geometric.nn import GATConv, BatchNorm
    import torch.nn as nn

    class _EmbExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            in_dim     = ckpt["in_dim"]
            hidden_dim = ckpt["hidden_dim"]
            n_heads    = ckpt["n_heads"]
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            self.gat1 = GATConv(hidden_dim, hidden_dim, heads=n_heads, concat=True)
            self.bn1  = BatchNorm(hidden_dim * n_heads)
            self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1, concat=False)
            self.bn2  = BatchNorm(hidden_dim)

    ext = _EmbExtractor()
    sd  = {k: v for k, v in ckpt["state_dict"].items()
           if any(k.startswith(p) for p in ["input_proj", "gat1", "bn1", "gat2", "bn2"])}
    ext.load_state_dict(sd, strict=False)
    ext.eval()

    with torch.no_grad():
        h = F.elu(ext.input_proj(data.x))
        h = F.elu(ext.bn1(ext.gat1(h, data.edge_index)))
        h = F.elu(ext.bn2(ext.gat2(h, data.edge_index)))
    embeds = h.numpy()[test_labeled]

    print("  Computing t-SNE (may take ~1 min) …")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(embeds[:3000])   # cap at 3000 for speed

    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = [PALETTE[0] if y == 0 else PALETTE[1] for y in y_true[:3000]]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.6)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[0], label="Licit"),
        Patch(facecolor=PALETTE[1], label="Illicit"),
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.set_title("Figure 8 — t-SNE of GAT Node Embeddings (Test Set)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    save_fig(fig, "fig08_tsne_embeddings")


# ── Fig 9: Chamfer Distance Comparison ────────────────────────────────────────

def fig_chamfer_comparison() -> None:
    comp_path = RESULTS_DIR / "11_chamfer_comparison.json"
    sw1d_path = RESULTS_DIR / "11_chamfer_threshold_sweep_1d.csv"
    swmd_path = RESULTS_DIR / "11_chamfer_threshold_sweep_md.csv"

    if not comp_path.exists():
        print("  ⚠  Skipping Fig 9 — run Step 11 first")
        return

    comp  = safe_load(comp_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # F1 vs. threshold sweep
    for path, label, color, ax in [
        (sw1d_path, "1D-Chamfer",  PALETTE[1], axes[0]),
        (swmd_path, f"MD-Chamfer (d=4)", PALETTE[4], axes[0]),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            ax.plot(df["threshold"], df["f1"], label=label, color=color)
    axes[0].set_xlabel("Threshold τ")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 vs. Threshold")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Bar chart: precision / recall / F1 comparison
    methods  = ["1D-Chamfer", "MD-Chamfer (d=4)"]
    f1_vals  = [comp["chamfer_1d"]["f1_score"],  comp["chamfer_md"]["f1_score"]]
    pr_vals  = [comp["chamfer_1d"]["precision"], comp["chamfer_md"]["precision"]]
    re_vals  = [comp["chamfer_1d"]["recall"],    comp["chamfer_md"]["recall"]]

    x    = np.arange(len(methods))
    w    = 0.25
    axes[1].bar(x - w, pr_vals, w, label="Precision", color=PALETTE[0])
    axes[1].bar(x,     f1_vals, w, label="F1",        color=PALETTE[1])
    axes[1].bar(x + w, re_vals, w, label="Recall",    color=PALETTE[2])
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Score")
    axes[1].set_title("1D vs. Multi-Dimensional Chamfer Distance")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Figure 9 — Chamfer Distance Comparison (CST Linking)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig09_chamfer_comparison")


# ── Fig 10: Taproot FPR Analysis ─────────────────────────────────────────────

def fig_taproot_analysis() -> None:
    tap_path = RESULTS_DIR / "10_taproot_analysis.json"
    if not tap_path.exists():
        print("  ⚠  Skipping Fig 10 — run Step 10 first")
        return

    data    = safe_load(tap_path)
    methods = ["Naive CIOH", "Taproot-Aware DBSCAN"]
    keys    = ["naive_cioh", "taproot_aware_dbscan"]
    fprs    = [data[k]["false_positive_rate"] for k in keys]
    fnrs    = [data[k]["false_negative_rate"] for k in keys]
    f1s     = [data[k]["f1_score"]            for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(methods)); w = 0.3

    axes[0].bar(x - w/2, fprs, w, label="FPR", color=PALETTE[0])
    axes[0].bar(x + w/2, fnrs, w, label="FNR", color=PALETTE[1])
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("False Positive / Negative Rates")
    axes[0].legend(); axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    for xi, fpr_val in zip(x - w/2, fprs):
        axes[0].text(xi, fpr_val + 0.005, f"{fpr_val:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, f1s, 0.4, color=[PALETTE[2], PALETTE[4]])
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score Comparison")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Figure 10 — Taproot-Aware Clustering vs. Naive CIOH",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig10_taproot_analysis")


# ── Fig 11: Method Comparison Bar Chart ──────────────────────────────────────

def fig_method_comparison() -> None:
    comp_path = RESULTS_DIR / "13_comparison_table.csv"
    if not comp_path.exists():
        print("  ⚠  Skipping Fig 11 — run Step 13 first")
        return

    df = pd.read_csv(comp_path)
    classifier_df = df[df["Category"].isin(
        ["Baseline Classifier", "GNN Ablation", "Proposed"]
    )].copy()

    if classifier_df.empty:
        print("  ⚠  No classifier results in comparison table")
        return

    methods  = classifier_df["Method"].tolist()
    f1_vals  = classifier_df["F1 (illicit)"].fillna(0).tolist()
    pr_vals  = classifier_df["PR-AUC"].fillna(0).tolist()

    x = np.arange(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w/2, f1_vals, w, label="F1 (illicit class)", color=PALETTE[0])
    bars2 = ax.bar(x + w/2, pr_vals, w, label="PR-AUC",             color=PALETTE[1])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Figure 11 — Method Comparison: F1 (illicit) and PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_fig(fig, "fig11_method_comparison")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 14: Publication Figures")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving all figures to  {FIGURES_DIR.relative_to(HERE)}/\n")

    fig_class_distribution()
    fig_dgi_loss()
    fig_training_curves()
    fig_pr_curves()
    fig_roc_curves()
    fig_confusion_matrices()
    fig_feature_importance()
    fig_tsne_embeddings()
    fig_chamfer_comparison()
    fig_taproot_analysis()
    fig_method_comparison()

    print(f"\n[OK] Step 14 complete. All figures saved to figures/\n"
          f"     Run Steps 1–13 to generate all result files before\n"
          f"     running this step for full figure output.\n")


if __name__ == "__main__":
    main()
