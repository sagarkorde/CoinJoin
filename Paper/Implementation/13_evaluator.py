"""
13_evaluator.py  |  MixTrace Pipeline — Step 13: Comprehensive Evaluation
==========================================================================
Aggregates all method results, produces a unified comparison table, and
runs statistical significance tests for the paper's evaluation section.

Methods Compared
----------------
  Baseline classifiers (Step 5):
    - Random Forest (RF)
    - K-Nearest Neighbours (KNN)
    - K-Means Clustering (unsupervised)

  GNN ablations (Step 12):
    - GAT without DGI skip (quantifies DGI contribution)

  Proposed framework (Steps 9 & 12):
    - DGI + GAT (MixTrace)

  Chamfer distance (Steps 4 & 11):
    - 1-D Chamfer Distance
    - Multi-Dimensional Chamfer Distance (MD-CD)

  Taproot analysis (Step 10):
    - Naive CIOH
    - Taproot-Aware DBSCAN

Statistical Testing
-------------------
  Wilcoxon signed-rank test on per-method F1 vs. MixTrace F1.
  (Note: meaningful only when k-fold results are available; here we compare
  point estimates and note this limitation for the paper.)

Run
---
    python 13_evaluator.py

Reads   : results/  (all Step 4–12 JSON output files)
Outputs : results/13_comparison_table.csv
          results/13_comparison_table.tex   — LaTeX table for paper
          results/13_evaluation_summary.txt
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


# ── Result Loaders ────────────────────────────────────────────────────────────

def safe_load(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    print(f"  ⚠  Not found: {path.name}")
    return None


def load_all_results() -> list[dict]:
    """Collect metrics dicts from all pipeline steps."""
    rows = []

    # ── Baselines ──
    for fname, label in [
        ("05_baseline_rf_metrics.json",     "Random Forest"),
        ("05_baseline_knn_metrics.json",    "KNN"),
        ("05_baseline_kmeans_metrics.json", "K-Means"),
    ]:
        d = safe_load(RESULTS_DIR / fname)
        if d:
            rows.append({
                "Method":          label,
                "Precision":       d.get("precision_illicit", 0),
                "Recall":          d.get("recall_illicit", 0),
                "F1 (illicit)":    d.get("f1_illicit", 0),
                "F1 (macro)":      d.get("f1_macro", 0),
                "PR-AUC":          d.get("pr_auc", 0),
                "ROC-AUC":         d.get("roc_auc", 0),
                "Category":        "Baseline Classifier",
            })

    # ── GNN Ablation ──
    d = safe_load(RESULTS_DIR / "12_final_test_metrics.json")
    if d:
        gat_no_dgi = d.get("gat_no_dgi", {})
        rows.append({
            "Method":       "GAT (no DGI)",
            "Precision":    gat_no_dgi.get("precision",  0),
            "Recall":       gat_no_dgi.get("recall",     0),
            "F1 (illicit)": gat_no_dgi.get("f1_illicit", 0),
            "F1 (macro)":   gat_no_dgi.get("f1_macro",   0),
            "PR-AUC":       gat_no_dgi.get("pr_auc",     0),
            "ROC-AUC":      gat_no_dgi.get("roc_auc",    0),
            "Category":     "GNN Ablation",
        })

    # ── Proposed: DGI + GAT ──
    if d:
        gat_full = d.get("dgi_gat_mixtrace", {})
        rows.append({
            "Method":       "MixTrace (DGI+GAT) ★",
            "Precision":    gat_full.get("precision",  0),
            "Recall":       gat_full.get("recall",     0),
            "F1 (illicit)": gat_full.get("f1_illicit", 0),
            "F1 (macro)":   gat_full.get("f1_macro",   0),
            "PR-AUC":       gat_full.get("pr_auc",     0),
            "ROC-AUC":      gat_full.get("roc_auc",    0),
            "Category":     "Proposed",
        })

    return rows


def load_chamfer_results() -> list[dict]:
    """Load Chamfer distance comparison."""
    d = safe_load(RESULTS_DIR / "11_chamfer_comparison.json")
    if d is None:
        return []
    rows = []
    for key in ["chamfer_1d", "chamfer_md"]:
        m = d.get(key, {})
        rows.append({
            "Method":       m.get("method", key),
            "Precision":    m.get("precision",  0),
            "Recall":       m.get("recall",     0),
            "F1 (illicit)": m.get("f1_score",   0),
            "F1 (macro)":   0,
            "PR-AUC":       0,
            "ROC-AUC":      0,
            "Category":     "CST Linking",
        })
    return rows


def load_taproot_results() -> list[dict]:
    """Load Taproot analysis results."""
    d = safe_load(RESULTS_DIR / "10_taproot_analysis.json")
    if d is None:
        return []
    rows = []
    for key, label in [("naive_cioh", "Naive CIOH"),
                        ("taproot_aware_dbscan", "Taproot-Aware DBSCAN")]:
        m = d.get(key, {})
        rows.append({
            "Method":       label,
            "Precision":    m.get("precision", 0),
            "Recall":       m.get("recall",    0),
            "F1 (illicit)": m.get("f1_score",  0),
            "F1 (macro)":   0,
            "PR-AUC":       0,
            "ROC-AUC":      0,
            "FPR":          m.get("false_positive_rate", 0),
            "Category":     "Taproot Analysis",
        })
    return rows


# ── LaTeX Table ───────────────────────────────────────────────────────────────

def to_latex(df: pd.DataFrame) -> str:
    """Generate a clean LaTeX booktabs table for the paper."""
    cols  = ["Method", "Precision", "Recall", "F1 (illicit)", "F1 (macro)", "PR-AUC"]
    sub   = df[[c for c in cols if c in df.columns]].copy()
    float_cols = [c for c in cols if c != "Method"]
    for c in float_cols:
        if c in sub.columns:
            sub[c] = sub[c].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Comparison of MixTrace against baseline methods on the Elliptic"
        r" Dataset (test set, chronological split).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    prev_cat = None
    for _, row in df.iterrows():
        cat = row.get("Category", "")
        if cat != prev_cat and prev_cat is not None:
            lines.append(r"\midrule")
        prev_cat = cat
        vals = []
        for c in cols:
            v = row.get(c, "—")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(" & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Statistical Testing ───────────────────────────────────────────────────────

def wilcoxon_comment(proposed_f1: float, baseline_f1: float) -> str:
    """
    Note: Wilcoxon signed-rank test requires matched pairs from k-fold runs.
    With single-split results we report absolute improvement only.
    """
    delta = proposed_f1 - baseline_f1
    return (
        f"  Absolute F1 improvement over RF baseline: {delta:+.4f}\n"
        "  (Wilcoxon signed-rank requires k-fold runs — "
        "see ablation study in paper)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 13: Comprehensive Evaluation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Gather all results ──
    classifier_rows = load_all_results()
    chamfer_rows    = load_chamfer_results()
    taproot_rows    = load_taproot_results()

    all_rows = classifier_rows + chamfer_rows + taproot_rows
    if not all_rows:
        print("\n  ⚠  No result files found. Run Steps 4–12 first.")
        return

    df = pd.DataFrame(all_rows)
    numeric_cols = ["Precision", "Recall", "F1 (illicit)", "F1 (macro)", "PR-AUC", "ROC-AUC"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ── Print comparison table ──
    print("\n── Method Comparison Table (Test Set)")
    display_cols = ["Method", "Precision", "Recall", "F1 (illicit)", "F1 (macro)", "PR-AUC"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False, float_format="{:.4f}".format))

    # ── Highlight best per metric ──
    print("\n── Best per Metric")
    for c in numeric_cols:
        if c in df.columns and df[c].max() > 0:
            best_idx = df[c].idxmax()
            print(f"  {c:<20}: {df.loc[best_idx, 'Method']:<35}  {df.loc[best_idx, c]:.4f}")

    # ── Statistical note ──
    proposed_rows = df[df["Method"].str.contains("MixTrace", na=False)]
    rf_rows       = df[df["Method"] == "Random Forest"]
    if not proposed_rows.empty and not rf_rows.empty:
        p_f1 = float(proposed_rows.iloc[0]["F1 (illicit)"])
        b_f1 = float(rf_rows.iloc[0]["F1 (illicit)"])
        print(f"\n{wilcoxon_comment(p_f1, b_f1)}")

    # ── Save CSV ──
    csv_path = RESULTS_DIR / "13_comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path.name}")

    # ── Save LaTeX ──
    tex = to_latex(df)
    tex_path = RESULTS_DIR / "13_comparison_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  Saved → {tex_path.name}")

    # ── Summary text ──
    summary_lines = [
        "=" * 65,
        "MixTrace — Evaluation Summary  (Step 13)",
        "=" * 65, "",
    ]
    for _, row in df.iterrows():
        summary_lines.append(f"  {row['Method']}")
        for c in numeric_cols:
            if c in row and row[c] != 0:
                summary_lines.append(f"    {c:<20}: {row[c]:.4f}")
        summary_lines.append("")

    summary_path = RESULTS_DIR / "13_evaluation_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  Saved → {summary_path.name}")

    print("\n[OK] Step 13 complete — run 14_visualizer.py next.\n")


if __name__ == "__main__":
    main()
