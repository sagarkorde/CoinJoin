"""
generate_multiseed_report.py  |  MixTrace Pipeline — Step 15: Multi-Seed Validation Report
================================================================================================
Generates a human-readable summary and CSV table from results/multiseed_summary.json
(produced by run_seed_experiment.py x5 seeds + aggregate_multiseed.py), and a paired
t-test comparing MixTrace against each baseline across the five seeds.

Run
---
    python generate_multiseed_report.py

Reads   : results/multiseed_summary.json
Outputs : results/15_multiseed_comparison.csv
          results/15_multiseed_summary.txt
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

s = json.loads((RESULTS_DIR / "multiseed_summary.json").read_text(encoding="utf-8"))

METHOD_LABELS = {
    "mixtrace_dgi_gat": "MixTrace (DGI+GAT)",
    "gat_no_dgi_clean": "GAT (no DGI, retrained end-to-end)",
    "RandomForest": "Random Forest",
    "KNN": "KNN",
    "KMeans": "K-Means",
    "RandomForest_ADASYN": "Random Forest + ADASYN",
}

mt_vals = s["mixtrace_dgi_gat"]["f1_illicit"]["values"]

rows = ["Method,Precision,Recall,F1 (illicit),F1 (macro),PR-AUC,ROC-AUC,F1_std,Paired_t_vs_MixTrace,p_value"]
lines = [
    "=" * 70,
    "MixTrace -- Multi-Seed Validation Summary (Step 15)",
    f"Seeds: {s['seeds']}  (n={s['n_seeds']})",
    "=" * 70,
    "",
]

for key, label in METHOD_LABELS.items():
    if key not in s or not s[key]:
        continue
    m = s[key]
    f1 = m.get("f1_illicit", {})
    f1mac = m.get("f1_macro", {})
    prauc = m.get("pr_auc", {})
    rocauc = m.get("roc_auc", {})
    prec = m.get("precision", {})
    rec = m.get("recall", {})

    if key == "mixtrace_dgi_gat":
        t_stat, p_val = "(reference)", "(reference)"
    else:
        vals = f1.get("values", [])
        try:
            t, p = stats.ttest_rel(mt_vals, vals)
            t_stat, p_val = f"{t:.3f}", f"{p:.4f}"
        except Exception:
            t_stat, p_val = "n/a", "n/a"

    lines.append(f"  {label}")
    lines.append(f"    Precision (mean+/-std) : {prec.get('mean',0):.4f} +/- {prec.get('std',0):.4f}")
    lines.append(f"    Recall    (mean+/-std) : {rec.get('mean',0):.4f} +/- {rec.get('std',0):.4f}")
    lines.append(f"    F1 illicit(mean+/-std) : {f1.get('mean',0):.4f} +/- {f1.get('std',0):.4f}")
    lines.append(f"    F1 macro  (mean+/-std) : {f1mac.get('mean',0):.4f} +/- {f1mac.get('std',0):.4f}")
    lines.append(f"    PR-AUC    (mean+/-std) : {prauc.get('mean',0):.4f} +/- {prauc.get('std',0):.4f}")
    lines.append(f"    ROC-AUC   (mean+/-std) : {rocauc.get('mean',0):.4f} +/- {rocauc.get('std',0):.4f}")
    lines.append(f"    Paired t vs MixTrace   : {t_stat}   p={p_val}")
    lines.append("")

    rows.append(
        f"{label},{prec.get('mean',0):.4f},{rec.get('mean',0):.4f},{f1.get('mean',0):.4f},"
        f"{f1mac.get('mean',0):.4f},{prauc.get('mean',0):.4f},{rocauc.get('mean',0):.4f},"
        f"{f1.get('std',0):.4f},{t_stat},{p_val}"
    )

(RESULTS_DIR / "15_multiseed_comparison.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
(RESULTS_DIR / "15_multiseed_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("Saved -> results/15_multiseed_comparison.csv")
print("Saved -> results/15_multiseed_summary.txt")
