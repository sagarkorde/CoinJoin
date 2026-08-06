"""
aggregate_multiseed.py  |  Aggregate per-seed JSON results into mean +/- std
================================================================================
Usage
-----
    python aggregate_multiseed.py --dir results_multiseed --out results/multiseed_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def agg(values):
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": len(arr), "values": [float(v) for v in arr]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="results_multiseed")
    ap.add_argument("--out", type=str, default="results/multiseed_summary.json")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("seed_*.json"))
    print(f"Found {len(files)} seed result files")
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    methods = ["mixtrace_dgi_gat", "gat_no_dgi_clean"]
    metrics = ["f1_illicit", "f1_macro", "pr_auc", "roc_auc", "precision", "recall"]

    summary = {"n_seeds": len(records), "seeds": [r["seed"] for r in records]}

    for method in methods:
        summary[method] = {}
        for metric in metrics:
            vals = [r[method][metric] for r in records if method in r]
            if vals:
                summary[method][metric] = agg(vals)

    baseline_methods = ["RandomForest", "KNN", "KMeans", "RandomForest_ADASYN"]
    for bm in baseline_methods:
        summary[bm] = {}
        for metric in ["f1_illicit", "f1_macro", "pr_auc", "roc_auc", "precision", "recall"]:
            vals = [r["baselines"][bm][metric] for r in records
                    if "baselines" in r and bm in r["baselines"] and metric in r["baselines"][bm]]
            if vals:
                summary[bm][metric] = agg(vals)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")

    # human-readable printout
    print("\n=== Multi-seed summary (mean +/- std, n={}) ===".format(len(records)))
    for method in methods + baseline_methods:
        if method in summary and summary[method]:
            f1 = summary[method].get("f1_illicit", {})
            if f1:
                print(f"{method:25s}  F1={f1['mean']:.4f} +/- {f1['std']:.4f}")


if __name__ == "__main__":
    main()
