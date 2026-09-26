"""
E03 - Leakage-free topological features: construction, verification, value.

E01/E02 withdrew the two round-1 engineered features. This experiment builds a
replacement set that is leakage-free by construction, verifies that property
mechanically, and measures whether the features carry any legitimate signal.

Three stages
------------
  1. Compute 15 label-free topological descriptors inside each node's own
     time-step subgraph.
  2. Verify label independence by permuting every label in the dataset and
     confirming the features are bit-identical.
  3. Ablate: base 165 vs base + topological, under the frozen-threshold
     protocol, over five seeds and two model families.

Outputs
-------
  results/e03_topological_features.parquet
  results/e03_verification.json
  results/e03_ablation.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS
from data import load_elliptic, labelled_masks, base_feature_columns
from features import (TOPO_FEATURES, compute_topological_features,
                      verify_label_independence)
from metrics import evaluate, select_threshold

SEEDS = [42, 43, 44, 45, 46]
FEAT_PATH = RESULTS / "e03_topological_features.parquet"


def build_model(name: str, seed: int):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed)


def main() -> None:
    nodes, edges = load_elliptic()

    # -- Stage 1: compute -------------------------------------------------------
    if FEAT_PATH.exists():
        feats = pd.read_parquet(FEAT_PATH)
        print(f"  loaded cached features {feats.shape}")
        elapsed = float("nan")
    else:
        t0 = time.time()
        feats = compute_topological_features(nodes, edges, seed=0)
        elapsed = time.time() - t0
        feats.to_parquet(FEAT_PATH, index=False)
        print(f"  computed {feats.shape} in {elapsed:.1f}s")

    # -- Stage 2: verify --------------------------------------------------------
    print("\n  verifying label independence (permutation test) ...")
    ver = verify_label_independence(nodes, edges, feats, seed=0)
    ver["compute_seconds"] = elapsed
    ver["n_features"] = len(TOPO_FEATURES)
    print(f"  invariant to label permutation: "
          f"{ver['all_features_invariant_to_label_permutation']}")

    # Correlation of each feature with the label, for the manuscript table.
    df = nodes.merge(feats, on="txId", how="left")
    lm = labelled_masks(df)
    lab = lm["train"] | lm["val"] | lm["test"]
    y = df["label"].to_numpy()
    corrs = {}
    for c in TOPO_FEATURES:
        v = df[c].to_numpy(dtype=float)[lab]
        corrs[c] = (0.0 if np.std(v) == 0
                    else round(float(np.corrcoef(v, y[lab])[0, 1]), 4))
    ver["label_correlation"] = corrs
    ver["max_abs_label_correlation"] = round(
        float(max(abs(v) for v in corrs.values())), 4)

    (RESULTS / "e03_verification.json").write_text(
        json.dumps(ver, indent=2), encoding="utf-8")

    print("\n  label correlations (leakage-free features):")
    for c, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {c:24s} r = {v:+.4f}")
    print(f"  max |r| = {ver['max_abs_label_correlation']}  "
          f"(round-1 SPTI was 0.9182)")

    # -- Stage 3: ablation ------------------------------------------------------
    base_cols = base_feature_columns(df)
    sets = {
        "base 165": base_cols,
        "base + topological": base_cols + TOPO_FEATURES,
        "topological only": TOPO_FEATURES,
    }

    rows = []
    for model_name in ("RandomForest", "HistGradientBoosting"):
        for fs_name, cols in sets.items():
            X = df[cols].to_numpy(dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            scaler = StandardScaler().fit(X[lm["train"]])
            Xs = scaler.transform(X).astype(np.float32)

            for seed in SEEDS:
                clf = build_model(model_name, seed)
                clf.fit(Xs[lm["train"]], y[lm["train"]])
                s_va = clf.predict_proba(Xs[lm["val"]])[:, 1]
                s_te = clf.predict_proba(Xs[lm["test"]])[:, 1]
                thr = select_threshold(y[lm["val"]], s_va, criterion="f1")
                m = evaluate(y[lm["test"]], s_te, thr)
                m.update(model=model_name, feature_set=fs_name, seed=seed,
                         n_features=len(cols))
                rows.append(m)
                print(f"  {model_name:22s} {fs_name:20s} seed={seed} "
                      f"F1={m['f1_illicit']:.4f} PR-AUC={m['pr_auc']:.4f} "
                      f"MCC={m['mcc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e03_ablation.csv", index=False)
    agg = (res.groupby(["model", "feature_set"])
              .agg(n_features=("n_features", "first"),
                   f1_mean=("f1_illicit", "mean"), f1_std=("f1_illicit", "std"),
                   precision_mean=("precision", "mean"),
                   recall_mean=("recall", "mean"),
                   pr_auc_mean=("pr_auc", "mean"),
                   roc_auc_mean=("roc_auc", "mean"),
                   mcc_mean=("mcc", "mean"))
              .round(4).reset_index())
    print("\n" + "=" * 100)
    print(agg.to_string(index=False))
    agg.to_csv(RESULTS / "e03_ablation_aggregate.csv", index=False)


if __name__ == "__main__":
    main()
