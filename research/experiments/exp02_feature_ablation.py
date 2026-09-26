"""
E02 - Quantifying the performance inflation caused by the leaked features.

Reviewer 1 requested exactly this ablation:
    (i)   the original 165 Elliptic features
    (ii)  + illicit_neighbor_ratio
    (iii) + shortest_path_to_illicit
    (iv)  + both

E01 established that (ii)-(iv) are contaminated. This experiment measures how
much apparent performance that contamination buys, which is the number the
manuscript needs in order to withdraw the features honestly.

Protocol
--------
  * Chronological split, fixed: train 1-34, val 35-41, test 42-49.
  * Only labelled nodes participate in fitting and scoring.
  * Features standardised with a scaler fitted on the training split alone.
  * Decision threshold maximises validation F1, then is frozen before test.
  * Repeated over several seeds; mean and standard deviation reported.
  * A gradient-boosting and a random-forest classifier are both run so the
    conclusion does not depend on one inductive bias.

Outputs
-------
  results/e02_feature_ablation.csv
  results/e02_feature_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS
from data import load_elliptic, labelled_masks, base_feature_columns
from metrics import evaluate, select_threshold

SEEDS = [42, 43, 44, 45, 46]

FEATURE_SETS = {
    "(i) base 165": [],
    "(ii) base + INR": ["inr_leaked"],
    "(iii) base + SPTI": ["spti_leaked"],
    "(iv) base + INR + SPTI": ["inr_leaked", "spti_leaked"],
}


def build_model(name: str, seed: int):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed)
    raise ValueError(name)


def main() -> None:
    nodes, _ = load_elliptic()
    eng = pd.read_parquet(RESULTS / "e01_engineered_features.parquet")
    df = nodes.merge(eng, on="txId", how="left")

    base_cols = base_feature_columns(df)
    lm = labelled_masks(df)
    y = df["label"].to_numpy()

    rows = []
    for model_name in ("RandomForest", "HistGradientBoosting"):
        for fs_name, extra in FEATURE_SETS.items():
            cols = base_cols + extra
            X = df[cols].to_numpy(dtype=np.float32)

            scaler = StandardScaler().fit(X[lm["train"]])
            Xs = scaler.transform(X).astype(np.float32)

            Xtr, ytr = Xs[lm["train"]], y[lm["train"]]
            Xva, yva = Xs[lm["val"]], y[lm["val"]]
            Xte, yte = Xs[lm["test"]], y[lm["test"]]

            for seed in SEEDS:
                clf = build_model(model_name, seed)
                clf.fit(Xtr, ytr)
                s_va = clf.predict_proba(Xva)[:, 1]
                s_te = clf.predict_proba(Xte)[:, 1]

                thr = select_threshold(yva, s_va, criterion="f1")
                m = evaluate(yte, s_te, thr)
                m.update(model=model_name, feature_set=fs_name, seed=seed,
                         n_features=len(cols))
                rows.append(m)
                print(f"  {model_name:22s} {fs_name:24s} seed={seed} "
                      f"F1={m['f1_illicit']:.4f} PR-AUC={m['pr_auc']:.4f} "
                      f"MCC={m['mcc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e02_feature_ablation.csv", index=False)

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

    summary = {"per_run": rows, "aggregate": agg.to_dict(orient="records")}
    # Inflation attributable to each leaked feature, per model.
    infl = {}
    for model_name in agg["model"].unique():
        sub = agg[agg["model"] == model_name].set_index("feature_set")
        base = sub.loc["(i) base 165", "f1_mean"]
        infl[model_name] = {
            fs: round(float(sub.loc[fs, "f1_mean"] - base), 4)
            for fs in FEATURE_SETS if fs != "(i) base 165"
        }
        infl[model_name]["base_f1"] = float(base)
    summary["f1_inflation_vs_base"] = infl

    (RESULTS / "e02_feature_ablation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\nF1 inflation attributable to leaked features:")
    print(json.dumps(infl, indent=2))


if __name__ == "__main__":
    main()
