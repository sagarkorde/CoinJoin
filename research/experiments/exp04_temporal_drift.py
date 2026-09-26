"""
E04 - Temporal behaviour: per-time-step performance and the dark-market shock.

Two purposes.

1. Reviewer 2 asked for test performance broken out by Elliptic time step
   rather than pooled, to expose concept drift.

2. Sanity-check the leakage-free baseline against the published literature.
   E02/E03 put a strong tabular model at F1 ~ 0.62 on test steps 42-49, which
   is below the figures usually quoted for Random Forest on Elliptic. Before
   building on that number the manuscript must establish whether the gap comes
   from a weak baseline or from the evaluation window. This experiment
   therefore also reproduces the split used by Weber et al. (train 1-34,
   test 35-49), so the two protocols can be compared directly.

Outputs
-------
  results/e04_per_timestep.csv
  results/e04_protocol_comparison.csv
  results/e04_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, TRAIN_END, VAL_END
from data import load_elliptic, base_feature_columns
from metrics import evaluate, select_threshold

SEEDS = [42, 43, 44, 45, 46]


def fit_and_score(X, y, tr, va, te, seed):
    """Fit on tr, select threshold on va, score te. Returns (scores, thr)."""
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=seed)
    clf.fit(X[tr], y[tr])
    s_va = clf.predict_proba(X[va])[:, 1]
    thr = select_threshold(y[va], s_va, criterion="f1")
    s_te = clf.predict_proba(X[te])[:, 1]
    return s_te, thr


def main() -> None:
    nodes, _ = load_elliptic()
    base_cols = base_feature_columns(nodes)
    y = nodes["label"].to_numpy()
    ts = nodes["time_step"].to_numpy()
    has_label = y != -1

    X_raw = nodes[base_cols].to_numpy(dtype=np.float32)

    # ---------------------------------------------------------------- protocol A
    # This study's protocol: train 1-34, val 35-41, test 42-49.
    trA = (ts <= TRAIN_END) & has_label
    vaA = (ts > TRAIN_END) & (ts <= VAL_END) & has_label
    teA = (ts > VAL_END) & has_label

    # ---------------------------------------------------------------- protocol B
    # Literature protocol (Weber et al.): train 1-34, test 35-49, no separate
    # validation period. A threshold still has to come from somewhere that is
    # not the test set, so the last 20 % of the training period is held out.
    cutoff = int(np.quantile(ts[(ts <= TRAIN_END) & has_label], 0.8))
    trB = (ts <= cutoff) & has_label
    vaB = (ts > cutoff) & (ts <= TRAIN_END) & has_label
    teB = (ts > TRAIN_END) & has_label

    protocols = {
        "A_this_study (train 1-34, val 35-41, test 42-49)": (trA, vaA, teA),
        f"B_literature (train 1-{cutoff}, val {cutoff+1}-34, test 35-49)": (trB, vaB, teB),
    }

    proto_rows, per_step_rows = [], []
    summary: dict = {}

    for pname, (tr, va, te) in protocols.items():
        scaler = StandardScaler().fit(X_raw[tr])
        X = scaler.transform(X_raw).astype(np.float32)

        for seed in SEEDS:
            s_te, thr = fit_and_score(X, y, tr, va, te, seed)
            m = evaluate(y[te], s_te, thr)
            m.update(protocol=pname, seed=seed,
                     n_train=int(tr.sum()), n_test=int(te.sum()))
            proto_rows.append(m)

            # Per-time-step breakdown, using the same frozen threshold.
            te_ts = ts[te]
            te_y = y[te]
            for step in sorted(np.unique(te_ts)):
                sel = te_ts == step
                if sel.sum() == 0:
                    continue
                ms = evaluate(te_y[sel], s_te[sel], thr)
                ms.update(protocol=pname, seed=seed, time_step=int(step),
                          n_nodes=int(sel.sum()),
                          n_illicit=int((te_y[sel] == 1).sum()))
                per_step_rows.append(ms)

    proto = pd.DataFrame(proto_rows)
    proto.to_csv(RESULTS / "e04_protocol_comparison.csv", index=False)
    pstep = pd.DataFrame(per_step_rows)
    pstep.to_csv(RESULTS / "e04_per_timestep.csv", index=False)

    pagg = (proto.groupby("protocol")
                 .agg(f1_mean=("f1_illicit", "mean"), f1_std=("f1_illicit", "std"),
                      precision=("precision", "mean"), recall=("recall", "mean"),
                      pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
                      mcc=("mcc", "mean"), n_test=("n_test", "first"))
                 .round(4).reset_index())
    print("=" * 110)
    print("PROTOCOL COMPARISON (RandomForest, 165 base features, 5 seeds)")
    print(pagg.to_string(index=False))

    print("\n" + "=" * 110)
    print("PER-TIME-STEP TEST PERFORMANCE (protocol B, spans the shock)")
    sub = pstep[pstep["protocol"].str.startswith("B_")]
    sagg = (sub.groupby("time_step")
               .agg(n_nodes=("n_nodes", "first"), n_illicit=("n_illicit", "first"),
                    f1=("f1_illicit", "mean"), precision=("precision", "mean"),
                    recall=("recall", "mean"), roc_auc=("roc_auc", "mean"))
               .round(4).reset_index())
    print(sagg.to_string(index=False))

    # Locate the largest step-to-step collapse in recall.
    r = sagg.set_index("time_step")["recall"]
    drops = r.diff()
    shock_step = int(drops.idxmin())
    summary["protocol_comparison"] = pagg.to_dict(orient="records")
    summary["per_timestep_protocolB"] = sagg.to_dict(orient="records")
    summary["largest_recall_drop"] = {
        "time_step": shock_step,
        "recall_before": float(r.loc[shock_step - 1]) if shock_step - 1 in r.index else None,
        "recall_after": float(r.loc[shock_step]),
        "delta": float(drops.loc[shock_step]),
    }
    pre = sagg[sagg["time_step"] < shock_step]
    post = sagg[sagg["time_step"] >= shock_step]
    summary["pre_post_shock"] = {
        "shock_step": shock_step,
        "pre_mean_f1": round(float(pre["f1"].mean()), 4),
        "post_mean_f1": round(float(post["f1"].mean()), 4),
        "pre_mean_recall": round(float(pre["recall"].mean()), 4),
        "post_mean_recall": round(float(post["recall"].mean()), 4),
        "pre_illicit_total": int(pre["n_illicit"].sum()),
        "post_illicit_total": int(post["n_illicit"].sum()),
    }

    (RESULTS / "e04_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print("\nLargest recall collapse:", json.dumps(summary["largest_recall_drop"], indent=2))
    print("Pre/post shock:", json.dumps(summary["pre_post_shock"], indent=2))


if __name__ == "__main__":
    main()
