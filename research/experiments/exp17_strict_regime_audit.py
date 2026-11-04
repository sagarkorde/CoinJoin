"""
E17 - Does the strict-regime detection result survive without the size family?

the audit (audit, item 4) asks that predictors which directly or indirectly
reproduce the verification criteria be controlled. The strict regime of E15
withholds every value and count column and keeps fee, size and timing, and the
paper states that none of what remains "appears in any detection rule".

That statement is in tension with Section 3.4.2 of the same paper, which
establishes that virtual size is an affine function of the input and output
counts (vsize ~ 20.51 + 90.75 n_in + 35.15 n_out, R-squared 0.817) and that
weight is exactly four times vsize. The Whirlpool rule requires a fixed
input/output shape, so the size family is an arithmetic route back to part of
the criterion the labels were built from. The same objection the study makes
against its own screening label applies here.

This experiment decomposes the strict regime so the question is answered with
numbers rather than argument, using E15's protocol exactly: chronological
60/15/25 split by timestamp, scaler fitted on training rows, threshold selected
on validation and frozen, the same learners and seeds.

Regimes, within each candidate group:
  strict          the published strict regime, for replication
  size_only       size, vsize, weight alone: what the count proxy can do
  strict_no_size  the strict regime with the size family withheld
  fee_timing      fee and calendar fields only

If strict_no_size holds up, the headline is not an artefact of shape leakage
and the paper can say so with evidence. If it collapses to size_only, the
claim needs rewriting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_AUTHOR, RESULTS, SEEDS  # noqa: E402
from src.metrics import evaluate, select_threshold  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VERDICTS = RESULTS / "e14b_all_verified.csv"
OUT_CSV = RESULTS / "e17_strict_regime_audit.csv"
OUT_AGG = RESULTS / "e17_strict_regime_aggregate.csv"

SIZE_FAMILY = ["size", "vsize", "weight"]
FEE_TIMING = ["fee", "fee_rate_sat_per_vbyte", "fee_rate_sat_per_byte",
              "address_reuse", "locktime", "version", "hour", "day_of_week",
              "rbf_enabled", "has_op_return"]
STRICT = FEE_TIMING[:3] + SIZE_FAMILY + FEE_TIMING[3:]

REGIMES = {
    "strict": STRICT,
    "size_only": SIZE_FAMILY,
    "strict_no_size": FEE_TIMING,
    "fee_timing": ["fee", "fee_rate_sat_per_vbyte", "fee_rate_sat_per_byte",
                   "hour", "day_of_week"],
}

GROUPS = {"whirlpool": "whirlpool_candidate", "wasabi2": "wasabi2_candidate"}


def build_model(name: str, seed: int):
    """Identical to E15's, so that the `strict` regime here replicates Table 12
    rather than merely resembling it."""
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              l2_regularization=1.0,
                                              random_state=seed)
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def run(name: str, data: pd.DataFrame, cols: list[str], rows: list) -> None:
    data = data.sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    i_tr, i_va = int(n * 0.6), int(n * 0.75)
    tr = np.zeros(n, bool); tr[:i_tr] = True
    va = np.zeros(n, bool); va[i_tr:i_va] = True
    te = np.zeros(n, bool); te[i_va:] = True

    X = np.nan_to_num(data[cols].to_numpy(dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"].to_numpy().astype(int)
    Xs = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)

    print("\n  %s: %d rows, %d features, test positives %d/%d"
          % (name, n, len(cols), int(y[te].sum()), int(te.sum())))

    for model_name in ("HistGradientBoosting", "RandomForest", "LogisticRegression"):
        per_seed = []
        for seed in SEEDS:
            clf = build_model(model_name, seed)
            clf.fit(Xs[tr], y[tr])
            thr = select_threshold(y[va], clf.predict_proba(Xs[va])[:, 1],
                                   criterion="f1")
            m = evaluate(y[te], clf.predict_proba(Xs[te])[:, 1], thr)
            m.update(task=name, model=model_name, seed=seed,
                     n_features=len(cols))
            rows.append(m)
            per_seed.append(m)
        mcc = np.mean([m["mcc"] for m in per_seed])
        f1 = np.mean([m["f1_illicit"] for m in per_seed])
        print("    %-22s F1=%.4f  MCC=%.4f  (sd %.4f)"
              % (model_name, f1, mcc,
                 np.std([m["mcc"] for m in per_seed])))


def main() -> None:
    v = pd.read_csv(VERDICTS)
    cols = ["txid", "timestamp"] + sorted(set(STRICT))
    corpus = pd.read_parquet(RAW_AUTHOR, columns=cols)
    merged = v.merge(corpus, on="txid", how="inner", suffixes=("", "_c"))
    merged["y"] = merged["confirmed_coinjoin"].astype(int)

    rows: list = []
    for gname, gkey in GROUPS.items():
        g = merged[merged["group"] == gkey]
        if g["y"].nunique() < 2:
            print("  %s: single class, skipped" % gname)
            continue
        print("\n=== %s: %d candidates, %d confirmed ==="
              % (gname, len(g), int(g["y"].sum())))
        for rname, rcols in REGIMES.items():
            run("%s_%s" % (gname, rname), g, rcols, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    agg = (df.groupby(["task", "model"])
             .agg(n_features=("n_features", "first"),
                  f1=("f1_illicit", "mean"), f1_sd=("f1_illicit", "std"),
                  precision=("precision", "mean"), recall=("recall", "mean"),
                  mcc=("mcc", "mean"), mcc_sd=("mcc", "std"))
             .round(4).reset_index())
    agg.to_csv(OUT_AGG, index=False)
    print("\n" + agg.to_string(index=False))
    print("\n  written to %s" % OUT_AGG.name)


if __name__ == "__main__":
    main()
