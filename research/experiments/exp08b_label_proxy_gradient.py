"""
E08b - How far does the screening label leak through proxies?

E08 removed the columns the label is computed from and a supervised learner
still reached an illicit-class F1-score above 0.91. That is not independent
detection. A Bitcoin transaction's serialised size is an affine function of its
input and output counts, so size, virtual size and weight reconstruct the very
counts the label thresholds. Fitting the counts against those columns recovers
the protocol constants directly:

    vsize  ~  20.5 + 90.75 * n_in + 35.15 * n_out     R^2 = 0.817
    weight ~  80.4 + 363.0 * n_in + 140.58 * n_out    R^2 = 0.817
    size   ~  71.3 + 163.62 * n_in + 36.32 * n_out    R^2 = 0.595

Column-level disjointness is therefore not information-level disjointness. This
experiment measures the gradient across three regimes of increasing strictness,
so the manuscript can state how much of the apparent screening performance
survives once every route back to the label is closed.

  A  all            every structural column, including the label's own inputs
  B  column-disjoint  label inputs removed, size family retained
  C  size-free        label inputs and the size family both removed
  D  value-only       only value and temporal descriptors

Outputs
-------
  results/e08b_proxy_gradient.csv
  results/e08b_proxy_gradient.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR
from metrics import evaluate, select_threshold

LABEL_INPUTS = ["input_count", "output_count", "input_output_ratio",
                "input_address_count", "output_address_count",
                "total_addresses", "input_script_count", "output_script_count",
                "is_consolidation", "is_distribution", "is_peer_to_peer",
                "is_batch_payment", "is_self_transfer"]
SIZE_FAMILY = ["size", "vsize", "weight", "fee"]
VALUE_TEMPORAL = ["total_input_value", "total_output_value", "value_difference",
                  "avg_input_value", "avg_output_value", "address_reuse",
                  "locktime", "version", "hour", "day_of_week", "rbf_enabled",
                  "has_op_return"]
RATE = ["fee_rate_sat_per_vbyte", "fee_rate_sat_per_byte"]

# avg_input_value == total_input_value / input_count, so the pair of columns
# recovers the count by division. Regime E removes the averages to close that
# route; regime F keeps only columns with no arithmetic path to either count.
AVERAGES = ["avg_input_value", "avg_output_value"]
TOTALS = ["total_input_value", "total_output_value", "value_difference"]
TEMPORAL = ["locktime", "version", "hour", "day_of_week", "rbf_enabled",
            "has_op_return", "address_reuse"]

REGIMES = {
    "A all columns": LABEL_INPUTS + SIZE_FAMILY + VALUE_TEMPORAL + RATE,
    "B column-disjoint": SIZE_FAMILY + VALUE_TEMPORAL + RATE,
    "C size-free": VALUE_TEMPORAL + RATE,
    "D value and temporal only": VALUE_TEMPORAL,
    "E no averages": TOTALS + TEMPORAL,
    "F temporal only": TEMPORAL,
}

TRAIN_END, VAL_END = "2023-12-31", "2024-04-30"
SAMPLE_N, TEST_N = 600_000, 300_000
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    cols = sorted({c for v in REGIMES.values() for c in v})
    read_cols = sorted(set(cols) | {"is_coinjoin_like", "timestamp",
                                    "input_count", "output_count"})
    df = pd.read_parquet(RAW_AUTHOR, columns=read_cols)
    for c in cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Quantify the proxy route explicitly for the manuscript.
    smp = df.sample(min(400_000, len(df)), random_state=0)
    Xc = smp[["input_count", "output_count"]].to_numpy(float)
    proxy = {}
    for t in ("size", "vsize", "weight", "fee"):
        lr = LinearRegression().fit(Xc, smp[t])
        proxy[t] = {"r2": round(float(lr.score(Xc, smp[t])), 4),
                    "coef_input": round(float(lr.coef_[0]), 2),
                    "coef_output": round(float(lr.coef_[1]), 2),
                    "intercept": round(float(lr.intercept_), 2)}
    print("  label-input reconstruction from the size family:")
    print(json.dumps(proxy, indent=2))

    # The value columns recover the counts exactly by division.
    ri = smp["total_input_value"] / smp["avg_input_value"].replace(0, np.nan)
    ro = smp["total_output_value"] / smp["avg_output_value"].replace(0, np.nan)
    rule = ((ri >= 3.5) & (ro >= 3.5)).fillna(False).astype(int)
    proxy["division_route"] = {
        "input_count_recovered_pct": round(float(
            (np.abs(ri - smp["input_count"]) < 0.5).mean() * 100), 4),
        "output_count_recovered_pct": round(float(
            (np.abs(ro - smp["output_count"]) < 0.5).mean() * 100), 4),
        "reconstructed_rule_agreement_pct": round(float(
            (rule == smp["is_coinjoin_like"].astype(int)).mean() * 100), 4),
        "note": ("avg_input_value == total_input_value / input_count, so the "
                 "pair recovers the count by division and the screening label "
                 "follows exactly."),
    }
    print("  division route:", json.dumps(proxy["division_route"], indent=2))

    t = pd.to_datetime(df["timestamp"])
    tr_m = (t <= TRAIN_END).to_numpy()
    va_m = ((t > TRAIN_END) & (t <= VAL_END)).to_numpy()
    te_m = (t > VAL_END).to_numpy()

    rng = np.random.default_rng(0)
    def sub(mask, n):
        ix = np.flatnonzero(mask)
        return ix if len(ix) <= n else rng.choice(ix, size=n, replace=False)

    itr, iva, ite = sub(tr_m, SAMPLE_N), sub(va_m, TEST_N), sub(te_m, TEST_N)
    y = df["is_coinjoin_like"].astype(int).to_numpy()
    print(f"\n  train {len(itr):,} | val {len(iva):,} | test {len(ite):,} "
          f"| test positive rate {y[ite].mean():.4f}")

    rows = []
    for name, fcols in REGIMES.items():
        X = df[fcols].to_numpy(dtype=np.float32)
        sc = StandardScaler().fit(X[itr])
        Xs = sc.transform(X).astype(np.float32)
        for seed in SEEDS:
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                         class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=seed)
            clf.fit(Xs[itr], y[itr])
            s_va = clf.predict_proba(Xs[iva])[:, 1]
            s_te = clf.predict_proba(Xs[ite])[:, 1]
            thr = select_threshold(y[iva], s_va, criterion="f1")
            m = evaluate(y[ite], s_te, thr)
            m.update(regime=name, seed=seed, n_features=len(fcols))
            rows.append(m)
        last = rows[-1]
        print(f"    {name:28s} ({len(fcols):2d} feats) "
              f"F1={last['f1_illicit']:.4f} P={last['precision']:.4f} "
              f"R={last['recall']:.4f} PR-AUC={last['pr_auc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e08b_proxy_gradient.csv", index=False)
    agg = (res.groupby("regime")
              .agg(n_features=("n_features", "first"),
                   f1=("f1_illicit", "mean"), f1_std=("f1_illicit", "std"),
                   precision=("precision", "mean"), recall=("recall", "mean"),
                   pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
                   mcc=("mcc", "mean"))
              .round(4).reset_index())
    print("\n" + "=" * 110)
    print(agg.to_string(index=False))
    agg.to_csv(RESULTS / "e08b_proxy_gradient_aggregate.csv", index=False)
    (RESULTS / "e08b_proxy_gradient.json").write_text(json.dumps({
        "label_rule": "input_count >= 4 AND output_count >= 4",
        "size_family_reconstruction": proxy,
        "aggregate": agg.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
