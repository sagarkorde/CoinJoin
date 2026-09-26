"""
E08 - Non-circular CoinJoin candidate detection.

Round-1 clustered transactions on structural features and scored the clusters
against `is_coinjoin_like`, assigning each cluster a class by majority vote
over the ground-truth labels of its own members. E06 showed the label is
exactly `input_count >= 4 AND output_count >= 4`, and both counts are among the
clustering features, so the experiment was doubly compromised: the target was a
function of the inputs, and the cluster-to-class map was fitted on the same
transactions it was scored on.

This experiment rebuilds it with three corrections.

  1. **Chronological split.** The corpus spans 2022-07 to 2025-07. Train on the
     earliest period, select on the middle, test on the latest. Round-1 had no
     temporal separation at all.
  2. **Frozen cluster-to-class map** (R1 comment 4). Clusters are fitted on
     training transactions only; the majority-vote map is frozen; test
     transactions are assigned to clusters by nearest fitted centroid and
     scored without ever contributing to the map.
  3. **Two feature regimes.** `all` keeps input_count / output_count and so
     reproduces the circularity. `disjoint` removes them and every column
     deterministically related to them, which measures what is actually
     detectable without reading the label's own inputs.

A supervised reference and an operating-point sweep are included, the latter
answering R1 comment 12, which asked for the precision-recall trade-off to be
shown across parameter settings rather than at a single point.

Outputs
-------
  results/e08_coinjoin_detection.csv
  results/e08_operating_points.csv
  results/e08_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR
from metrics import evaluate, select_threshold

# Columns the label is computed from, plus columns that are deterministic
# functions of them. All are excluded in the "disjoint" regime.
LABEL_INPUTS = ["input_count", "output_count", "input_output_ratio",
                "input_address_count", "output_address_count",
                "total_addresses", "input_script_count", "output_script_count",
                "is_consolidation", "is_distribution", "is_peer_to_peer",
                "is_batch_payment", "is_self_transfer"]

# Value, fee, size and temporal descriptors. None enters the label rule.
DISJOINT_COLS = ["total_input_value", "total_output_value", "fee",
                 "value_difference", "avg_input_value", "avg_output_value",
                 "size", "vsize", "weight", "fee_rate_sat_per_vbyte",
                 "fee_rate_sat_per_byte", "address_reuse", "locktime",
                 "version", "hour", "day_of_week", "rbf_enabled",
                 "has_op_return"]

ALL_COLS = sorted(set(LABEL_INPUTS + DISJOINT_COLS))

# The corpus is dense from 2022-07 to 2024-09 and effectively stops there
# (2024-10 contains 102 rows, with a handful in 2025). The split is placed
# inside the dense region so the test period carries a usable positive count.
TRAIN_END = "2023-12-31"
VAL_END = "2024-04-30"
SAMPLE_N = 600_000      # stratified working sample; DBSCAN is O(n^2) in memory
TEST_N = 300_000        # test sample; the dense test period holds ~1.55 M rows
DBSCAN_FIT_N = 40_000   # transactions DBSCAN itself is fitted on
SEEDS = [42, 43, 44, 45, 46]


def load_corpus() -> pd.DataFrame:
    cols = ALL_COLS + ["is_coinjoin_like", "timestamp"]
    df = pd.read_parquet(RAW_AUTHOR, columns=cols)
    for c in ALL_COLS:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def chrono_split(df: pd.DataFrame):
    t = pd.to_datetime(df["timestamp"])
    return (t <= TRAIN_END).to_numpy(), \
           ((t > TRAIN_END) & (t <= VAL_END)).to_numpy(), \
           (t > VAL_END).to_numpy()


def dbscan_frozen(Xtr, ytr, Xva, Xte, eps: float, min_samples: int, seed: int):
    """
    Fit DBSCAN on training data, freeze a cluster-to-score map, then assign
    validation and test points to the nearest fitted centroid.

    Round-1 labelled each cluster by majority vote over its members. Under a
    2 % positive class majority vote makes almost every cluster negative, which
    is an artefact of the imbalance rather than a property of the clustering.
    Each cluster is therefore scored by its *training* positive rate, and the
    cut-off on that score is selected on validation and frozen, exactly as for
    every other model in this study. Noise points keep the global training
    positive rate.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(DBSCAN_FIT_N, len(Xtr)), replace=False)
    Xf, yf = Xtr[idx], ytr[idx]

    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree", n_jobs=-1)
    lab = db.fit_predict(Xf)

    centroids, score = [], []
    for c in sorted(set(lab)):
        if c == -1:
            continue
        m = lab == c
        centroids.append(Xf[m].mean(axis=0))
        # Positive rate over TRAINING members only; frozen from here on.
        score.append(float(yf[m].mean()))
    if not centroids:
        z = np.full(len(Xte), float(yf.mean()))
        return np.full(len(Xva), float(yf.mean())), z, 0

    C = np.vstack(centroids)
    score = np.asarray(score, dtype=float)

    def assign(X):
        out = np.empty(len(X), dtype=float)
        step = 20_000
        for i in range(0, len(X), step):
            blk = X[i:i + step]
            d = ((blk[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            out[i:i + step] = score[d.argmin(axis=1)]
        return out

    return assign(Xva), assign(Xte), len(centroids)


def main() -> None:
    print("  loading corpus ...")
    df = load_corpus()
    print(f"  {len(df):,} transactions")

    tr_m, va_m, te_m = chrono_split(df)
    print(f"  chronological split: train {tr_m.sum():,} | "
          f"val {va_m.sum():,} | test {te_m.sum():,}")

    # Stratified subsample per split to keep DBSCAN and the sweep tractable.
    rng = np.random.default_rng(0)
    def sub(mask, n):
        ix = np.flatnonzero(mask)
        return ix if len(ix) <= n else rng.choice(ix, size=n, replace=False)

    itr = sub(tr_m, SAMPLE_N)
    iva = sub(va_m, TEST_N)
    ite = sub(te_m, TEST_N)
    y = df["is_coinjoin_like"].astype(int).to_numpy()
    print(f"  working sample: {len(itr):,} / {len(iva):,} / {len(ite):,}  "
          f"(positive rate train {y[itr].mean():.4f}, test {y[ite].mean():.4f})")

    rows, sweep_rows = [], []

    for regime, cols in (("all (reproduces round-1)", ALL_COLS),
                         ("disjoint from label", DISJOINT_COLS)):
        X = df[cols].to_numpy(dtype=np.float32)
        scaler = StandardScaler().fit(X[itr])
        Xs = scaler.transform(X).astype(np.float32)
        Xtr, Xva, Xte = Xs[itr], Xs[iva], Xs[ite]
        ytr, yva, yte = y[itr], y[iva], y[ite]

        # -- structural DBSCAN with a frozen cluster-to-class map ------------
        for seed in SEEDS:
            s_va, s_te, n_cl = dbscan_frozen(Xtr, ytr, Xva, Xte, eps=0.6,
                                             min_samples=5, seed=seed)
            thr = select_threshold(yva, s_va, criterion="f1")
            m = evaluate(yte, s_te, thr)
            m.update(method="DBSCAN (frozen map)", regime=regime, seed=seed,
                     n_clusters=n_cl)
            rows.append(m)
            print(f"    DBSCAN   {regime:26s} seed={seed} clusters={n_cl:4d} "
                  f"F1={m['f1_illicit']:.4f} P={m['precision']:.4f} "
                  f"R={m['recall']:.4f} FPR={m['fpr']:.4f}")

        # -- supervised reference --------------------------------------------
        for seed in SEEDS:
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                         class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=seed)
            clf.fit(Xtr, ytr)
            s_va = clf.predict_proba(Xva)[:, 1]
            s_te = clf.predict_proba(Xte)[:, 1]
            thr = select_threshold(yva, s_va, criterion="f1")
            m = evaluate(yte, s_te, thr)
            m.update(method="RandomForest (supervised)", regime=regime,
                     seed=seed, n_clusters=0)
            rows.append(m)
            print(f"    RF       {regime:26s} seed={seed} "
                  f"F1={m['f1_illicit']:.4f} P={m['precision']:.4f} "
                  f"R={m['recall']:.4f} PR-AUC={m['pr_auc']:.4f}")

            # Operating-point sweep on the primary seed (R1 comment 12).
            if seed == SEEDS[0]:
                for q in np.linspace(0.50, 0.999, 40):
                    t = float(np.quantile(s_te, q))
                    mm = evaluate(yte, s_te, t)
                    mm.update(method="RandomForest", regime=regime, quantile=q)
                    sweep_rows.append(mm)

        # -- DBSCAN eps sweep, primary seed ----------------------------------
        for eps in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
            s_va, s_te, n_cl = dbscan_frozen(Xtr, ytr, Xva, Xte, eps=eps,
                                             min_samples=5, seed=SEEDS[0])
            thr = select_threshold(yva, s_va, criterion="f1")
            mm = evaluate(yte, s_te, thr)
            mm.update(method="DBSCAN", regime=regime, eps=eps, n_clusters=n_cl)
            sweep_rows.append(mm)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e08_coinjoin_detection.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(RESULTS / "e08_operating_points.csv",
                                    index=False)

    agg = (res.groupby(["method", "regime"])
              .agg(f1=("f1_illicit", "mean"), f1_std=("f1_illicit", "std"),
                   precision=("precision", "mean"), recall=("recall", "mean"),
                   fpr=("fpr", "mean"), fnr=("fnr", "mean"),
                   mcc=("mcc", "mean"), pr_auc=("pr_auc", "mean"),
                   clusters=("n_clusters", "mean"))
              .round(4).reset_index())
    print("\n" + "=" * 118)
    print(agg.to_string(index=False))
    agg.to_csv(RESULTS / "e08_aggregate.csv", index=False)
    (RESULTS / "e08_summary.json").write_text(
        json.dumps({"aggregate": agg.to_dict(orient="records"),
                    "label_rule": "input_count >= 4 AND output_count >= 4",
                    "split": {"train_end": TRAIN_END, "val_end": VAL_END},
                    "n_train": int(len(itr)), "n_test": int(len(ite))},
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
