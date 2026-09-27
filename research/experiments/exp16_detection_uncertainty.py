"""
E16 - Uncertainty on the confirmed-label detection result.

E15 reported a Matthews correlation of 0.9559 for the strict-regime model
against 0.5823 for the rule baseline, with no interval attached. That is the
omission this study criticises elsewhere, and it matters here more than usual:
the Whirlpool task is 92.8 percent positive, so the test split holds only
about a hundred negatives and every discrimination statistic rests on them.

Two sources of variation are separated, as in E10.

  Seed variance    spread of the point estimate across training runs on a
                   fixed split. Says how reproducible a number is, not how
                   far it would move on different data.

  Test variance    percentile bootstrap over the test transactions. Says how
                   far the number would move had a different sample of
                   transactions been drawn, which is the question the small
                   negative class raises.

A paired bootstrap compares the model against the rule baseline on the same
resampled transactions, so the comparison is not confounded by which
transactions a given resample happens to contain.

Resamples in which one class disappears are counted and excluded rather than
silently dropped, because with so few negatives that is a real possibility and
its frequency is itself informative.

Outputs
-------
  results/e16_detection_uncertainty.csv
  results/e16_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR
from metrics import select_threshold

SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 5000
ALPHA = 0.05

STRICT_COLS = ["fee", "fee_rate_sat_per_vbyte", "fee_rate_sat_per_byte",
               "size", "vsize", "weight", "address_reuse", "locktime",
               "version", "hour", "day_of_week", "rbf_enabled",
               "has_op_return"]
POOLS = np.array([0.001, 0.01, 0.05, 0.5])


def metrics_at(y: np.ndarray, pred: np.ndarray) -> dict:
    """Threshold-dependent metrics, computed directly for speed."""
    return {
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = 0) -> dict:
    """Percentile bootstrap over transactions, reporting degenerate draws."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ("mcc", "f1", "precision", "recall")
    vals = {k: [] for k in keys}
    degenerate = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            degenerate += 1
            continue
        m = metrics_at(y[idx], pred[idx])
        for k in keys:
            vals[k].append(m[k])
    point = metrics_at(y, pred)
    out = {"n_test": int(n), "n_negatives": int((y == 0).sum()),
           "n_positives": int((y == 1).sum()),
           "degenerate_resamples": int(degenerate), "n_boot": int(n_boot)}
    for k in keys:
        a = np.asarray(vals[k])
        out[k] = round(float(point[k]), 4)
        out[f"{k}_lo"] = round(float(np.quantile(a, ALPHA / 2)), 4)
        out[f"{k}_hi"] = round(float(np.quantile(a, 1 - ALPHA / 2)), 4)
    return out


def paired_ci(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
              metric: str = "mcc", n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired bootstrap of metric(A) - metric(B) on shared resamples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs, degenerate = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            degenerate += 1
            continue
        diffs.append(metrics_at(y[idx], pred_a[idx])[metric]
                     - metrics_at(y[idx], pred_b[idx])[metric])
    d = np.asarray(diffs)
    obs = metrics_at(y, pred_a)[metric] - metrics_at(y, pred_b)[metric]
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"metric": metric, "difference": round(float(obs), 4),
            "ci_lo": round(float(np.quantile(d, ALPHA / 2)), 4),
            "ci_hi": round(float(np.quantile(d, 1 - ALPHA / 2)), 4),
            "p_bootstrap": round(float(min(1.0, p)), 4),
            "degenerate_resamples": int(degenerate)}


def main() -> None:
    ver = pd.read_csv(RESULTS / "e14b_all_verified.csv")
    ver = ver[ver["verdict"] == "ok"]
    cand = ver[ver["group"] == "whirlpool_candidate"][
        ["txid", "confirmed_coinjoin"]]

    cols = ["txid", "timestamp", "total_output_value", "output_count"] + STRICT_COLS
    corpus = pd.read_parquet(RAW_AUTHOR, columns=cols)
    for c in STRICT_COLS:
        if corpus[c].dtype == bool:
            corpus[c] = corpus[c].astype(np.int8)
    corpus = corpus.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    d = cand.merge(corpus, on="txid", how="left").sort_values("timestamp")
    d = d.reset_index(drop=True)
    y_all = d["confirmed_coinjoin"].astype(int).to_numpy()

    n = len(d)
    i_tr, i_va = int(n * 0.6), int(n * 0.75)
    tr = np.zeros(n, bool); tr[:i_tr] = True
    va = np.zeros(n, bool); va[i_tr:i_va] = True
    te = np.zeros(n, bool); te[i_va:] = True

    X = d[STRICT_COLS].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)

    y_te = y_all[te]
    print(f"  Whirlpool task: {n:,} rows, test {int(te.sum()):,} "
          f"({int((y_te == 1).sum()):,} positive, {int((y_te == 0).sum()):,} negative)")
    print(f"  the negative class is what every discrimination statistic rests on\n")

    # -- rule baseline ------------------------------------------------------
    per_out = (d["total_output_value"] / d["output_count"].replace(0, np.nan)).to_numpy(float)
    dist = np.array([float(np.min(np.abs(v - POOLS) / POOLS)) if np.isfinite(v)
                     else 1.0 for v in per_out])
    rule_pred = (dist[te] <= 1e-7).astype(int)

    rows = []
    r = bootstrap_ci(y_te, rule_pred, seed=0)
    r.update(model="RuleBaseline", seed=-1)
    rows.append(r)
    print(f"  RuleBaseline          MCC {r['mcc']:.4f} "
          f"[{r['mcc_lo']:.4f}, {r['mcc_hi']:.4f}]   "
          f"F1 {r['f1']:.4f} [{r['f1_lo']:.4f}, {r['f1_hi']:.4f}]")

    # -- models, per seed ---------------------------------------------------
    preds_by_model: dict[str, list[np.ndarray]] = {}
    for model_name in ("HistGradientBoosting", "RandomForest"):
        preds_by_model[model_name] = []
        for seed in SEEDS:
            clf = (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                  l2_regularization=1.0,
                                                  random_state=seed)
                   if model_name == "HistGradientBoosting"
                   else RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=seed))
            clf.fit(Xs[tr], y_all[tr])
            s_va = clf.predict_proba(Xs[va])[:, 1]
            thr = select_threshold(y_all[va], s_va, criterion="f1")
            pred = (clf.predict_proba(Xs[te])[:, 1] >= thr).astype(int)
            preds_by_model[model_name].append(pred)

            rr = bootstrap_ci(y_te, pred, seed=seed)
            rr.update(model=model_name, seed=seed)
            rows.append(rr)
        mccs = [r_["mcc"] for r_ in rows if r_.get("model") == model_name]
        print(f"  {model_name:22s} MCC across seeds "
              f"{np.mean(mccs):.4f} +/- {np.std(mccs):.4f}   "
              f"per-seed 95% CI width "
              f"{np.mean([r_['mcc_hi'] - r_['mcc_lo'] for r_ in rows if r_.get('model') == model_name]):.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e16_detection_uncertainty.csv", index=False)

    # -- paired comparison against the rule ---------------------------------
    print("\n  paired bootstrap, model minus rule, on shared resamples:")
    paired = []
    for model_name, preds in preds_by_model.items():
        best = preds[0]          # primary seed, matching the headline figure
        for metric in ("mcc", "f1"):
            pc = paired_ci(y_te, best, rule_pred, metric=metric, seed=0)
            pc.update(model=model_name, comparator="RuleBaseline")
            paired.append(pc)
            if metric == "mcc":
                print(f"    {model_name:22s} d(MCC) {pc['difference']:+.4f} "
                      f"[{pc['ci_lo']:+.4f}, {pc['ci_hi']:+.4f}]  "
                      f"p={pc['p_bootstrap']:.4f}")

    agg = (res[res.seed >= 0].groupby("model")
              .agg(mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
                   mcc_lo=("mcc_lo", "mean"), mcc_hi=("mcc_hi", "mean"),
                   f1_mean=("f1", "mean"), f1_lo=("f1_lo", "mean"),
                   f1_hi=("f1_hi", "mean"))
              .round(4).reset_index())
    print("\n" + agg.to_string(index=False))

    (RESULTS / "e16_summary.json").write_text(json.dumps({
        "task": "Whirlpool candidates, strict regime (fee, size, timing only)",
        "n_test": int(te.sum()),
        "n_test_negatives": int((y_te == 0).sum()),
        "n_bootstrap": N_BOOT,
        "rule_baseline": {k: v for k, v in rows[0].items() if k != "model"},
        "per_model": agg.to_dict(orient="records"),
        "paired_vs_rule": paired,
    }, indent=2), encoding="utf-8")
    print("\n[OK] wrote results/e16_*")


if __name__ == "__main__":
    main()
