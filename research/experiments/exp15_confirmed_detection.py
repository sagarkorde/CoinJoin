"""
E15 - CoinJoin detection against externally confirmed labels.

E14b produced what the study previously lacked: 4,491 transactions confirmed
as CoinJoin outputs from blockchain data, and 1,168 transactions that carry
CoinJoin shape but failed verification. This experiment uses them as a labelled
dataset.

Two tasks, because only one of them is informative.

  Task A, confirmed CoinJoin against random corpus transactions.
    Reported for completeness and labelled trivial. Whirlpool is 5-in/5-out
    with equal outputs and Wasabi2 has at least fifty inputs, so shape alone
    separates these classes almost perfectly. A high score here measures the
    obviousness of the sampling frame, not detection ability.

  Task B, confirmed against unconfirmed within the candidate pool.
    The informative task. Every transaction here already satisfies the shape
    conditions the corpus columns can express, so those columns are close to
    constant across the classes and cannot carry the decision. What separates
    the two is whether the outputs are exactly equal at a pool denomination,
    and the corpus does not record per-output values. The question is
    therefore whether fee, size, timing and script composition can stand in
    for information the corpus does not hold. This is the forensic problem:
    rejecting look-alikes.

Both tasks use a chronological split, so a model is never fitted on
transactions later than those it is scored on.

Outputs
-------
  results/e15_detection.csv
  results/e15_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR
from metrics import evaluate, select_threshold

SEEDS = [42, 43, 44, 45, 46]
N_RANDOM_NEGATIVES = 40_000

# Columns that express the candidate-selection conditions. Retained for
# Task A, which is trivial anyway, and withheld from Task B, where leaving
# them in would let a model re-derive candidacy instead of discriminating.
SHAPE_COLS = ["input_count", "output_count", "input_output_ratio"]

# Everything else the corpus offers that is not an identifier or a timestamp.
# The strictest regime. Every value and count column is withheld, because
# each offers an arithmetic route back to the per-output value that defines
# the label: avg_output_value IS the common output value of a confirmed
# Whirlpool transaction, and total_output_value divided by output_count
# recovers it. What remains is fee, size and timing, none of which appears in
# any detection rule. If separation survives here it is not definitional.
STRICT_COLS = ["fee", "fee_rate_sat_per_vbyte", "fee_rate_sat_per_byte",
               "size", "vsize", "weight", "address_reuse", "locktime",
               "version", "hour", "day_of_week", "rbf_enabled",
               "has_op_return"]

OTHER_COLS = ["total_input_value", "total_output_value", "fee",
              "value_difference", "avg_input_value", "avg_output_value",
              "size", "vsize", "weight", "fee_rate_sat_per_vbyte",
              "fee_rate_sat_per_byte", "address_reuse", "locktime", "version",
              "hour", "day_of_week", "rbf_enabled", "has_op_return",
              "is_self_transfer", "is_consolidation", "is_distribution",
              "is_peer_to_peer", "is_batch_payment"]


def build_model(name: str, seed: int):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              l2_regularization=1.0,
                                              random_state=seed)
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def run_task(name: str, data: pd.DataFrame, feature_cols: list[str],
             rows: list, note: str) -> None:
    """Chronological split, threshold frozen on validation, several learners."""
    data = data.sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    i_tr, i_va = int(n * 0.6), int(n * 0.75)
    tr = np.zeros(n, bool); tr[:i_tr] = True
    va = np.zeros(n, bool); va[i_tr:i_va] = True
    te = np.zeros(n, bool); te[i_va:] = True

    X = data[feature_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"].to_numpy().astype(int)

    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype(np.float32)

    print(f"\n  {name}: {n:,} rows, {len(feature_cols)} features, "
          f"positives {y.mean()*100:.1f}%  "
          f"(train {int(tr.sum()):,} / val {int(va.sum()):,} / test {int(te.sum()):,})")
    print(f"    test positives: {int(y[te].sum()):,}")

    for model_name in ("RandomForest", "HistGradientBoosting", "LogisticRegression"):
        for seed in SEEDS:
            clf = build_model(model_name, seed)
            clf.fit(Xs[tr], y[tr])
            s_va = clf.predict_proba(Xs[va])[:, 1]
            s_te = clf.predict_proba(Xs[te])[:, 1]
            thr = select_threshold(y[va], s_va, criterion="f1")
            m = evaluate(y[te], s_te, thr)
            m.update(task=name, model=model_name, seed=seed,
                     n_features=len(feature_cols), note=note)
            rows.append(m)
        last = rows[-1]
        print(f"    {model_name:22s} F1={last['f1_illicit']:.4f} "
              f"P={last['precision']:.4f} R={last['recall']:.4f} "
              f"PR-AUC={last['pr_auc']:.4f} MCC={last['mcc']:.4f}")


def rule_baseline(data: pd.DataFrame, rows: list) -> None:
    """
    The strongest single rule the corpus columns can express, as a reference
    point for the trained models.

    A confirmed Whirlpool transaction has every output equal to a pool
    denomination, so its mean output value sits exactly on one. The corpus
    stores values as float32, so exact equality must be tested with a
    tolerance near that precision; 1e-7 relative is used, and the verdict is
    stable from 1e-7 through 1e-4. Roughly half the unconfirmed candidates
    also sit on a denomination and fail on other protocol conditions, which is
    what the rule cannot see and a model might.
    """
    pools = np.array([0.001, 0.01, 0.05, 0.5])
    per_out = (data["total_output_value"]
               / data["output_count"].replace(0, np.nan)).to_numpy(float)
    dist = np.array([float(np.min(np.abs(v - pools) / pools))
                     if np.isfinite(v) else 1.0 for v in per_out])
    data = data.assign(_dist=dist).sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    te = slice(int(n * 0.75), n)
    y = data["y"].to_numpy()[te]
    pred = (data["_dist"].to_numpy()[te] <= 1e-7).astype(float)
    m = evaluate(y, pred, 0.5)
    m.update(task="C_whirlpool_only", model="RuleBaseline", seed=0,
             n_features=1, note="per-output value exactly at a pool denomination")
    rows.append(m)
    print(f"    {'RuleBaseline':22s} F1={m['f1_illicit']:.4f} "
          f"P={m['precision']:.4f} R={m['recall']:.4f} MCC={m['mcc']:.4f}")


def main() -> None:
    ver = pd.read_csv(RESULTS / "e14b_all_verified.csv")
    ver = ver[ver["verdict"] == "ok"].copy()
    print(f"  verified transactions: {len(ver):,}")

    cand = ver[ver["group"].isin(["whirlpool_candidate", "wasabi2_candidate"])].copy()
    conf = cand[cand["confirmed_coinjoin"]]
    unconf = cand[~cand["confirmed_coinjoin"]]
    print(f"  confirmed CoinJoins  : {len(conf):,}")
    print(f"  shaped but unconfirmed: {len(unconf):,}")

    cols = ["txid", "timestamp"] + SHAPE_COLS + OTHER_COLS
    corpus = pd.read_parquet(RAW_AUTHOR, columns=cols)
    for c in SHAPE_COLS + OTHER_COLS:
        if corpus[c].dtype == bool:
            corpus[c] = corpus[c].astype(np.int8)
    corpus = corpus.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feats = corpus.set_index("txid")
    rng = np.random.default_rng(42)

    # -- Task A: confirmed against random corpus transactions ---------------
    conf_ids = set(conf["txid"])
    cand_ids = set(cand["txid"])
    pool = corpus[~corpus["txid"].isin(cand_ids)]
    neg_ids = rng.choice(pool["txid"].to_numpy(),
                         size=min(N_RANDOM_NEGATIVES, len(pool)), replace=False)

    a = pd.concat([
        feats.loc[list(conf_ids)].assign(y=1),
        feats.loc[list(neg_ids)].assign(y=0),
    ]).reset_index()

    # -- Task B: confirmed against unconfirmed, inside the candidate pool ---
    b = pd.concat([
        feats.loc[list(conf_ids)].assign(y=1),
        feats.loc[list(unconf["txid"])].assign(y=0),
    ]).reset_index()

    rows: list[dict] = []
    run_task("A_confirmed_vs_random", a, SHAPE_COLS + OTHER_COLS, rows,
             "trivial by construction; shape separates the sampling frame")
    run_task("B_confirmed_vs_lookalike", b, OTHER_COLS, rows,
             "shape columns withheld; both classes already satisfy them")
    run_task("B_lookalike_with_shape", b, SHAPE_COLS + OTHER_COLS, rows,
             "shape columns restored, to show how little they add here")

    # -- Task C: within each candidate group separately --------------------
    # Task B pools two groups with very different positive rates, 92.8 % for
    # Whirlpool-shaped and 42.8 % for Wasabi2-shaped. Those groups differ by
    # input count, which size, virtual size and weight encode even when the
    # count columns are withheld, so a model can score well on Task B by
    # recognising the group and predicting its base rate. Running inside each
    # group removes that shortcut: every transaction shares the same shape
    # regime and only genuine discrimination can separate the classes.
    for grp, label in (("whirlpool_candidate", "C_whirlpool_only"),
                       ("wasabi2_candidate", "C_wasabi2_only")):
        ids_pos = set(cand.loc[cand["confirmed_coinjoin"] & (cand["group"] == grp),
                               "txid"])
        ids_neg = set(cand.loc[~cand["confirmed_coinjoin"] & (cand["group"] == grp),
                               "txid"])
        if len(ids_pos) < 50 or len(ids_neg) < 50:
            print(f"\n  {label}: too few of one class "
                  f"({len(ids_pos)} positive, {len(ids_neg)} negative), skipped")
            continue
        c = pd.concat([
            feats.loc[list(ids_pos)].assign(y=1),
            feats.loc[list(ids_neg)].assign(y=0),
        ]).reset_index()
        run_task(label, c, OTHER_COLS, rows,
                 "single shape regime; the group-membership shortcut is removed")
        run_task(label + "_strict", c, STRICT_COLS, rows,
                 "fee, size and timing only; every arithmetic route to the "
                 "per-output value is withheld")
        if grp == "whirlpool_candidate":
            rule_baseline(c, rows)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e15_detection.csv", index=False)

    agg = (res.groupby(["task", "model"])
              .agg(n_features=("n_features", "first"),
                   f1=("f1_illicit", "mean"), f1_std=("f1_illicit", "std"),
                   precision=("precision", "mean"), recall=("recall", "mean"),
                   pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
                   mcc=("mcc", "mean"))
              .round(4).reset_index())
    print("\n" + "=" * 110)
    print(agg.to_string(index=False))
    agg.to_csv(RESULTS / "e15_aggregate.csv", index=False)

    (RESULTS / "e15_summary.json").write_text(json.dumps({
        "n_confirmed": int(len(conf)),
        "n_lookalike": int(len(unconf)),
        "n_random_negatives": int(len(neg_ids)),
        "label_source": "blockchain verification against Dumplings rules (E14b)",
        "split": "chronological 60/15/25 by timestamp",
        "aggregate": agg.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
