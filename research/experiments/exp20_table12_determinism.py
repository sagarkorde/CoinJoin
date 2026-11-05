"""
E20 - Why Table 12's Random Forest rows cannot be reproduced.

E15 reports detection within each candidate group and is the source of
Table 12. E16 bootstraps the same strict-regime result and E17 replicates the
same protocol independently. E16 and E17 agree with each other to four decimal
places and both disagree with E15 on every Random Forest row, while agreeing
with it exactly on Hist. Gradient Boosting, Logistic Regression and the rule
baseline.

The cause is in how E15 assembles its row set. Membership is collected into
Python `set` objects and the frame is then built with
`feats.loc[list(ids_pos)]`. Iteration order over a set of strings depends on
the interpreter's hash seed, which is randomised per process unless
PYTHONHASHSEED is fixed, so the pre-sort row order differs between runs. The
chronological split then sorts on `timestamp` with the pandas default, which is
not a stable sort, and 791 of the 4,137 Whirlpool candidates share a timestamp
with at least one other candidate. Rows tied on the sort key therefore land on
different sides of the 60/15/25 boundaries from one run to the next.

Deterministic learners absorb this: Hist. Gradient Boosting and Logistic
Regression reach the same figures regardless, which is why only the Random
Forest rows disagree.

This experiment does two things.

  --probe   reruns E15's own construction under several hash seeds and reports
            the spread, which demonstrates the defect rather than asserting it.
  --fix     recomputes every Table 12 figure under a construction that is
            deterministic: identifiers are taken in sorted order and the
            chronological sort is stable, so the split is a function of the
            data alone.

Everything else is E15's protocol unchanged: the same columns, the same
60/15/25 chronological split, the scaler fitted on training rows, the
threshold selected on validation and frozen, and the same learners.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RAW_AUTHOR, RESULTS            # noqa: E402
from metrics import evaluate, select_threshold    # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 43, 44, 45, 46]

SHAPE_COLS = ["input_count", "output_count", "input_output_ratio"]

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

MODELS = ("HistGradientBoosting", "RandomForest", "LogisticRegression")

SIZE_FAMILY = ["size", "vsize", "weight"]
FEE_TIMING = [c for c in STRICT_COLS if c not in SIZE_FAMILY]
REGIMES = {
    "strict": STRICT_COLS,
    "size_only": SIZE_FAMILY,
    "strict_no_size": FEE_TIMING,
    "fee_timing": ["fee", "fee_rate_sat_per_vbyte", "fee_rate_sat_per_byte",
                   "hour", "day_of_week"],
}


def build_model(name: str, seed: int):
    """Identical to E15's."""
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              l2_regularization=1.0,
                                              random_state=seed)
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ver = pd.read_csv(RESULTS / "e14b_all_verified.csv")
    ver = ver[ver["verdict"] == "ok"].copy()
    cand = ver[ver["group"].isin(["whirlpool_candidate",
                                  "wasabi2_candidate"])].copy()
    cols = ["txid", "timestamp"] + SHAPE_COLS + OTHER_COLS
    corpus = pd.read_parquet(RAW_AUTHOR, columns=list(dict.fromkeys(cols)))
    for c in SHAPE_COLS + OTHER_COLS:
        if corpus[c].dtype == bool:
            corpus[c] = corpus[c].astype(np.int8)
    corpus = corpus.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cand, corpus.set_index("txid")


def group_frame(cand: pd.DataFrame, feats: pd.DataFrame, grp: str,
                deterministic: bool) -> pd.DataFrame:
    """
    Assemble one candidate group.

    With `deterministic` the identifiers are taken in sorted order; otherwise
    they pass through a Python set first, which is what E15 does and what makes
    the row order depend on the interpreter's hash seed.
    """
    pos = cand.loc[cand["confirmed_coinjoin"] & (cand["group"] == grp), "txid"]
    neg = cand.loc[~cand["confirmed_coinjoin"] & (cand["group"] == grp), "txid"]
    if deterministic:
        ids_pos, ids_neg = sorted(pos.tolist()), sorted(neg.tolist())
    else:
        ids_pos, ids_neg = list(set(pos)), list(set(neg))
    return pd.concat([feats.loc[ids_pos].assign(y=1),
                      feats.loc[ids_neg].assign(y=0)]).reset_index()


def split(data: pd.DataFrame, deterministic: bool):
    """Chronological 60/15/25. A stable sort makes the split well defined."""
    kind = "mergesort" if deterministic else "quicksort"
    data = data.sort_values("timestamp", kind=kind).reset_index(drop=True)
    n = len(data)
    i_tr, i_va = int(n * 0.6), int(n * 0.75)
    tr = np.zeros(n, bool); tr[:i_tr] = True
    va = np.zeros(n, bool); va[i_tr:i_va] = True
    te = np.zeros(n, bool); te[i_va:] = True
    return data, tr, va, te


def run_task(name: str, data: pd.DataFrame, cols: list[str], rows: list,
             deterministic: bool, seeds: list[int], quiet: bool = False) -> None:
    data, tr, va, te = split(data, deterministic)
    X = np.nan_to_num(data[cols].to_numpy(dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"].to_numpy().astype(int)
    Xs = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)

    if not quiet:
        print("\n  %s: %d rows, %d features, test positives %d/%d"
              % (name, len(data), len(cols), int(y[te].sum()), int(te.sum())))

    for model_name in MODELS:
        per = []
        for seed in seeds:
            clf = build_model(model_name, seed)
            clf.fit(Xs[tr], y[tr])
            thr = select_threshold(y[va], clf.predict_proba(Xs[va])[:, 1],
                                   criterion="f1")
            m = evaluate(y[te], clf.predict_proba(Xs[te])[:, 1], thr)
            m.update(task=name, model=model_name, seed=seed,
                     n_features=len(cols))
            rows.append(m)
            per.append(m)
        if not quiet:
            print("    %-22s F1=%.4f P=%.4f R=%.4f MCC=%.4f (sd %.4f)"
                  % (model_name,
                     np.mean([r["f1_illicit"] for r in per]),
                     np.mean([r["precision"] for r in per]),
                     np.mean([r["recall"] for r in per]),
                     np.mean([r["mcc"] for r in per]),
                     np.std([r["mcc"] for r in per])))


def rule_baseline(data: pd.DataFrame, rows: list, deterministic: bool) -> None:
    """
    The strongest single rule the corpus columns express: a confirmed Whirlpool
    round has every output on a pool denomination, so its mean output value
    sits exactly on one. Values are stored in single precision, so equality is
    tested at a relative tolerance of 1e-7.
    """
    POOLS = [0.001, 0.01, 0.05, 0.5]
    data, tr, va, te = split(data, deterministic)
    y = data["y"].to_numpy().astype(int)
    avg = data["avg_output_value"].to_numpy(dtype=np.float64)
    hit = np.zeros(len(data), bool)
    for p in POOLS:
        hit |= np.isclose(avg, p, rtol=1e-7, atol=0.0)
    m = evaluate(y[te], hit[te].astype(float), 0.5)
    m.update(task="C_whirlpool_only", model="RuleBaseline", seed=-1,
             n_features=1)
    rows.append(m)
    print("    %-22s F1=%.4f P=%.4f R=%.4f MCC=%.4f"
          % ("RuleBaseline", m["f1_illicit"], m["precision"], m["recall"],
             m["mcc"]))


def tasks(cand, feats, rows, deterministic, seeds, quiet=False):
    for grp, label in (("whirlpool_candidate", "C_whirlpool_only"),
                       ("wasabi2_candidate", "C_wasabi2_only")):
        g = group_frame(cand, feats, grp, deterministic)
        run_task(label, g, OTHER_COLS, rows, deterministic, seeds, quiet)
        run_task(label + "_strict", g, STRICT_COLS, rows, deterministic,
                 seeds, quiet)
        if grp == "whirlpool_candidate" and not quiet:
            rule_baseline(g, rows, deterministic)


def probe() -> None:
    """Rerun E15's construction in child processes with different hash seeds."""
    here = Path(__file__).resolve()
    out = []
    for hs in ("0", "1", "2", "3", "4"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, str(here), "--child"],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit("child failed at PYTHONHASHSEED=%s" % hs)
        out.append(json.loads(r.stdout.strip().splitlines()[-1]))
        print("  PYTHONHASHSEED=%s  strict RF MCC=%.4f  F1=%.4f  "
              "test positives %d/%d"
              % (hs, out[-1]["mcc"], out[-1]["f1"], out[-1]["pos"],
                 out[-1]["n"]))

    mccs = [o["mcc"] for o in out]
    print("\n  E15 construction, Random Forest, strict regime:")
    print("    distinct MCC values over 5 hash seeds: %d" % len(set(
        round(m, 4) for m in mccs)))
    print("    range %.4f to %.4f, spread %.4f"
          % (min(mccs), max(mccs), max(mccs) - min(mccs)))
    print("    Table 12 reports 0.8136; e15_aggregate.csv holds 0.8754;")
    print("    E16 and E17 both give 0.8595.")
    (RESULTS / "e20_hashseed_probe.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("\n  written to e20_hashseed_probe.json")


def child() -> None:
    """One probe replicate: E15's construction, strict regime, mean over seeds."""
    cand, feats = load()
    g = group_frame(cand, feats, "whirlpool_candidate", deterministic=False)
    rows: list = []
    run_task("probe", g, STRICT_COLS, rows, False, SEEDS, quiet=True)
    rf = [r for r in rows if r["model"] == "RandomForest"]
    d, tr, va, te = split(g, False)
    y = d["y"].to_numpy().astype(int)
    print(json.dumps({"mcc": float(np.mean([r["mcc"] for r in rf])),
                      "f1": float(np.mean([r["f1_illicit"] for r in rf])),
                      "n": int(te.sum()), "pos": int(y[te].sum())}))


def fix(seeds: list[int], tag: str) -> None:
    cand, feats = load()
    rows: list = []
    tasks(cand, feats, rows, deterministic=True, seeds=seeds)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / ("e20_%s.csv" % tag), index=False)
    agg = (df.groupby(["task", "model"])
             .agg(n_features=("n_features", "first"),
                  f1=("f1_illicit", "mean"), precision=("precision", "mean"),
                  recall=("recall", "mean"), mcc=("mcc", "mean"),
                  mcc_sd=("mcc", "std"))
             .round(4).reset_index())
    agg.to_csv(RESULTS / ("e20_%s_aggregate.csv" % tag), index=False)
    print("\n" + agg.to_string(index=False))
    print("\n  written to e20_%s_aggregate.csv" % tag)


def regimes(seeds: list[int]) -> None:
    """E17's regime decomposition under the deterministic construction."""
    cand, feats = load()
    rows: list = []
    for grp, label in (("whirlpool_candidate", "whirlpool"),
                       ("wasabi2_candidate", "wasabi2")):
        g = group_frame(cand, feats, grp, deterministic=True)
        for rname, rcols in REGIMES.items():
            run_task("%s_%s" % (label, rname), g, rcols, rows,
                     deterministic=True, seeds=seeds)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e20_regimes.csv", index=False)
    agg = (df.groupby(["task", "model"])
             .agg(n_features=("n_features", "first"),
                  f1=("f1_illicit", "mean"), precision=("precision", "mean"),
                  recall=("recall", "mean"), mcc=("mcc", "mean"),
                  mcc_sd=("mcc", "std"))
             .round(4).reset_index())
    agg.to_csv(RESULTS / "e20_regimes_aggregate.csv", index=False)
    print("\n" + agg.to_string(index=False))
    piv = agg.pivot_table(index="task", columns="model", values="mcc")
    print("\nMCC by regime:\n" + piv.round(4).to_string())
    print("\n  written to e20_regimes_aggregate.csv")


def fingerprint(_: list[int]) -> None:
    """
    Why the size family carries the Whirlpool result.

    Whirlpool rounds are built to a fixed template, so within one input and
    output shape the serialised size of a confirmed round is almost constant.
    A look-alike that sits on the same pool denomination with the same shape
    but fails another protocol condition is not built to that template, so its
    size varies. The table reports, per shape, the mean and standard deviation
    of virtual size for each class and the share of look-alikes falling inside
    the 5th to 95th percentile band of the confirmed rounds.
    """
    ver = pd.read_csv(RESULTS / "e14b_all_verified.csv")
    ver = ver[(ver["verdict"] == "ok")
              & (ver["group"] == "whirlpool_candidate")].copy()
    corpus = pd.read_parquet(RAW_AUTHOR,
                             columns=["txid", "vsize", "input_count",
                                      "output_count"])
    d = ver.merge(corpus, on="txid", how="inner")
    d["shape"] = (d["input_count"].astype(int).astype(str) + "x"
                  + d["output_count"].astype(int).astype(str))
    conf = d[d["confirmed_coinjoin"]]
    look = d[~d["confirmed_coinjoin"]]

    print("  confirmed %d, look-alikes %d" % (len(conf), len(look)))
    print("  distinct vsize values: confirmed %d, look-alikes %d"
          % (conf["vsize"].nunique(), look["vsize"].nunique()))

    rows = []
    for shape in (conf["shape"].value_counts().index[:4]):
        c = conf[conf["shape"] == shape]["vsize"].to_numpy(dtype=float)
        l = look[look["shape"] == shape]["vsize"].to_numpy(dtype=float)
        lo, hi = np.percentile(c, [5, 95])
        inside = float(((l >= lo) & (l <= hi)).mean() * 100) if len(l) else float("nan")
        rows.append({"shape": shape, "n_confirmed": len(c), "n_lookalike": len(l),
                     "confirmed_mean": round(float(c.mean()), 1),
                     "confirmed_sd": round(float(c.std(ddof=1)), 1),
                     "confirmed_cv": round(float(c.std(ddof=1) / c.mean()), 4),
                     "confirmed_distinct": int(len(set(c))),
                     "lookalike_mean": round(float(l.mean()), 1) if len(l) else None,
                     "lookalike_sd": round(float(l.std(ddof=1)), 1) if len(l) > 1 else None,
                     "band_lo": round(float(lo), 1), "band_hi": round(float(hi), 1),
                     "lookalike_inside_band_pct": round(inside, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "e20_vsize_fingerprint.csv", index=False)
    summary = {"n_confirmed": int(len(conf)), "n_lookalike": int(len(look)),
               "confirmed_distinct_vsize": int(conf["vsize"].nunique()),
               "lookalike_distinct_vsize": int(look["vsize"].nunique()),
               "confirmed_mean_input_count": round(float(conf["input_count"].mean()), 2),
               "lookalike_mean_input_count": round(float(look["input_count"].mean()), 2),
               "per_shape": rows}
    (RESULTS / "e20_vsize_fingerprint.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print()
    print(out.to_string(index=False))
    print()
    print("  mean input count: confirmed %.2f, look-alikes %.2f"
          % (conf["input_count"].mean(), look["input_count"].mean()))
    print("  written to e20_vsize_fingerprint.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", action="store_true")
    ap.add_argument("--fingerprint", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--tag", default="table12")
    a = ap.parse_args()
    if a.child:
        child()
    elif a.regimes:
        regimes(a.seeds)
    elif a.fingerprint:
        fingerprint(a.seeds)
    elif a.probe:
        probe()
    elif a.fix:
        fix(a.seeds, a.tag)
    else:
        probe()
        print()
        fix(a.seeds, a.tag)


if __name__ == "__main__":
    main()
