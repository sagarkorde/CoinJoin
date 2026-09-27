"""
E14 - External CoinJoin ground truth.

Every previous experiment measured the screening heuristic against a label the
same heuristic family produced. This one measures it against transactions
confirmed as CoinJoin outputs by protocol rules applied to real blockchain
data, which is the validation the study has lacked.

Source of the rules
-------------------
The detection conditions are those implemented by Dumplings
(github.com/nopara73/Dumplings), the reference tool used in the CoinJoin
measurement literature to extract Wasabi, Whirlpool and JoinMarket
transactions from mainnet. They are reproduced here rather than invented:

  Whirlpool (Samourai)
    native SegWit only; 5 to 10 inputs and outputs with equal counts;
    every output exactly equal; that value one of the pools
    {0.001, 0.01, 0.05, 0.5} BTC; at least one input exactly pool-sized;
    every other input within 0.0011 BTC above the pool size.

  Wasabi 2.x (WabiSabi)
    inputs exclusively P2WPKH or Taproot; at least 50 inputs;
    input and output value sequences sorted descending;
    more than 80 % of outputs drawn from the WabiSabi denomination set.

Method
------
The corpus carries no per-output values, so a transaction cannot be confirmed
locally. Candidates are therefore selected by the necessary conditions that
the corpus columns can express, and each candidate is then fetched from a
public block explorer and tested against the full rule using real per-output
values. Confirmation is by blockchain data, not by the corpus.

A random sample of transactions the corpus labels `is_coinjoin_like` is
verified the same way. That sample yields the quantity the study has been
unable to state: the precision of the screening label against confirmed
CoinJoin transactions.

Results are cached per transaction, so the run is resumable and a rerun costs
no further requests.

Outputs
-------
  results/e14_verified/<txid>.json     per-transaction explorer records
  results/e14_ground_truth.csv         verification verdict per transaction
  results/e14_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR

API = "https://blockstream.info/api/tx/{}"
CACHE = RESULTS / "e14_verified"
CACHE.mkdir(parents=True, exist_ok=True)

SAT = 100_000_000
POOLS_SAT = [int(round(p * SAT)) for p in (0.001, 0.01, 0.05, 0.5)]
POOL_TOLERANCE_SAT = int(round(0.0011 * SAT))
NATIVE_SEGWIT = {"v0_p2wpkh"}
W2_INPUT_TYPES = {"v0_p2wpkh", "v1_p2tr"}

REQUEST_DELAY = 0.34          # polite pacing against a public explorer
MAX_RETRIES = 3


def wasabi2_denominations() -> set[int]:
    """
    The WabiSabi denomination set: powers of 2, 3 and 10 and the standard
    combinations, bounded as Dumplings bounds them.
    """
    vals: set[int] = set()
    for base in (2, 3, 10):
        v = 1
        while v <= 134_375_000_000:
            vals.add(v)
            vals.add(v * 2)
            vals.add(v * 5)
            v *= base
    return {v for v in vals if 5_000 <= v <= 134_375_000_000}


W2_DENOMS = wasabi2_denominations()


def fetch(txid: str) -> dict | None:
    """Fetch one transaction, using the on-disk cache when present."""
    path = CACHE / f"{txid}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink()

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                API.format(txid),
                headers={"User-Agent": "coinjoin-research/1.0 (academic replication)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            path.write_text(json.dumps(d), encoding="utf-8")
            time.sleep(REQUEST_DELAY)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2.0 * (attempt + 1))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None


def classify(tx: dict) -> dict:
    """Apply the Dumplings rules to a fetched transaction."""
    try:
        outs = [int(o["value"]) for o in tx["vout"]]
        ins = [int(i["prevout"]["value"]) for i in tx["vin"]]
        otypes = {o.get("scriptpubkey_type", "") for o in tx["vout"]}
        itypes = {i["prevout"].get("scriptpubkey_type", "") for i in tx["vin"]}
    except (KeyError, TypeError):
        return {"verdict": "unparseable"}

    n_in, n_out = len(ins), len(outs)
    res = {"n_in": n_in, "n_out": n_out,
           "out_types": ";".join(sorted(otypes)),
           "in_types": ";".join(sorted(itypes))}

    # -- Whirlpool ---------------------------------------------------------
    native = otypes <= NATIVE_SEGWIT and itypes <= NATIVE_SEGWIT
    whirlpool = False
    pool = None
    if (native and 5 <= n_in <= 10 and 5 <= n_out <= 10 and n_in == n_out
            and len(set(outs)) == 1):
        pool = outs[0]
        if pool in POOLS_SAT:
            exact = sum(1 for v in ins if v == pool)
            others_ok = all(pool <= v <= pool + POOL_TOLERANCE_SAT
                            for v in ins if v != pool)
            whirlpool = exact >= 1 and others_ok

    # -- Wasabi 2.x --------------------------------------------------------
    wasabi2 = False
    w2_frac = 0.0
    if itypes <= W2_INPUT_TYPES and n_in >= 50:
        desc_in = ins == sorted(ins, reverse=True)
        desc_out = outs == sorted(outs, reverse=True)
        w2_frac = sum(1 for v in outs if v in W2_DENOMS) / max(n_out, 1)
        wasabi2 = desc_in and desc_out and w2_frac > 0.8

    # -- generic equal-output CoinJoin (weaker, reported separately) -------
    counts = pd.Series(outs).value_counts()
    top_val = int(counts.index[0])
    top_n = int(counts.iloc[0])
    generic = (top_n >= 3 and top_n >= n_out * 0.4 and n_in >= 3
               and len(set(outs)) >= 2)

    res.update({
        "whirlpool": bool(whirlpool),
        "whirlpool_pool_sat": int(pool) if whirlpool else 0,
        "wasabi2": bool(wasabi2),
        "wasabi2_denom_fraction": round(float(w2_frac), 4),
        "generic_equal_output": bool(generic),
        "max_equal_output_count": top_n,
        "max_equal_output_value": top_val,
        "confirmed_coinjoin": bool(whirlpool or wasabi2),
        "verdict": "ok",
    })
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-whirlpool", type=int, default=250)
    ap.add_argument("--n-wasabi2", type=int, default=120)
    ap.add_argument("--n-label-sample", type=int, default=400)
    ap.add_argument("--n-negative", type=int, default=150)
    args = ap.parse_args()

    cols = ["txid", "input_count", "output_count", "total_output_value",
            "avg_output_value", "is_coinjoin_like", "timestamp", "block_height"]
    df = pd.read_parquet(RAW_AUTHOR, columns=cols)
    print(f"  corpus {len(df):,} transactions")

    per_out = df["total_output_value"] / df["output_count"].replace(0, np.nan)
    eq_shape = (df["input_count"].between(5, 10)
                & df["output_count"].between(5, 10)
                & (df["input_count"] == df["output_count"]))
    near_pool = pd.Series(False, index=df.index)
    for p in (0.001, 0.01, 0.05, 0.5):
        near_pool |= (per_out - p).abs() < p * 0.02
    whirl_cand = eq_shape & near_pool
    w2_cand = (df["input_count"] >= 50) & (df["output_count"] >= 50)
    label_pos = df["is_coinjoin_like"].astype(bool)

    print(f"  Whirlpool-shaped candidates : {int(whirl_cand.sum()):,}")
    print(f"  Wasabi2-shaped candidates   : {int(w2_cand.sum()):,}")
    print(f"  labelled is_coinjoin_like   : {int(label_pos.sum()):,}")

    rng = np.random.default_rng(42)

    def pick(mask, n, exclude=frozenset()):
        ix = df.index[mask].to_numpy()
        ix = np.array([i for i in ix if df.at[i, "txid"] not in exclude])
        if len(ix) == 0:
            return np.array([], dtype=int)
        return rng.choice(ix, size=min(n, len(ix)), replace=False)

    groups = {
        "whirlpool_candidate": pick(whirl_cand, args.n_whirlpool),
        "wasabi2_candidate": pick(w2_cand, args.n_wasabi2),
        "label_positive_random": pick(label_pos & ~whirl_cand & ~w2_cand,
                                      args.n_label_sample),
        "label_negative_random": pick(~label_pos, args.n_negative),
    }

    rows, seen = [], set()
    total = sum(len(v) for v in groups.values())
    done = 0
    for gname, idx in groups.items():
        for i in idx:
            txid = str(df.at[i, "txid"])
            done += 1
            if txid in seen:
                continue
            seen.add(txid)
            tx = fetch(txid)
            if done % 25 == 0:
                print(f"    verified {done}/{total}", end="\r")
            if tx is None:
                rows.append({"txid": txid, "group": gname, "verdict": "not_found",
                             "corpus_label": bool(df.at[i, "is_coinjoin_like"])})
                continue
            r = classify(tx)
            r.update(txid=txid, group=gname,
                     corpus_label=bool(df.at[i, "is_coinjoin_like"]),
                     corpus_in=int(df.at[i, "input_count"]),
                     corpus_out=int(df.at[i, "output_count"]))
            rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e14_ground_truth.csv", index=False)
    ok = res[res["verdict"] == "ok"]
    print(f"\n  verified {len(ok):,} of {len(res):,} sampled "
          f"({int((res['verdict'] != 'ok').sum())} unavailable)")

    summary: dict = {
        "n_sampled": int(len(res)),
        "n_verified": int(len(ok)),
        "rules_source": "Dumplings (github.com/nopara73/Dumplings)",
        "explorer": "blockstream.info",
        "corpus_candidates": {
            "whirlpool_shaped": int(whirl_cand.sum()),
            "wasabi2_shaped": int(w2_cand.sum()),
            "labelled_positive": int(label_pos.sum()),
        },
    }

    by_group = {}
    for g, sub in ok.groupby("group"):
        by_group[g] = {
            "n": int(len(sub)),
            "confirmed_coinjoin": int(sub["confirmed_coinjoin"].sum()),
            "confirmed_rate": round(float(sub["confirmed_coinjoin"].mean()), 4),
            "whirlpool": int(sub["whirlpool"].sum()),
            "wasabi2": int(sub["wasabi2"].sum()),
            "generic_equal_output": int(sub["generic_equal_output"].sum()),
        }
    summary["by_group"] = by_group

    # The headline quantity: precision of the corpus label against confirmed
    # CoinJoin transactions, estimated on the random label-positive sample.
    lp = ok[ok["group"] == "label_positive_random"]
    if len(lp):
        summary["screening_label_precision"] = {
            "n": int(len(lp)),
            "confirmed_coinjoin": int(lp["confirmed_coinjoin"].sum()),
            "precision_strict": round(float(lp["confirmed_coinjoin"].mean()), 4),
            "generic_equal_output_rate": round(
                float(lp["generic_equal_output"].mean()), 4),
            "note": ("Random sample of transactions the corpus labels "
                     "is_coinjoin_like, excluding protocol-shaped candidates, "
                     "verified against real per-output values."),
        }

    # Stratified estimate over the whole label-positive population.
    # Protocol-shaped transactions satisfy the label rule by construction
    # (Whirlpool is 5-in/5-out, Wasabi2 is >=50/>=50), so the label-positive
    # set partitions into the two shaped strata plus the remainder, and each
    # stratum's confirmed rate is estimated from its own verified sample.
    n_label = int(label_pos.sum())
    n_whirl = int((whirl_cand & label_pos).sum())
    n_w2 = int((w2_cand & label_pos).sum())
    n_rest = n_label - n_whirl - n_w2

    def rate(group):
        sub = ok[ok["group"] == group]
        return (float(sub["confirmed_coinjoin"].mean()), int(len(sub))) if len(sub) else (0.0, 0)

    r_whirl, nv_whirl = rate("whirlpool_candidate")
    r_w2, nv_w2 = rate("wasabi2_candidate")
    r_rest, nv_rest = rate("label_positive_random")

    est_confirmed = n_whirl * r_whirl + n_w2 * r_w2 + n_rest * r_rest
    summary["stratified_label_precision"] = {
        "n_label_positive": n_label,
        "strata": {
            "whirlpool_shaped": {"n": n_whirl, "confirmed_rate": round(r_whirl, 4),
                                 "n_verified": nv_whirl},
            "wasabi2_shaped": {"n": n_w2, "confirmed_rate": round(r_w2, 4),
                               "n_verified": nv_w2},
            "remainder": {"n": n_rest, "confirmed_rate": round(r_rest, 4),
                          "n_verified": nv_rest},
        },
        "estimated_confirmed_coinjoins": int(round(est_confirmed)),
        "estimated_precision": round(est_confirmed / max(n_label, 1), 4),
        "note": ("Precision of the corpus screening label against CoinJoin "
                 "transactions confirmed from blockchain data, estimated "
                 "stratum-wise over the whole label-positive population."),
    }

    # Recall of the label over the confirmed transactions found.
    conf = ok[ok["confirmed_coinjoin"]]
    if len(conf):
        summary["label_recall_on_confirmed"] = {
            "n_confirmed_verified": int(len(conf)),
            "also_labelled_positive": int(conf["corpus_label"].sum()),
            "recall": round(float(conf["corpus_label"].mean()), 4),
            "note": ("Of transactions confirmed as CoinJoin outputs, the "
                     "fraction the corpus label also flags."),
        }

    (RESULTS / "e14_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
