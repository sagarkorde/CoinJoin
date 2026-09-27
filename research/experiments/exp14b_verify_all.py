"""
E14b - Verify every protocol-shaped candidate in the corpus.

E14 verified a sample. This verifies the whole candidate population: all
4,137 Whirlpool-shaped and 1,522 Wasabi2-shaped transactions, plus the random
label-positive and label-negative samples that estimate the remaining stratum.

Fetching is concurrent because a sequential pass over six thousand
transactions against a public explorer takes hours. Concurrency is held low
and each worker paces itself, since the explorer is a free service. Every
response is cached on disk, so a rerun costs no requests and an interrupted
run resumes where it stopped.

Classification reuses the Dumplings rules implemented in E14 rather than
restating them, so the two experiments cannot drift apart.

Outputs
-------
  results/e14b_all_verified.csv
  results/e14b_summary.json
  results/e14_verified/<txid>.json   (shared cache)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from config import RESULTS, RAW_AUTHOR

# Reuse the rule implementation and cache from E14.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "exp14", HERE / "exp14_external_ground_truth.py")
_e14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e14)

CACHE = _e14.CACHE
classify = _e14.classify

# Fetching goes through ExplorerPool, which spreads requests across several
# public explorers, rests an endpoint that returns 429 or 5xx while the others
# continue, and paces the aggregate against a global rate ceiling. An earlier
# pass used ten workers against a single explorer, was rate-limited, and then
# made almost no progress while still issuing requests.
from explorer import ExplorerPool

WORKERS = 4
_print_lock = threading.Lock()
POOL = ExplorerPool(CACHE, target_rps=2.5)


def fetch_one(txid: str) -> tuple[str, dict | None, bool]:
    """Return (txid, record, came_from_cache)."""
    before = POOL.stats["cache_hits"]
    rec = POOL.get(txid)
    return txid, rec, POOL.stats["cache_hits"] > before


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--n-label-sample", type=int, default=400)
    ap.add_argument("--n-negative", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total transactions, for a quick check")
    args = ap.parse_args()

    cols = ["txid", "input_count", "output_count", "total_output_value",
            "is_coinjoin_like", "timestamp", "block_height"]
    df = pd.read_parquet(RAW_AUTHOR, columns=cols)
    print(f"  corpus {len(df):,} transactions")

    per_out = df["total_output_value"] / df["output_count"].replace(0, np.nan)
    whirl = (df["input_count"].between(5, 10)
             & df["output_count"].between(5, 10)
             & (df["input_count"] == df["output_count"]))
    near = pd.Series(False, index=df.index)
    for p in (0.001, 0.01, 0.05, 0.5):
        near |= (per_out - p).abs() < p * 0.02
    whirl &= near
    w2 = (df["input_count"] >= 50) & (df["output_count"] >= 50)
    label_pos = df["is_coinjoin_like"].astype(bool)

    rng = np.random.default_rng(42)

    def pick(mask, n):
        ix = df.index[mask].to_numpy()
        return ix if len(ix) <= n else rng.choice(ix, size=n, replace=False)

    groups = {
        "whirlpool_candidate": df.index[whirl].to_numpy(),
        "wasabi2_candidate": df.index[w2].to_numpy(),
        "label_positive_random": pick(label_pos & ~whirl & ~w2,
                                      args.n_label_sample),
        "label_negative_random": pick(~label_pos, args.n_negative),
    }
    for g, ix in groups.items():
        print(f"    {g:24s} {len(ix):>6,}")

    # Build the work list, de-duplicated, keeping the first group a txid hits.
    work: dict[str, tuple[int, str]] = {}
    for g, ix in groups.items():
        for i in ix:
            t = str(df.at[i, "txid"])
            work.setdefault(t, (i, g))
    items = list(work.items())
    if args.limit:
        items = items[:args.limit]

    cached_now = sum(1 for t, _ in items if (CACHE / f"{t}.json").exists())
    print(f"\n  {len(items):,} transactions to verify "
          f"({cached_now:,} already cached, {len(items) - cached_now:,} to fetch)")

    t0 = time.time()
    records: dict[str, dict | None] = {}
    done = fetched = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, t): t for t, _ in items}
        for fut in as_completed(futures):
            txid, rec, from_cache = fut.result()
            records[txid] = rec
            done += 1
            if not from_cache:
                fetched += 1
            if done % 100 == 0:
                el = time.time() - t0
                rate = fetched / el if el > 0 else 0
                left = len(items) - done
                eta = left / rate / 60 if rate > 0 else 0
                with _print_lock:
                    print(f"    {done:,}/{len(items):,}  "
                          f"fetched {fetched:,}  {rate:.1f} req/s  "
                          f"eta {eta:.0f} min", end="\r")

    print(f"\n  fetched {fetched:,} in {(time.time()-t0)/60:.1f} min")

    rows = []
    for txid, (i, g) in work.items():
        if args.limit and txid not in records:
            continue
        rec = records.get(txid)
        base = {"txid": txid, "group": g,
                "corpus_label": bool(df.at[i, "is_coinjoin_like"]),
                "corpus_in": int(df.at[i, "input_count"]),
                "corpus_out": int(df.at[i, "output_count"]),
                "timestamp": str(df.at[i, "timestamp"]),
                "block_height": int(df.at[i, "block_height"])}
        if rec is None:
            base["verdict"] = "not_found"
        else:
            base.update(classify(rec))
        rows.append(base)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "e14b_all_verified.csv", index=False)
    ok = res[res["verdict"] == "ok"]
    print(f"  classified {len(ok):,} of {len(res):,} "
          f"({int((res['verdict'] != 'ok').sum())} unavailable)")

    summary: dict = {
        "n_candidates_total": int(len(res)),
        "n_verified": int(len(ok)),
        "rules_source": "Dumplings (github.com/nopara73/Dumplings)",
        "explorers": POOL.endpoints,
        "explorer_stats": POOL.summary(),
    }
    by_group = {}
    for g, sub in ok.groupby("group"):
        by_group[g] = {
            "n": int(len(sub)),
            "confirmed_coinjoin": int(sub["confirmed_coinjoin"].sum()),
            "confirmed_rate": round(float(sub["confirmed_coinjoin"].mean()), 4),
            "whirlpool": int(sub["whirlpool"].sum()),
            "wasabi2": int(sub["wasabi2"].sum()),
        }
    summary["by_group"] = by_group

    # Whirlpool pool breakdown, from confirmed transactions only.
    wp = ok[ok["whirlpool"]]
    if len(wp):
        pools = (wp["whirlpool_pool_sat"] / 1e8).round(3).value_counts().sort_index()
        summary["confirmed_whirlpool_pools_btc"] = {
            str(k): int(v) for k, v in pools.items()}
        summary["n_confirmed_whirlpool"] = int(len(wp))

    # Stratified precision of the corpus screening label.
    n_label = int(label_pos.sum())
    n_wh = int((whirl & label_pos).sum())
    n_w2 = int((w2 & label_pos).sum())
    n_rest = n_label - n_wh - n_w2

    def rate_of(g):
        sub = ok[ok["group"] == g]
        return (float(sub["confirmed_coinjoin"].mean()), int(len(sub))) if len(sub) else (0.0, 0)

    r_wh, nv_wh = rate_of("whirlpool_candidate")
    r_w2, nv_w2 = rate_of("wasabi2_candidate")
    r_rest, nv_rest = rate_of("label_positive_random")
    est = n_wh * r_wh + n_w2 * r_w2 + n_rest * r_rest

    # Wilson interval for the remainder stratum, which dominates the count.
    from math import sqrt
    if nv_rest:
        p_hat = r_rest
        z = 1.96
        den = 1 + z * z / nv_rest
        centre = (p_hat + z * z / (2 * nv_rest)) / den
        half = z * sqrt(p_hat * (1 - p_hat) / nv_rest
                        + z * z / (4 * nv_rest ** 2)) / den
        rest_ci = (max(0.0, centre - half), min(1.0, centre + half))
    else:
        rest_ci = (0.0, 1.0)

    summary["screening_label_precision"] = {
        "n_label_positive": n_label,
        "strata": {
            "whirlpool_shaped": {"n": n_wh, "confirmed_rate": round(r_wh, 4),
                                 "n_verified": nv_wh},
            "wasabi2_shaped": {"n": n_w2, "confirmed_rate": round(r_w2, 4),
                               "n_verified": nv_w2},
            "remainder": {"n": n_rest, "confirmed_rate": round(r_rest, 4),
                          "n_verified": nv_rest,
                          "wilson_95ci": [round(rest_ci[0], 5), round(rest_ci[1], 5)]},
        },
        "estimated_confirmed": int(round(est)),
        "estimated_precision": round(est / max(n_label, 1), 5),
        "upper_bound_precision_95": round(
            (n_wh * r_wh + n_w2 * r_w2 + n_rest * rest_ci[1]) / max(n_label, 1), 5),
    }

    (RESULTS / "e14b_summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
