"""
04_baseline_chamfer.py  |  MixTrace Pipeline — Step 4: 1-D Chamfer Distance Baseline
======================================================================================
Implements and evaluates the one-sided Chamfer distance as a baseline for
linking CoinJoin Spending Transactions (CSTs) from the same user.

Mathematical Definition
-----------------------
Given two sets of 1-D points U (target CST timestamps) and V (candidate CST):

    d_C(U, V) = (1 / |U|) * Σ_{u_i ∈ U} min_{v_j ∈ V} |u_i - v_j|

The metric is asymmetric and robust to unequal set sizes — outlier points in V
that are far from U do not inflate the score as long as close matches exist.

Evaluation Protocol
-------------------
  1. Extract CoinJoin-like transactions from the Author Dataset.
  2. Group them into temporal windows (1-hour blocks) as proxy CST clusters.
  3. For each pair of windows, compute d_C and predict "same user" if distance < τ.
  4. Compare prediction against ground-truth same-window label.
  5. Sweep τ and report the threshold that maximises F1-score (baseline).

Results are saved for comparison with the Multi-Dimensional Chamfer (Step 11).

Run
---
    python 04_baseline_chamfer.py

Reads   : data/processed/author_raw.parquet
Outputs : results/04_chamfer_1d_results.json
          results/04_chamfer_1d_threshold_sweep.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"

# Maximum number of window-pairs to evaluate (limits O(n²) computation)
MAX_PAIRS = 5_000
RANDOM_SEED = 42


# ── Core Algorithm ────────────────────────────────────────────────────────────

def chamfer_1d(U: np.ndarray, V: np.ndarray) -> float:
    """
    One-sided Chamfer distance from point cloud U to point cloud V.

    Parameters
    ----------
    U : shape [n] — target timestamps (normalised seconds)
    V : shape [m] — candidate timestamps

    Returns
    -------
    float — mean minimum distance from each point in U to its nearest in V
    """
    if len(U) == 0 or len(V) == 0:
        return float("inf")
    U_col = U.reshape(-1, 1)
    V_col = V.reshape(-1, 1)
    dist_matrix = cdist(U_col, V_col, metric="euclidean")   # [n, m]
    return float(dist_matrix.min(axis=1).mean())


# ── Data Preparation ──────────────────────────────────────────────────────────

def extract_cst_windows(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Group CoinJoin-like transactions into 1-hour temporal windows.

    Each window becomes a 'pseudo-CST cluster' whose timestamp distribution
    serves as the 1-D point cloud for Chamfer distance computation.

    Returns
    -------
    dict mapping window_key → normalised timestamp array
    """
    cj = df[df["is_coinjoin_like"]].copy()

    # Unix timestamp in seconds
    cj["ts_unix"] = cj["timestamp"].astype("int64") // 10**9

    # 1-hour bucket key
    cj["window"] = (cj["block_height"] // 6).astype(str) + "_" + \
                   cj["hour"].astype(str) + "_" + cj["day_of_week"].astype(str)

    windows = {}
    for key, grp in cj.groupby("window"):
        ts = grp["ts_unix"].values.astype(np.float64)
        if len(ts) >= 3:
            windows[key] = ts

    return windows


def normalise_timestamps(windows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Min-max normalise all timestamps to [0, 1] using the global range."""
    all_ts = np.concatenate(list(windows.values()))
    ts_min, ts_max = all_ts.min(), all_ts.max()
    span = max(ts_max - ts_min, 1.0)
    return {k: (v - ts_min) / span for k, v in windows.items()}


# ── Evaluation ────────────────────────────────────────────────────────────────

def build_pairs(
    windows: dict[str, np.ndarray],
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[int]]:
    """
    Sample window pairs, compute Chamfer distance, and assign ground-truth labels.

    Label = 1 if both windows belong to the same 24-hour day (same-user proxy),
            0 otherwise.
    """
    keys = list(windows.keys())
    if len(keys) < 2:
        raise ValueError("Not enough windows to form pairs.")

    # Sample random pairs
    idx_pairs = []
    seen: set[tuple[int, int]] = set()
    while len(idx_pairs) < max_pairs:
        i, j = rng.integers(0, len(keys), size=2)
        if i != j and (i, j) not in seen and (j, i) not in seen:
            idx_pairs.append((i, j))
            seen.add((i, j))

    distances, labels = [], []
    for i, j in tqdm(idx_pairs, desc="  Computing Chamfer distances"):
        dist = chamfer_1d(windows[keys[i]], windows[keys[j]])
        distances.append(dist)
        # Same day-of-week bucket → treat as same-user proxy
        day_i = keys[i].split("_")[2]
        day_j = keys[j].split("_")[2]
        labels.append(1 if day_i == day_j else 0)

    return distances, labels


def threshold_sweep(
    distances: list[float],
    labels: list[int],
    n_thresholds: int = 200,
) -> tuple[float, pd.DataFrame]:
    """
    Sweep distance thresholds and compute F1 at each.

    A pair is predicted 'same user' when d_C < τ.

    Returns
    -------
    best_threshold : float
    sweep_df       : DataFrame with columns [threshold, precision, recall, f1]
    """
    dists  = np.array(distances)
    labels = np.array(labels)
    thresholds = np.linspace(dists.min(), np.percentile(dists, 95), n_thresholds)

    rows = []
    best_f1, best_tau = 0.0, thresholds[0]
    for tau in thresholds:
        preds = (dists < tau).astype(int)
        if preds.sum() == 0:
            continue
        p  = precision_score(labels, preds, zero_division=0)
        r  = recall_score(labels,    preds, zero_division=0)
        f1 = f1_score(labels,        preds, zero_division=0)
        rows.append({"threshold": tau, "precision": p, "recall": r, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return best_tau, pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 4: 1-D Chamfer Distance Baseline")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("\n  Loading Author Dataset (sample for speed) …")
    author = pd.read_parquet(DATA_PROC / "author_raw.parquet")

    # Use a representative 20% sample to keep pairwise computation tractable
    author = author.sample(frac=0.20, random_state=RANDOM_SEED)
    print(f"  Sample size: {len(author):,} transactions")

    print("\n  Extracting CoinJoin temporal windows …")
    windows = extract_cst_windows(author)
    windows = normalise_timestamps(windows)
    print(f"  Windows (≥3 txs): {len(windows):,}")

    print(f"\n  Sampling up to {MAX_PAIRS:,} window pairs …")
    distances, labels = build_pairs(windows, MAX_PAIRS, rng)

    print(f"\n  Label distribution: same-user={sum(labels):,}  "
          f"different={len(labels)-sum(labels):,}")

    best_tau, sweep_df = threshold_sweep(distances, labels)

    # Evaluate at best threshold
    dists_arr = np.array(distances)
    preds     = (dists_arr < best_tau).astype(int)
    labels_arr = np.array(labels)

    f1   = f1_score(labels_arr,        preds, zero_division=0)
    prec = precision_score(labels_arr, preds, zero_division=0)
    rec  = recall_score(labels_arr,    preds, zero_division=0)

    # AUC: predict 'same user' when distance is LOW → negate for AUC
    neg_dists = -dists_arr
    finite_mask = np.isfinite(neg_dists)
    auc = roc_auc_score(labels_arr[finite_mask], neg_dists[finite_mask]) \
        if len(np.unique(labels_arr)) > 1 else 0.0

    results = {
        "method":         "1D-Chamfer",
        "best_threshold": float(best_tau),
        "precision":      float(prec),
        "recall":         float(rec),
        "f1_score":       float(f1),
        "roc_auc":        float(auc),
        "n_pairs":        len(labels),
        "n_windows":      len(windows),
    }

    print("\n── Results (1-D Chamfer Distance Baseline)")
    for k, v in results.items():
        print(f"  {k:<20}: {v}")

    out_json = RESULTS_DIR / "04_chamfer_1d_results.json"
    out_csv  = RESULTS_DIR / "04_chamfer_1d_threshold_sweep.csv"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sweep_df.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_json.name}  |  {out_csv.name}")

    print("\n[OK] Step 4 complete — run 05_baseline_classifiers.py next.\n")


if __name__ == "__main__":
    main()
