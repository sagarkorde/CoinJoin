"""
11_multidim_chamfer.py  |  MixTrace Pipeline — Step 11: Multi-Dimensional Chamfer (Gap 2)
===========================================================================================
Addresses Research Gap 2: Multi-Dimensional Post-Mix Spending (CST) Deanonymization.

Motivation
----------
The baseline one-sided Chamfer distance (Step 4) uses only timestamps as the
1-D point cloud, capturing temporal clustering of CoinJoin inputs in a CST.
This is insufficient because two temporally-close transactions may originate from
different users if their structural signatures (fee patterns, UTXO counts) differ.

The Multi-Dimensional Chamfer Distance (MD-CD) extends the metric to an
n-dimensional feature space, enabling richer CST similarity measurement:

    d_C(U, V) = (1 / |U|) * Σ_{u_i ∈ U}  min_{v_j ∈ V}  ||u_i - v_j||_2

where each point u_i ∈ R^d captures [timestamp, fee_rate, input_count, output_count].

Evaluation Protocol
-------------------
  1. Extract CoinJoin temporal windows (same protocol as Step 4).
  2. Build 4-D feature vectors for each window.
  3. Compute pairwise MD-CD between window pairs.
  4. Sweep thresholds; report best F1, precision, recall.
  5. Compare against the 1-D baseline from Step 4.

Run
---
    python 11_multidim_chamfer.py

Reads   : data/processed/author_features_engineered.parquet  (Step 3)
          results/04_chamfer_1d_results.json                  (Step 4)
Outputs : results/11_chamfer_comparison.json
          results/11_chamfer_threshold_sweep_md.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"

MAX_PAIRS   = 5_000
RANDOM_SEED = 42

# Feature dimensions used in MD-CD (must exist in Author Dataset)
MD_FEATURES = [
    "timestamp_unix",          # temporal dimension (computed inline)
    "fee_rate_sat_per_vbyte",  # fee behaviour
    "input_count",             # structural
    "output_count",            # structural
]


# ── Core Algorithms ───────────────────────────────────────────────────────────

def chamfer_1d(U: np.ndarray, V: np.ndarray) -> float:
    """One-sided Chamfer distance on 1-D timestamp arrays."""
    if len(U) == 0 or len(V) == 0:
        return float("inf")
    U_col = U.reshape(-1, 1)
    V_col = V.reshape(-1, 1)
    dist  = cdist(U_col, V_col, metric="euclidean")
    return float(dist.min(axis=1).mean())


def chamfer_multidim(U: np.ndarray, V: np.ndarray) -> float:
    """
    One-sided Chamfer distance on d-dimensional feature arrays.

    Parameters
    ----------
    U : [n, d] — target CST feature matrix
    V : [m, d] — candidate CST feature matrix

    Returns
    -------
    float — mean minimum Euclidean distance from each row of U to V
    """
    if len(U) == 0 or len(V) == 0:
        return float("inf")
    dist = cdist(U, V, metric="euclidean")   # [n, m]
    return float(dist.min(axis=1).mean())


# ── Data Preparation ──────────────────────────────────────────────────────────

def load_and_prepare(sample_frac: float = 0.20) -> pd.DataFrame:
    """Load Author Dataset and compute Unix timestamp column."""
    eng_path = DATA_PROC / "author_features_engineered.parquet"
    raw_path = DATA_PROC / "author_raw.parquet"
    df = pd.read_parquet(eng_path if eng_path.exists() else raw_path)
    df = df.sample(frac=sample_frac, random_state=RANDOM_SEED).copy()

    df["timestamp_unix"] = df["timestamp"].astype("int64") // 10**9
    df["window"] = (
        (df["block_height"] // 6).astype(str) + "_" +
        df["hour"].astype(str) + "_" +
        df["day_of_week"].astype(str)
    )
    return df


def build_window_clouds(
    df: pd.DataFrame,
    mode: str,
) -> dict[str, np.ndarray]:
    """
    Group CoinJoin-like transactions into temporal windows.

    Parameters
    ----------
    mode : '1d' → return [n, 1] timestamp-only arrays
           'md' → return [n, 4] multi-dimensional arrays
    """
    cj = df[df["is_coinjoin_like"]].copy()

    # Normalise all feature columns globally
    feat_cols = MD_FEATURES
    missing   = [c for c in feat_cols if c not in cj.columns]
    if missing:
        for c in missing:
            cj[c] = 0.0

    X_raw = cj[feat_cols].values.astype(np.float64)
    X_scaled = StandardScaler().fit_transform(X_raw)
    cj[feat_cols] = X_scaled

    windows: dict[str, np.ndarray] = {}
    for key, grp in cj.groupby("window"):
        if len(grp) < 3:
            continue
        if mode == "1d":
            windows[key] = grp[["timestamp_unix"]].values.astype(np.float64)
        else:
            windows[key] = grp[feat_cols].values.astype(np.float64)

    return windows


# ── Pair Evaluation ───────────────────────────────────────────────────────────

def build_pairs_and_distances(
    windows: dict[str, np.ndarray],
    distance_fn,
    max_pairs: int,
    rng: np.random.Generator,
    label: str = "",
) -> tuple[list[float], list[int]]:
    keys  = list(windows.keys())
    seen: set[tuple[int, int]] = set()
    idx_pairs = []
    while len(idx_pairs) < max_pairs:
        i, j = rng.integers(0, len(keys), size=2)
        if i != j and (i, j) not in seen and (j, i) not in seen:
            idx_pairs.append((i, j))
            seen.add((i, j))

    distances, labels = [], []
    for i, j in tqdm(idx_pairs, desc=f"  Computing {label} Chamfer"):
        d = distance_fn(windows[keys[i]], windows[keys[j]])
        distances.append(d)
        day_i = keys[i].split("_")[2]
        day_j = keys[j].split("_")[2]
        labels.append(1 if day_i == day_j else 0)

    return distances, labels


def threshold_sweep(
    distances: list[float],
    labels:    list[int],
    n_steps:   int = 200,
) -> tuple[float, float, float, float, pd.DataFrame]:
    dists  = np.array(distances)
    labels = np.array(labels)
    finite = np.isfinite(dists)
    dists  = dists[finite]
    labels = labels[finite]

    thresholds = np.linspace(dists.min(), np.percentile(dists, 95), n_steps)
    rows, best_f1, best_row = [], 0.0, {}

    for tau in thresholds:
        preds = (dists < tau).astype(int)
        if preds.sum() == 0:
            continue
        p  = precision_score(labels, preds, zero_division=0)
        r  = recall_score(labels,    preds, zero_division=0)
        f1 = f1_score(labels,        preds, zero_division=0)
        row = {"threshold": tau, "precision": p, "recall": r, "f1": f1}
        rows.append(row)
        if f1 > best_f1:
            best_f1 = f1
            best_row = row

    return (
        best_row.get("threshold", 0.0),
        best_row.get("f1",        0.0),
        best_row.get("precision", 0.0),
        best_row.get("recall",    0.0),
        pd.DataFrame(rows),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 11: Multi-Dimensional Chamfer (Gap 2)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("\n  Loading and preparing Author Dataset …")
    df = load_and_prepare(sample_frac=0.20)
    print(f"  Sample: {len(df):,} txs  |  "
          f"CoinJoin-like: {int(df['is_coinjoin_like'].sum()):,}")

    # ── 1-D Chamfer ──
    print("\n── 1-D Chamfer Distance (timestamp only)")
    windows_1d = build_window_clouds(df, mode="1d")
    print(f"  Windows: {len(windows_1d):,}")
    dists_1d, labels_1d = build_pairs_and_distances(
        windows_1d, chamfer_1d, MAX_PAIRS, rng, label="1D"
    )
    tau_1d, f1_1d, prec_1d, rec_1d, sweep_1d = threshold_sweep(dists_1d, labels_1d)
    print(f"  Best threshold : {tau_1d:.6f}")
    print(f"  F1             : {f1_1d:.4f}")
    print(f"  Precision      : {prec_1d:.4f}")
    print(f"  Recall         : {rec_1d:.4f}")

    # ── Multi-Dimensional Chamfer ──
    print(f"\n── Multi-Dimensional Chamfer Distance  (d={len(MD_FEATURES)})")
    print(f"  Dimensions: {MD_FEATURES}")
    windows_md = build_window_clouds(df, mode="md")
    print(f"  Windows: {len(windows_md):,}")
    dists_md, labels_md = build_pairs_and_distances(
        windows_md, chamfer_multidim, MAX_PAIRS, rng, label="MD"
    )
    tau_md, f1_md, prec_md, rec_md, sweep_md = threshold_sweep(dists_md, labels_md)
    print(f"  Best threshold : {tau_md:.6f}")
    print(f"  F1             : {f1_md:.4f}")
    print(f"  Precision      : {prec_md:.4f}")
    print(f"  Recall         : {rec_md:.4f}")

    # ── Comparison ──
    f1_lift = f1_md - f1_1d
    print(f"\n── Comparison")
    print(f"  F1 improvement (MD over 1D) : {f1_lift:+.4f}  "
          f"({f1_lift / max(f1_1d, 1e-9) * 100:.1f} % relative)")

    # Load 1D baseline from Step 4 if available (for cross-step consistency)
    baseline_path = RESULTS_DIR / "04_chamfer_1d_results.json"
    baseline_f1   = None
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as fh:
            baseline_f1 = json.load(fh).get("f1_score")
        print(f"  Step-4 1D baseline F1 : {baseline_f1:.4f}")

    # ── Save ──
    results = {
        "chamfer_1d": {
            "method": "1D-Chamfer (timestamp)",
            "best_threshold": float(tau_1d),
            "f1_score":   float(f1_1d),
            "precision":  float(prec_1d),
            "recall":     float(rec_1d),
            "dimensions": 1,
        },
        "chamfer_md": {
            "method": f"MD-Chamfer (d={len(MD_FEATURES)})",
            "dimensions": len(MD_FEATURES),
            "dimension_names": MD_FEATURES,
            "best_threshold": float(tau_md),
            "f1_score":   float(f1_md),
            "precision":  float(prec_md),
            "recall":     float(rec_md),
        },
        "f1_absolute_improvement": float(f1_lift),
        "f1_relative_improvement_pct": float(
            f1_lift / max(f1_1d, 1e-9) * 100
        ),
    }
    if baseline_f1 is not None:
        results["step4_baseline_f1"] = float(baseline_f1)

    out_json = RESULTS_DIR / "11_chamfer_comparison.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sweep_md.to_csv(RESULTS_DIR / "11_chamfer_threshold_sweep_md.csv", index=False)
    sweep_1d.to_csv(RESULTS_DIR / "11_chamfer_threshold_sweep_1d.csv", index=False)
    print(f"\n  Saved → {out_json.name}")

    print("\n[OK] Step 11 complete — run 12_coinjoin_detector.py next.\n")


if __name__ == "__main__":
    main()
