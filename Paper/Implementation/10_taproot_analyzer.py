"""
10_taproot_analyzer.py  |  MixTrace Pipeline — Step 10: Taproot Analysis (Gap 1)
==================================================================================
Addresses Research Gap 1: Taproot-Upgrade Clustering Bypasses and False Positives.

Background
----------
Bitcoin's Taproot soft fork (BIP 340/341/342) introduced Pay-to-Taproot (P2TR)
scripts using Schnorr signatures and key aggregation. This makes single-user
public keys and multi-party aggregated keys (e.g., MuSig2 for CoinJoin) visually
indistinguishable on the blockchain, defeating the traditional Common Input
Ownership Heuristic (CIOH).

Analysis Protocol
-----------------
  1. Isolate Taproot transactions from the Author Dataset.
  2. Baseline: apply naive CIOH — predict all multi-input Taproot txs as CoinJoin.
  3. Proposed: Taproot-aware DBSCAN clustering on structural+behavioural features:
       [input_count, output_count, fee_rate_sat_per_vbyte, value_concentration_ratio,
        address_reuse, mixing_index (novel)]
  4. Compare false-positive rates: naive CIOH vs. Taproot-aware clustering.
  5. Measure cluster purity using the ground-truth `is_coinjoin_like` label.

Metrics
-------
  - False Positive Rate (FPR): non-CoinJoin txs incorrectly flagged as CoinJoin
  - False Negative Rate (FNR): CoinJoin txs missed
  - F1-score, Precision, Recall (treating CoinJoin as positive class)
  - Silhouette score (cluster quality)

Run
---
    python 10_taproot_analyzer.py

Reads   : data/processed/author_features_engineered.parquet  (Step 3)
          data/processed/author_raw.parquet                   (fallback)
Outputs : results/10_taproot_analysis.json
          results/10_taproot_cluster_summary.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"

# Features used for Taproot-aware clustering
CLUSTER_FEATURES = [
    "input_count",
    "output_count",
    "fee_rate_sat_per_vbyte",
    "value_concentration_ratio",
    "address_reuse",
    "input_output_ratio",
]
NOVEL_CLUSTER_FEAT = "mixing_index"   # added if Step 3 ran

RANDOM_SEED = 42
SAMPLE_SIZE   = 200_000   # overall analysis sample
DBSCAN_SAMPLE = 30_000    # DBSCAN radius-neighbor graph is O(n²) memory; cap here


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_taproot_data() -> tuple[pd.DataFrame, str]:
    """
    Load transactions for Taproot analysis.

    Primary: filter to `has_taproot == True`.
    Fallback: if the dataset contains no Taproot transactions (e.g. the Author
    Dataset predates widespread P2TR adoption), use P2WPKH transactions as a
    structurally similar proxy — both script types are affected by the CIOH
    limitation addressed in Research Gap 1.  A second fallback uses all
    multi-input transactions to ensure the analysis always runs.
    """
    eng_path = DATA_PROC / "author_features_engineered.parquet"
    raw_path = DATA_PROC / "author_raw.parquet"

    df = pd.read_parquet(eng_path if eng_path.exists() else raw_path)

    # ── Primary: Taproot ──
    subset = df[df["has_taproot"]].copy()
    source = "Taproot (P2TR)"

    # ── Fallback 1: P2WPKH (SegWit native) ──
    if len(subset) == 0 and "has_p2wpkh" in df.columns:
        print("  ⚠  No Taproot txs in dataset — falling back to P2WPKH transactions")
        subset = df[df["has_p2wpkh"]].copy()
        source = "P2WPKH (SegWit native, Taproot proxy)"

    # ── Fallback 2: all multi-input transactions ──
    if len(subset) == 0:
        print("  ⚠  No P2WPKH txs either — using all multi-input transactions")
        subset = df[df["input_count"] > 1].copy()
        source = "Multi-input transactions (general CIOH proxy)"

    print(f"  Analysis scope         : {source}")
    print(f"  Total transactions     : {len(subset):,}")
    cj_n = int(subset["is_coinjoin_like"].sum())
    cj_r = subset["is_coinjoin_like"].mean() * 100 if len(subset) > 0 else 0
    print(f"  CoinJoin-like          : {cj_n:,}  ({cj_r:.2f} %)")

    if len(subset) == 0:
        raise RuntimeError("No transactions available for Taproot analysis.")

    # Subsample for tractable DBSCAN
    if len(subset) > SAMPLE_SIZE:
        subset = subset.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)
        print(f"  Subsampled to          : {len(subset):,}")

    return subset.reset_index(drop=True), source


# ── Naive CIOH Baseline ───────────────────────────────────────────────────────

def naive_cioh(df: pd.DataFrame) -> np.ndarray:
    """
    Naive Common Input Ownership Heuristic:
    predict CoinJoin if a Taproot transaction has > 2 inputs AND > 2 outputs.
    """
    return ((df["input_count"] > 2) & (df["output_count"] > 2)).astype(int).values


# ── Taproot-Aware DBSCAN Clustering ──────────────────────────────────────────

def taproot_aware_clustering(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Structural DBSCAN clustering on multi-dimensional transaction features.

    DBSCAN's radius-neighbor graph requires O(n²) memory.  To stay within
    RAM limits the function draws a stratified subsample of DBSCAN_SAMPLE rows,
    fits DBSCAN on that subsample, then propagates labels to the full DataFrame
    via a 1-NN lookup (predict by nearest cluster centroid).

    Returns
    -------
    preds         : int array [N]  (1 = predicted CoinJoin, 0 = non-CoinJoin)
    cluster_labels: int array [N]  (raw cluster ids; -1 = noise, propagated)
    X_scaled      : float array [N, d]  (full scaled feature matrix)
    """
    feat_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    if NOVEL_CLUSTER_FEAT in df.columns:
        feat_cols.append(NOVEL_CLUSTER_FEAT)

    if len(df) == 0:
        raise RuntimeError("taproot_aware_clustering received an empty DataFrame.")

    X_full   = StandardScaler().fit_transform(
        df[feat_cols].fillna(0).values.astype(np.float64)
    )
    y_true_full = df["is_coinjoin_like"].astype(int).values

    # ── Stratified subsample for DBSCAN ──
    n_sub = min(DBSCAN_SAMPLE, len(df))
    rng   = np.random.default_rng(RANDOM_SEED)
    sub_idx = rng.choice(len(df), size=n_sub, replace=False)
    X_sub   = X_full[sub_idx]
    y_sub   = y_true_full[sub_idx]

    print(f"  DBSCAN subsample       : {n_sub:,} / {len(df):,}")

    # n_jobs=1 + algorithm='kd_tree' avoids the parallel MemoryError
    db = DBSCAN(eps=0.6, min_samples=5, algorithm="kd_tree", n_jobs=1)
    sub_labels = db.fit_predict(X_sub)

    n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    n_noise    = int((sub_labels == -1).sum())
    print(f"  DBSCAN clusters        : {n_clusters}")
    print(f"  Noise points           : {n_noise:,}  ({n_noise/n_sub*100:.1f} %)")

    # ── Resolve cluster → CoinJoin label (majority vote on subsample) ──
    cluster_to_label: dict[int, int] = {}
    for c_id in set(sub_labels):
        if c_id == -1:
            continue
        mask = sub_labels == c_id
        cluster_to_label[c_id] = int(pd.Series(y_sub[mask]).value_counts().idxmax())

    # ── Propagate labels to full dataset via nearest cluster centroid ──
    # Compute centroid for each cluster in scaled feature space
    centroids, centroid_labels = [], []
    for c_id, lbl in cluster_to_label.items():
        mask = sub_labels == c_id
        centroids.append(X_sub[mask].mean(axis=0))
        centroid_labels.append(lbl)

    full_cluster_labels = np.full(len(df), -1, dtype=int)
    full_preds          = naive_cioh(df)   # default for noise/no-cluster points

    if centroids:
        from scipy.spatial.distance import cdist
        C   = np.array(centroids)           # [K, d]
        # Assign each full point to its nearest centroid
        D   = cdist(X_full, C, metric="euclidean")   # [N, K]
        nn  = D.argmin(axis=1)
        # Only reassign if closest centroid is within eps threshold
        in_range = D[np.arange(len(D)), nn] <= 0.6
        full_cluster_labels[in_range] = nn[in_range]
        for i, in_r in enumerate(in_range):
            if in_r:
                full_preds[i] = centroid_labels[int(nn[i])]

    n_clusters = len(set(full_cluster_labels)) - (1 if -1 in full_cluster_labels else 0)
    print(f"  Assigned clusters (full): {n_clusters}")

    return full_preds, full_cluster_labels, X_full


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    fp    = int(((y_pred == 1) & (y_true == 0)).sum())
    fn    = int(((y_pred == 0) & (y_true == 1)).sum())
    tp    = int(((y_pred == 1) & (y_true == 1)).sum())
    tn    = int(((y_pred == 0) & (y_true == 0)).sum())
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true,    y_pred, zero_division=0)
    f1    = f1_score(y_true,        y_pred, zero_division=0)
    return {
        "method": name, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 10: Taproot Analysis (Gap 1)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Loading transactions for analysis …")
    df, analysis_scope = load_taproot_data()
    y_true = df["is_coinjoin_like"].astype(int).values

    # ── Baseline: Naive CIOH ──
    print("\n── Naive CIOH Baseline")
    cioh_preds  = naive_cioh(df)
    cioh_metrics = compute_metrics("Naive CIOH", y_true, cioh_preds)
    for k, v in cioh_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:<25}: {v}")

    # ── Proposed: Taproot-Aware DBSCAN ──
    print("\n── Taproot-Aware DBSCAN Clustering")
    tap_preds, cluster_labels, X_scaled = taproot_aware_clustering(df)
    tap_metrics = compute_metrics("Taproot-Aware DBSCAN", y_true, tap_preds)
    for k, v in tap_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:<25}: {v}")

    # Silhouette score (exclude noise points)
    valid_mask = cluster_labels != -1
    if valid_mask.sum() > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
        sil = silhouette_score(X_scaled[valid_mask], cluster_labels[valid_mask],
                               sample_size=min(5000, valid_mask.sum()))
        tap_metrics["silhouette_score"] = round(float(sil), 4)
        print(f"  silhouette_score         : {sil:.4f}")

    # ── FPR Reduction ──
    fpr_reduction = cioh_metrics["false_positive_rate"] - tap_metrics["false_positive_rate"]
    print(f"\n  FPR reduction (DBSCAN vs CIOH) : "
          f"{fpr_reduction:+.4f}  "
          f"({fpr_reduction / max(cioh_metrics['false_positive_rate'], 1e-9) * 100:.1f} % relative)")

    # ── Cluster summary table ──
    cluster_summary = []
    for c_id in sorted(set(cluster_labels)):
        mask  = cluster_labels == c_id
        n     = int(mask.sum())
        n_cj  = int(y_true[mask].sum())
        purity = n_cj / n if n > 0 else 0.0
        cluster_summary.append({
            "cluster_id": c_id,
            "size":       n,
            "n_coinjoin": n_cj,
            "purity":     round(purity, 4),
        })

    cluster_df = pd.DataFrame(cluster_summary).sort_values("size", ascending=False)

    # ── Save ──
    results = {
        "analysis_scope":        analysis_scope,
        "naive_cioh":            cioh_metrics,
        "taproot_aware_dbscan":  tap_metrics,
        "fpr_reduction_absolute": round(float(fpr_reduction), 4),
        "n_transactions_analyzed": int(len(df)),
    }
    out_json = RESULTS_DIR / "10_taproot_analysis.json"
    out_csv  = RESULTS_DIR / "10_taproot_cluster_summary.csv"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    cluster_df.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_json.name}  |  {out_csv.name}")

    print("\n[OK] Step 10 complete — run 11_multidim_chamfer.py next.\n")


if __name__ == "__main__":
    main()
