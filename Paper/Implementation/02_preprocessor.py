"""
02_preprocessor.py  |  MixTrace Pipeline — Step 2: Preprocessing
=================================================================
Cleans, normalises, and splits both datasets for downstream modelling.

Author Dataset
--------------
  - Drops non-numeric / identifier columns
  - Converts booleans to int, timestamp to Unix seconds
  - Imputes rare nulls with column medians
  - Applies StandardScaler to numeric features
  - Produces X_author (feature matrix) and y_author (binary labels)
  - Stratified 70 / 10 / 20 train-val-test split

Elliptic Dataset
----------------
  - Scales 165-dimensional feature vector (time_step kept raw as temporal index)
  - Chronological masks: train steps 1–34, val 35–41, test 42–49
  - Filters supervised subset (labeled nodes only) for baseline classifiers

Run
---
    python 02_preprocessor.py

Reads   : data/processed/  (Step 1 output)
Outputs : data/processed/
    author_X_train.npy, author_X_val.npy, author_X_test.npy
    author_y_train.npy, author_y_val.npy, author_y_test.npy
    author_feature_names.txt
    elliptic_X_labeled.npy, elliptic_y_labeled.npy
    elliptic_X_all.npy          — scaled features for all nodes (GNN input)
    elliptic_node_ids.npy
    elliptic_mask_train.npy, elliptic_mask_val.npy, elliptic_mask_test.npy
    scalers/author_scaler.pkl
    scalers/elliptic_scaler.pkl
    results/02_split_stats.txt
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"
SCALERS_DIR = DATA_PROC / "scalers"

# Elliptic chronological split boundaries (standard in literature)
VAL_START  = 35
TEST_START = 42

# Author Dataset: columns to drop before modelling
AUTHOR_DROP = {
    "txid", "input_addresses", "output_addresses",
    "input_script_types", "output_script_types",
    "op_return_data", "month_1", "sample_size",
}


# ── Author Dataset ────────────────────────────────────────────────────────────

def preprocess_author(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Clean and scale Author Dataset.

    Returns X (float32 array), y (int array), feature_names.
    """
    df = df.copy()

    # Convert timestamp → Unix seconds (float)
    df["timestamp_unix"] = df["timestamp"].astype("int64") // 10**9

    # Drop identifier/string/redundant columns
    drop_cols = [c for c in AUTHOR_DROP if c in df.columns] + ["timestamp"]
    df.drop(columns=drop_cols, inplace=True)

    # Booleans → int
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)

    # Separate target
    y = df.pop("is_coinjoin_like").astype(int).values

    # Impute remaining nulls with column medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    feature_names = df.columns.tolist()
    X = df[feature_names].values.astype(np.float32)

    return X, y, feature_names


def split_author(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Stratified 70/10/20 train-val-test split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.667, random_state=42, stratify=y_tmp
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def scale_author(
    X_tr: np.ndarray, X_val: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te  = scaler.transform(X_te).astype(np.float32)
    return X_tr, X_val, X_te, scaler


# ── Elliptic Dataset ──────────────────────────────────────────────────────────

def preprocess_elliptic(
    nodes: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale Elliptic features and create chronological node masks.

    Returns
    -------
    X_all     : scaled features for all nodes  [N, 165]
    y_all     : labels for all nodes (-1 = unknown)
    node_ids  : txId array for index mapping
    X_labeled : X rows for labeled nodes only
    y_labeled : y values for labeled nodes (0 or 1)
    mask_train, mask_val, mask_test : boolean masks over all N nodes
    scaler    : fitted StandardScaler
    """
    # Feature columns: drop txId; keep time_step as-is (temporal index)
    feat_cols = [c for c in nodes.columns
                 if c not in {"txId", "class", "label", "time_step"}]

    X_raw    = nodes[feat_cols].values.astype(np.float32)
    y_all    = nodes["label"].values.astype(int)
    ts       = nodes["time_step"].values
    node_ids = nodes["txId"].values

    # Fit scaler on training nodes only (time_step 1–34) to avoid leakage
    train_mask = ts < VAL_START
    scaler     = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X_all = scaler.transform(X_raw).astype(np.float32)

    # Chronological masks
    mask_train = (ts <  VAL_START).astype(bool)
    mask_val   = ((ts >= VAL_START) & (ts < TEST_START)).astype(bool)
    mask_test  = (ts >= TEST_START).astype(bool)

    # Labeled subset (for tabular baselines)
    labeled_idx = y_all >= 0
    X_labeled   = X_all[labeled_idx]
    y_labeled   = y_all[labeled_idx]

    return (X_all, y_all, node_ids, X_labeled, y_labeled,
            mask_train, mask_val, mask_test, scaler)


# ── Save / Report ─────────────────────────────────────────────────────────────

def save_stats(
    y_tr: np.ndarray, y_val: np.ndarray, y_te: np.ndarray,
    y_labeled: np.ndarray,
    mask_train: np.ndarray, mask_val: np.ndarray, mask_test: np.ndarray,
) -> None:
    lines = [
        "=" * 65,
        "MixTrace — Preprocessing Statistics  (Step 2)",
        "=" * 65,
        "",
        "── Author Dataset Split",
        f"   Train  : {len(y_tr):>8,}  "
        f"  CJ={y_tr.sum():,}  ({y_tr.mean()*100:.2f} %)",
        f"   Val    : {len(y_val):>8,}  "
        f"  CJ={y_val.sum():,}  ({y_val.mean()*100:.2f} %)",
        f"   Test   : {len(y_te):>8,}  "
        f"  CJ={y_te.sum():,}  ({y_te.mean()*100:.2f} %)",
        "",
        "── Elliptic Dataset Split (chronological by time_step)",
        f"   Train nodes  (steps  1–34) : {mask_train.sum():>8,}",
        f"   Val   nodes  (steps 35–41) : {mask_val.sum():>8,}",
        f"   Test  nodes  (steps 42–49) : {mask_test.sum():>8,}",
        "",
        "── Elliptic Labeled Subset",
        f"   Total labeled : {len(y_labeled):,}",
        f"   Illicit (1)   : {(y_labeled == 1).sum():,}",
        f"   Licit   (0)   : {(y_labeled == 0).sum():,}",
        f"   Imbalance     : {(y_labeled == 0).sum() / (y_labeled == 1).sum():.1f} : 1",
    ]
    path = RESULTS_DIR / "02_split_stats.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {path.relative_to(HERE)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 2: Preprocessing")
    print("=" * 60)

    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Author Dataset ──
    print("\n[A] Processing Author Dataset …")
    author = pd.read_parquet(DATA_PROC / "author_raw.parquet")

    X, y, feat_names = preprocess_author(author)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_author(X, y)
    X_tr, X_val, X_te, auth_scaler = scale_author(X_tr, X_val, X_te)

    print(f"  Feature dim   : {X_tr.shape[1]}")
    print(f"  Train / Val / Test : {len(y_tr):,} / {len(y_val):,} / {len(y_te):,}")
    print(f"  CoinJoin rate  train: {y_tr.mean()*100:.2f}%  "
          f"val: {y_val.mean()*100:.2f}%  test: {y_te.mean()*100:.2f}%")

    np.save(DATA_PROC / "author_X_train.npy", X_tr)
    np.save(DATA_PROC / "author_X_val.npy",   X_val)
    np.save(DATA_PROC / "author_X_test.npy",  X_te)
    np.save(DATA_PROC / "author_y_train.npy", y_tr)
    np.save(DATA_PROC / "author_y_val.npy",   y_val)
    np.save(DATA_PROC / "author_y_test.npy",  y_te)
    (DATA_PROC / "author_feature_names.txt").write_text(
        "\n".join(feat_names), encoding="utf-8"
    )
    joblib.dump(auth_scaler, SCALERS_DIR / "author_scaler.pkl")

    # ── Elliptic Dataset ──
    print("\n[B] Processing Elliptic Dataset …")
    nodes = pd.read_parquet(DATA_PROC / "elliptic_nodes.parquet")

    (X_all, y_all, node_ids, X_labeled, y_labeled,
     mask_train, mask_val, mask_test, ell_scaler) = preprocess_elliptic(nodes)

    print(f"  Feature dim      : {X_all.shape[1]}")
    print(f"  Train / Val / Test : {mask_train.sum():,} / "
          f"{mask_val.sum():,} / {mask_test.sum():,}")
    print(f"  Labeled subset   : {len(y_labeled):,}  "
          f"(illicit={int((y_labeled==1).sum())}, licit={int((y_labeled==0).sum())})")

    np.save(DATA_PROC / "elliptic_X_all.npy",       X_all)
    np.save(DATA_PROC / "elliptic_y_all.npy",        y_all)
    np.save(DATA_PROC / "elliptic_node_ids.npy",     node_ids)
    np.save(DATA_PROC / "elliptic_X_labeled.npy",   X_labeled)
    np.save(DATA_PROC / "elliptic_y_labeled.npy",   y_labeled)
    np.save(DATA_PROC / "elliptic_mask_train.npy",  mask_train)
    np.save(DATA_PROC / "elliptic_mask_val.npy",    mask_val)
    np.save(DATA_PROC / "elliptic_mask_test.npy",   mask_test)
    joblib.dump(ell_scaler, SCALERS_DIR / "elliptic_scaler.pkl")

    save_stats(y_tr, y_val, y_te, y_labeled, mask_train, mask_val, mask_test)

    print("\n[OK] Step 2 complete — run 03_feature_engineer.py next.\n")


if __name__ == "__main__":
    main()
