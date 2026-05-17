"""
05_baseline_classifiers.py  |  MixTrace Pipeline — Step 5: Baseline Classifiers
==================================================================================
Trains and evaluates three classical ML baselines on the Elliptic Dataset:

  1. Random Forest (RF)      — supervised, class-weighted, tuned via GridSearchCV
  2. K-Nearest Neighbours (KNN) — supervised, tuned via GridSearchCV
  3. K-Means Clustering      — unsupervised, cluster label assigned by majority vote

All three operate on the pre-scaled 165-dimensional Elliptic feature matrix
(including the four novel engineered features from Step 3 where available).

Evaluation follows the chronological train/val/test split from Step 2 and
reports Precision, Recall, F1-score (macro + illicit-class), and PR-AUC.
Results are saved for the comparison table in Step 13.

Run
---
    python 05_baseline_classifiers.py

Reads   : data/processed/  (Steps 2–3 output)
Outputs : results/05_baseline_rf_metrics.json
          results/05_baseline_knn_metrics.json
          results/05_baseline_kmeans_metrics.json
          models/rf_baseline.pkl
          models/knn_baseline.pkl
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"
MODELS_DIR  = HERE / "models"

RANDOM_SEED = 42


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_elliptic_splits() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Load Elliptic scaled features and chronological masks, then
    filter to labeled nodes and apply the train/val/test split.
    """
    X_all      = np.load(DATA_PROC / "elliptic_X_all.npy")
    y_all      = np.load(DATA_PROC / "elliptic_y_all.npy")
    mask_train = np.load(DATA_PROC / "elliptic_mask_train.npy")
    mask_val   = np.load(DATA_PROC / "elliptic_mask_val.npy")
    mask_test  = np.load(DATA_PROC / "elliptic_mask_test.npy")

    labeled = y_all >= 0

    # Labeled nodes in each split
    train_idx = np.where(mask_train & labeled)[0]
    val_idx   = np.where(mask_val   & labeled)[0]
    test_idx  = np.where(mask_test  & labeled)[0]

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx],  y_all[val_idx]
    X_te, y_te   = X_all[test_idx],  y_all[test_idx]

    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ── Metrics Helper ────────────────────────────────────────────────────────────

def compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Return a metrics dict with precision, recall, F1 for illicit class and macro."""
    pr_ill = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    re_ill = recall_score(   y_true, y_pred, pos_label=1, zero_division=0)
    f1_ill = f1_score(       y_true, y_pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(       y_true, y_pred, average="macro", zero_division=0)

    pr_auc = 0.0
    if y_proba is not None:
        pr_auc = average_precision_score(y_true, y_proba)

    metrics = {
        "method":            name,
        "precision_illicit": float(pr_ill),
        "recall_illicit":    float(re_ill),
        "f1_illicit":        float(f1_ill),
        "f1_macro":          float(f1_mac),
        "pr_auc":            float(pr_auc),
        "support_illicit":   int((y_true == 1).sum()),
        "support_licit":     int((y_true == 0).sum()),
    }
    return metrics


def print_metrics(m: dict) -> None:
    print(f"  Precision (illicit) : {m['precision_illicit']:.4f}")
    print(f"  Recall    (illicit) : {m['recall_illicit']:.4f}")
    print(f"  F1        (illicit) : {m['f1_illicit']:.4f}")
    print(f"  F1 (macro)          : {m['f1_macro']:.4f}")
    print(f"  PR-AUC              : {m['pr_auc']:.4f}")


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray,  y_te: np.ndarray,
) -> dict:
    print("\n── Random Forest Baseline")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    gs = GridSearchCV(rf, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")

    # Combine train + val for final evaluation (standard practice)
    X_tv = np.vstack([X_tr, X_val])
    y_tv = np.concatenate([y_tr, y_val])
    best.fit(X_tv, y_tv)

    y_pred  = best.predict(X_te)
    y_proba = best.predict_proba(X_te)[:, 1]

    metrics = compute_metrics("RandomForest", y_te, y_pred, y_proba)
    print_metrics(metrics)

    joblib.dump(best, MODELS_DIR / "rf_baseline.pkl")
    return metrics


# ── K-Nearest Neighbours ──────────────────────────────────────────────────────

def train_knn(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray,  y_te: np.ndarray,
) -> dict:
    print("\n── K-Nearest Neighbours Baseline")

    param_grid = {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    knn = KNeighborsClassifier(n_jobs=-1)
    gs  = GridSearchCV(knn, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")

    X_tv = np.vstack([X_tr, X_val])
    y_tv = np.concatenate([y_tr, y_val])
    best.fit(X_tv, y_tv)

    y_pred  = best.predict(X_te)
    y_proba = best.predict_proba(X_te)[:, 1]

    metrics = compute_metrics("KNN", y_te, y_pred, y_proba)
    print_metrics(metrics)

    joblib.dump(best, MODELS_DIR / "knn_baseline.pkl")
    return metrics


# ── K-Means Clustering ────────────────────────────────────────────────────────

def train_kmeans(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,  y_te: np.ndarray,
) -> dict:
    """
    Unsupervised K-Means with k=2 clusters.

    Cluster labels are resolved by majority vote against the training labels
    to ensure the cluster 'illicit' / 'licit' mapping is consistent.
    """
    print("\n── K-Means Clustering Baseline (k=2, unsupervised)")

    km = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    X_all = np.vstack([X_tr, X_te])
    km.fit(X_all)

    # Determine which cluster corresponds to illicit (majority vote on train)
    train_clusters = km.predict(X_tr)
    illicit_cluster = int(
        pd.Series(train_clusters[y_tr == 1]).value_counts().idxmax()
    )

    raw_preds = km.predict(X_te)
    y_pred    = (raw_preds == illicit_cluster).astype(int)

    metrics = compute_metrics("KMeans", y_te, y_pred, y_proba=None)
    print_metrics(metrics)

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 5: Baseline Classifiers")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Loading Elliptic splits …")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_elliptic_splits()
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}")
    print(f"  Test illicit: {(y_te==1).sum():,}  licit: {(y_te==0).sum():,}")

    rf_metrics  = train_random_forest(X_tr, y_tr, X_val, y_val, X_te, y_te)
    knn_metrics = train_knn(X_tr, y_tr, X_val, y_val, X_te, y_te)
    km_metrics  = train_kmeans(X_tr, y_tr, X_te, y_te)

    # Classification reports
    print("\n── Full Classification Report — Random Forest (Test)")
    rf_model = joblib.load(MODELS_DIR / "rf_baseline.pkl")
    print(classification_report(y_te, rf_model.predict(X_te),
                                 target_names=["licit", "illicit"]))

    # Save results
    for name, metrics in [
        ("05_baseline_rf_metrics",     rf_metrics),
        ("05_baseline_knn_metrics",    knn_metrics),
        ("05_baseline_kmeans_metrics", km_metrics),
    ]:
        path = RESULTS_DIR / f"{name}.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  Saved → {path.name}")

    print("\n[OK] Step 5 complete — run 06_graph_builder.py next.\n")


if __name__ == "__main__":
    main()
