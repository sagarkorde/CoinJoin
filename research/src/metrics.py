"""
Evaluation metrics and uncertainty quantification.

Two protocol decisions are enforced here because round-1 drifted on both:

1. A decision threshold is always selected on the validation split and then
   frozen before the test split is touched. `select_threshold` is the only
   sanctioned way to obtain one.
2. Threshold-free metrics (ROC-AUC, PR-AUC) are reported alongside
   threshold-dependent ones, so a conclusion never rests on a single
   operating point.

Round-1 reviewers also noted that five seeds measure training stochasticity,
not generalisation. `bootstrap_metric_ci` and `paired_bootstrap_difference`
resample the test set itself, which quantifies the complementary source of
uncertainty.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

EPS = 1e-12


def evaluate(y_true: np.ndarray, y_score: np.ndarray,
             threshold: float) -> dict[str, float]:
    """Full metric suite at a fixed decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_illicit": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": float(fp / (fp + tn + EPS)),
        "fnr": float(fn / (fn + tp + EPS)),
    }
    # Threshold-free metrics are undefined for a single-class ground truth.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def select_threshold(y_val: np.ndarray, score_val: np.ndarray,
                     criterion: str = "f1", n_grid: int = 512) -> float:
    """
    Choose a decision threshold on validation data only.

    criterion:
      "f1"  maximise F1 on the illicit class (default; balanced operating point)
      "mcc" maximise Matthews correlation (robust under heavy imbalance)
    """
    y_val = np.asarray(y_val).astype(int)
    score_val = np.asarray(score_val, dtype=float)
    lo, hi = float(score_val.min()), float(score_val.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5
    grid = np.quantile(score_val, np.linspace(0.0, 1.0, n_grid))
    grid = np.unique(np.clip(grid, lo, hi))

    best_t, best_v = 0.5, -np.inf
    for t in grid:
        pred = (score_val >= t).astype(int)
        if criterion == "mcc":
            v = matthews_corrcoef(y_val, pred) if len(set(pred)) > 1 else -1.0
        else:
            v = f1_score(y_val, pred, zero_division=0)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def bootstrap_metric_ci(y_true: np.ndarray, y_score: np.ndarray,
                        threshold: float, metric: str = "f1_illicit",
                        n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> dict[str, float]:
    """Percentile bootstrap CI for one metric, resampling test nodes."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            vals[b] = np.nan
            continue
        vals[b] = evaluate(y_true[idx], y_score[idx], threshold)[metric]
    vals = vals[~np.isnan(vals)]
    return {
        "metric": metric,
        "point": evaluate(y_true, y_score, threshold)[metric],
        "ci_lo": float(np.quantile(vals, alpha / 2)),
        "ci_hi": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": int(len(vals)),
    }


def paired_bootstrap_difference(y_true: np.ndarray, score_a: np.ndarray,
                                score_b: np.ndarray, thr_a: float, thr_b: float,
                                metric: str = "f1_illicit", n_boot: int = 2000,
                                seed: int = 0, alpha: float = 0.05) -> dict:
    """
    Paired bootstrap on the difference metric(A) - metric(B).

    Both systems are scored on the same resampled test nodes, so the pairing
    removes test-set composition as a source of variance. This answers the
    round-1 concern that five seeds cannot support population-level claims.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    n = len(y_true)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            diffs[b] = np.nan
            continue
        diffs[b] = (evaluate(y_true[idx], score_a[idx], thr_a)[metric]
                    - evaluate(y_true[idx], score_b[idx], thr_b)[metric])
    diffs = diffs[~np.isnan(diffs)]
    obs = (evaluate(y_true, score_a, thr_a)[metric]
           - evaluate(y_true, score_b, thr_b)[metric])
    # Two-sided bootstrap p-value: how often the difference crosses zero.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "metric": metric,
        "difference": float(obs),
        "ci_lo": float(np.quantile(diffs, alpha / 2)),
        "ci_hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_bootstrap": float(min(1.0, p)),
        "n_boot": int(len(diffs)),
    }
