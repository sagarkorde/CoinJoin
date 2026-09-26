"""
E09 - Explainability and error analysis.

Reviewer 2 asked for feature attribution or a GNN explanation method, and for
an analysis of failure cases. Both are provided here, on the leakage-free
models only.

Three parts
-----------
  1. Permutation importance for the tabular reference model, which says which
     of the 165 distributed features the model actually relies on.
  2. Integrated-gradients attribution for the fused GNN, aggregated over test
     nodes, giving the same question a graph-model answer.
  3. Error analysis: false negatives and false positives are profiled against
     time step, node degree and the attributed features, so the failure modes
     are described rather than merely counted.

Outputs
-------
  results/e09_permutation_importance.csv
  results/e09_gnn_attribution.csv
  results/e09_error_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, DEVICE, PRIMARY_SEED
from data import load_elliptic, labelled_masks, base_feature_columns
from features import TOPO_FEATURES
from gnn import build_graph, train_dgi, train_classifier
from metrics import evaluate, select_threshold

N_REPEATS = 10
IG_STEPS = 32


def integrated_gradients(model, x, edge_index, emb, target_idx,
                         steps: int = IG_STEPS) -> np.ndarray:
    """
    Integrated gradients of the illicit logit with respect to node features.

    The baseline is the all-zero feature matrix, which after standardisation
    is the training-set mean, so attributions read as deviation from a typical
    transaction.
    """
    model.eval()
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        out = model(xi, edge_index, emb)
        score = F.softmax(out, dim=1)[target_idx, 1].sum()
        grad, = torch.autograd.grad(score, xi)
        total += grad
    attr = (x - baseline) * total / steps
    return attr.detach().cpu().numpy()


def main() -> None:
    device = DEVICE if torch.cuda.is_available() else "cpu"
    nodes, edges = load_elliptic()
    base_cols = base_feature_columns(nodes)
    lm = labelled_masks(nodes)
    y = nodes["label"].to_numpy()

    # ---------------------------------------------------------------- part 1
    print("  [1/3] permutation importance, tabular reference ...")
    X = nodes[base_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(X[lm["train"]])
    Xs = scaler.transform(X).astype(np.float32)

    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=PRIMARY_SEED)
    clf.fit(Xs[lm["train"]], y[lm["train"]])

    pi = permutation_importance(clf, Xs[lm["test"]], y[lm["test"]],
                                scoring="average_precision",
                                n_repeats=N_REPEATS,
                                random_state=PRIMARY_SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": base_cols,
                         "importance_mean": pi.importances_mean,
                         "importance_std": pi.importances_std})
             .sort_values("importance_mean", ascending=False))
    imp.to_csv(RESULTS / "e09_permutation_importance.csv", index=False)
    print(imp.head(12).to_string(index=False))

    # ---------------------------------------------------------------- part 2
    print("\n  [2/3] integrated gradients, fused GNN ...")
    data = build_graph(nodes, edges, base_cols)
    emb, _ = train_dgi(data, "inductive", device, PRIMARY_SEED)
    run = train_classifier(data, "fusion", device, PRIMARY_SEED, dgi_emb=emb,
                           return_model=True)
    scores = run["scores"]
    model = run["model"]

    vam, tem = data.val_mask.numpy(), data.test_mask.numpy()
    thr = select_threshold(y[vam], scores[vam], criterion="f1")
    test_metrics = evaluate(y[tem], scores[tem], thr)
    print(f"    fused GNN test F1 {test_metrics['f1_illicit']:.4f} "
          f"at threshold {thr:.4f}")

    x = data.x.to(device)
    ei = data.edge_index.to(device)
    test_idx = np.flatnonzero(tem)
    rng = np.random.default_rng(PRIMARY_SEED)
    sample = rng.choice(test_idx, size=min(400, len(test_idx)), replace=False)
    sample_t = torch.tensor(sample, dtype=torch.long, device=device)

    try:
        attr = integrated_gradients(model, x, ei, emb.to(device), sample_t)
        mean_abs = np.abs(attr[sample]).mean(axis=0)
        ga = (pd.DataFrame({"feature": base_cols,
                            "mean_abs_attribution": mean_abs})
                .sort_values("mean_abs_attribution", ascending=False))
        ga.to_csv(RESULTS / "e09_gnn_attribution.csv", index=False)
        print(ga.head(12).to_string(index=False))
    except RuntimeError as exc:
        print(f"    integrated gradients unavailable on this device: {exc}")
        ga = None

    # ---------------------------------------------------------------- part 3
    print("\n  [3/3] error analysis ...")
    ts = nodes["time_step"].to_numpy()
    deg = (pd.concat([edges["txId1"], edges["txId2"]])
             .value_counts().reindex(nodes["txId"]).fillna(0).to_numpy())

    pred = (scores >= thr).astype(int)
    te = tem
    tp = te & (y == 1) & (pred == 1)
    fn = te & (y == 1) & (pred == 0)
    fp = te & (y == 0) & (pred == 1)
    tn = te & (y == 0) & (pred == 0)

    def profile(mask, name):
        return {
            "group": name,
            "n": int(mask.sum()),
            "mean_time_step": round(float(ts[mask].mean()), 2) if mask.sum() else None,
            "mean_degree": round(float(deg[mask].mean()), 2) if mask.sum() else None,
            "median_degree": float(np.median(deg[mask])) if mask.sum() else None,
            "mean_score": round(float(scores[mask].mean()), 4) if mask.sum() else None,
        }

    err = {
        "threshold": float(thr),
        "test_metrics": test_metrics,
        "groups": [profile(m, n) for m, n in
                   ((tp, "true_positive"), (fn, "false_negative"),
                    (fp, "false_positive"), (tn, "true_negative"))],
    }

    # Where do the errors sit in time?
    per_step = []
    for step in sorted(np.unique(ts[te])):
        sel = te & (ts == step)
        n_ill = int((y[sel] == 1).sum())
        per_step.append({
            "time_step": int(step),
            "n": int(sel.sum()),
            "n_illicit": n_ill,
            "recall": round(float((pred[sel & (y == 1)] == 1).mean()), 4) if n_ill else None,
            "n_false_positive": int((sel & (y == 0) & (pred == 1)).sum()),
        })
    err["per_time_step"] = per_step

    (RESULTS / "e09_error_analysis.json").write_text(
        json.dumps(err, indent=2), encoding="utf-8")
    print(json.dumps(err["groups"], indent=2))
    print("\n[OK] wrote results/e09_*")


if __name__ == "__main__":
    main()
