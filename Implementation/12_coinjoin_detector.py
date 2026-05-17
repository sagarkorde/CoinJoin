"""
12_coinjoin_detector.py  |  MixTrace Pipeline — Step 12: End-to-End Inference
===============================================================================
Loads the trained MixTrace GAT model and runs full-graph inference on the
Elliptic test set.  Optimises the classification threshold to maximise F1
on the validation set, then reports test-set metrics.

Also performs ablation inference:
  - GAT without DGI skip (zeroed DGI embeddings) — quantifies DGI contribution
  - GAT without novel features (base 165-dim input) — quantifies feature contribution

Outputs node-level prediction scores that are consumed by Step 13 (evaluation)
and Step 14 (visualisation).

Run
---
    python 12_coinjoin_detector.py

Reads   : models/mixtrace_gat_best.pt  (Step 9)
          models/dgi_encoder.pt        (Step 7)
          data/processed/elliptic_graph.pt  (Step 6)
Outputs : results/12_test_predictions.csv     — per-node scores + labels
          results/12_threshold_sweep.csv      — val-set F1 vs. threshold
          results/12_final_test_metrics.json  — final evaluation metrics
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"
MODELS_DIR  = HERE / "models"

RANDOM_SEED = 42


# ── Model Reconstruction ──────────────────────────────────────────────────────

def build_gat_model(ckpt: dict, device: torch.device) -> nn.Module:
    """Reconstruct MixTraceGAT from checkpoint metadata."""
    from torch_geometric.nn import BatchNorm, GATConv

    in_dim        = ckpt["in_dim"]
    hidden_dim    = ckpt["hidden_dim"]
    n_heads       = ckpt["n_heads"]
    dgi_embed_dim = ckpt["dgi_embed_dim"]
    dropout       = ckpt["dropout"]

    class _GAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout    = dropout
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            self.gat1       = GATConv(hidden_dim, hidden_dim, heads=n_heads,
                                      dropout=dropout, concat=True)
            self.bn1        = BatchNorm(hidden_dim * n_heads)
            self.gat2       = GATConv(hidden_dim * n_heads, hidden_dim, heads=1,
                                      dropout=dropout, concat=False)
            self.bn2        = BatchNorm(hidden_dim)
            self.skip_proj  = nn.Linear(dgi_embed_dim, hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            self._embed_dim = hidden_dim

        def forward(self, x, edge_index, dgi_embeds=None):
            h = F.elu(self.input_proj(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.elu(self.bn1(self.gat1(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.elu(self.bn2(self.gat2(h, edge_index)))
            if dgi_embeds is not None:
                skip = F.elu(self.skip_proj(dgi_embeds))
                h = torch.cat([h, skip], dim=-1)
            else:
                h = torch.cat([h, torch.zeros_like(h)], dim=-1)
            return self.classifier(h)

        def get_embeddings(self, x, edge_index):
            h = F.elu(self.input_proj(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.elu(self.bn1(self.gat1(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            return F.elu(self.bn2(self.gat2(h, edge_index)))

    model = _GAT().to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def load_dgi_embeddings(data: Data, device: torch.device) -> torch.Tensor:
    """Recompute frozen DGI embeddings."""
    from torch_geometric.nn import GCNConv

    ckpt = torch.load(MODELS_DIR / "dgi_encoder.pt", map_location=device, weights_only=False)

    class _GCNEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(ckpt["in_dim"],    ckpt["hidden_dim"])
            self.conv2 = GCNConv(ckpt["hidden_dim"], ckpt["embed_dim"])
            self._d    = ckpt["dropout"]

        def forward(self, x, ei):
            x = F.relu(self.conv1(x, ei))
            x = F.dropout(x, p=self._d, training=False)
            return self.conv2(x, ei)

    enc = _GCNEnc().to(device)
    enc.load_state_dict(ckpt["state_dict"])
    with torch.no_grad():
        emb = enc(data.x.to(device), data.edge_index.to(device))
    return emb


# ── Threshold Optimisation ────────────────────────────────────────────────────

@torch.no_grad()
def score_nodes(
    model:      nn.Module,
    data:       Data,
    dgi_embeds: torch.Tensor,
    device:     torch.device,
) -> np.ndarray:
    """Return P(illicit) for all nodes."""
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device), dgi_embeds)
    return F.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def optimise_threshold(
    probs:     np.ndarray,
    y_true:    np.ndarray,
    val_mask:  np.ndarray,
    n_steps:   int = 200,
) -> tuple[float, pd.DataFrame]:
    """Sweep classification thresholds on the validation set; maximise F1."""
    y_val  = y_true[val_mask & (y_true >= 0)]
    p_val  = probs[  val_mask & (y_true >= 0)]
    taus   = np.linspace(p_val.min(), p_val.max(), n_steps)

    rows, best_f1, best_tau = [], 0.0, 0.5
    for tau in taus:
        preds = (p_val > tau).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        rows.append({"threshold": tau,
                     "f1": f1,
                     "precision": precision_score(y_val, preds, zero_division=0),
                     "recall":    recall_score(y_val,    preds, zero_division=0)})
        if f1 > best_f1:
            best_f1 = f1; best_tau = tau

    return best_tau, pd.DataFrame(rows)


# ── Evaluation at Fixed Threshold ─────────────────────────────────────────────

def evaluate_at_threshold(
    probs:    np.ndarray,
    y_true:   np.ndarray,
    mask:     np.ndarray,
    tau:      float,
    label:    str,
) -> dict:
    labeled = mask & (y_true >= 0)
    yt = y_true[labeled]
    pp = probs[  labeled]
    yp = (pp > tau).astype(int)

    return {
        "method":      label,
        "threshold":   float(tau),
        "f1_illicit":  float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "f1_macro":    float(f1_score(yt, yp, average="macro", zero_division=0)),
        "precision":   float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "recall":      float(recall_score(yt,    yp, pos_label=1, zero_division=0)),
        "pr_auc":      float(average_precision_score(yt, pp))
                       if len(np.unique(yt)) > 1 else 0.0,
        "roc_auc":     float(roc_auc_score(yt, pp))
                       if len(np.unique(yt)) > 1 else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 12: End-to-End Inference")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("  Loading Elliptic graph …")
    data   = torch.load(DATA_PROC / "elliptic_graph.pt", weights_only=False)
    y_true = data.y.numpy()

    print("  Loading MixTrace GAT checkpoint …")
    ckpt   = torch.load(MODELS_DIR / "mixtrace_gat_best.pt", map_location=device, weights_only=False)
    model  = build_gat_model(ckpt, device)
    model.eval()

    print("  Recomputing DGI embeddings …")
    dgi_embeds = load_dgi_embeddings(data, device)

    # ── Full-model inference ──
    print("  Running full-model inference (DGI + GAT) …")
    probs_full = score_nodes(model, data, dgi_embeds, device)

    # ── Ablation: GAT without DGI skip ──
    print("  Ablation: GAT without DGI skip …")
    probs_no_dgi = score_nodes(model, data, None, device)

    # ── Threshold optimisation on validation set ──
    print("  Optimising threshold on validation set …")
    val_mask = data.val_mask.numpy()
    best_tau, sweep_df = optimise_threshold(probs_full, y_true, val_mask)
    print(f"  Optimal threshold: {best_tau:.4f}")

    # ── Test set evaluation ──
    test_mask = data.test_mask.numpy()
    print("\n── Test Set Results (MixTrace DGI+GAT)")
    metrics_full   = evaluate_at_threshold(probs_full,   y_true, test_mask,
                                            best_tau, "DGI+GAT (MixTrace)")
    metrics_no_dgi = evaluate_at_threshold(probs_no_dgi, y_true, test_mask,
                                            best_tau, "GAT (no DGI skip)")
    for k, v in metrics_full.items():
        print(f"  {k:<20}: {v}")

    print("\n── Ablation — GAT without DGI skip")
    for k, v in metrics_no_dgi.items():
        if isinstance(v, float):
            diff = v - metrics_full.get(k, v)
            print(f"  {k:<20}: {v:.4f}  (Δ {diff:+.4f})")

    # ── Save predictions ──
    node_ids = np.load(DATA_PROC / "elliptic_node_ids.npy")
    pred_df  = pd.DataFrame({
        "txId":          node_ids,
        "label_true":    y_true,
        "prob_illicit":  probs_full,
        "pred_illicit":  (probs_full > best_tau).astype(int),
        "prob_no_dgi":   probs_no_dgi,
        "in_test_set":   test_mask,
        "in_val_set":    val_mask,
        "in_train_set":  data.train_mask.numpy(),
    })
    pred_df.to_csv(RESULTS_DIR / "12_test_predictions.csv", index=False)
    sweep_df.to_csv(RESULTS_DIR / "12_threshold_sweep.csv",    index=False)

    final_metrics = {
        "dgi_gat_mixtrace": metrics_full,
        "gat_no_dgi":       metrics_no_dgi,
        "optimal_threshold": float(best_tau),
    }
    out_path = RESULTS_DIR / "12_final_test_metrics.json"
    out_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path.name}")

    print("\n[OK] Step 12 complete — run 13_evaluator.py next.\n")


if __name__ == "__main__":
    main()
