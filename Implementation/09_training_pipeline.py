"""
09_training_pipeline.py  |  MixTrace Pipeline — Step 9: GAT Fine-tuning
=========================================================================
Supervised fine-tuning of the MixTrace GAT classifier on the Elliptic Dataset
using the DGI pre-trained encoder as a frozen skip connection.

Training Strategy
-----------------
  1. Load the Elliptic graph and the DGI encoder weights (Step 7).
  2. Pre-compute frozen DGI node embeddings (no gradient through encoder).
  3. Initialise MixTraceGAT (Step 8).
  4. Handle class imbalance via:
       a. Class-weighted cross-entropy loss  (inverse frequency weighting)
       b. ADASYN oversampling on the labeled node feature matrix for diagnostic
          comparison — results saved but GNN training uses loss weighting only.
  5. Train with Adam + ReduceLROnPlateau scheduler.
  6. Early stopping on validation F1 (illicit class) with patience=20.
  7. Save the best checkpoint and final training curves.

Run
---
    python 09_training_pipeline.py

Reads   : data/processed/elliptic_graph.pt  (Step 6)
          models/dgi_encoder.pt              (Step 7)
Outputs : models/mixtrace_gat_best.pt
          results/09_training_curves.csv
          results/09_gat_test_metrics.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import ADASYN
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"
MODELS_DIR  = HERE / "models"

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_DIM    = 128
N_HEADS       = 8
DGI_EMBED_DIM = 128
DROPOUT       = 0.5
LR            = 5e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 500
PATIENCE      = 20
RANDOM_SEED   = 42

ADASYN_SEED   = 42


# ── DGI Encoder Loading ───────────────────────────────────────────────────────

def load_dgi_encoder(device: torch.device):
    """Load and freeze DGI encoder for use as a skip connection."""
    from torch_geometric.nn import GCNConv

    class _GCNEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, dropout):
            super().__init__()
            self.conv1    = GCNConv(in_dim,    hidden_dim)
            self.conv2    = GCNConv(hidden_dim, out_dim)
            self._dropout = dropout

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self._dropout, training=self.training)
            return self.conv2(x, edge_index)

    ckpt    = torch.load(MODELS_DIR / "dgi_encoder.pt", map_location=device, weights_only=False)
    encoder = _GCNEncoder(
        in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"],
        out_dim=ckpt["embed_dim"], dropout=ckpt["dropout"],
    )
    encoder.load_state_dict(ckpt["state_dict"])
    for p in encoder.parameters():
        p.requires_grad = False   # freeze DGI encoder
    return encoder.to(device).eval()


# ── Class Weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights for labeled training nodes."""
    y_train  = y[mask & (y >= 0)]
    n_ill    = (y_train == 1).sum().item()
    n_lic    = (y_train == 0).sum().item()
    n_total  = n_ill + n_lic
    w_ill    = n_total / (2.0 * n_ill) if n_ill > 0 else 1.0
    w_lic    = n_total / (2.0 * n_lic) if n_lic > 0 else 1.0
    return torch.tensor([w_lic, w_ill], dtype=torch.float)


# ── ADASYN (diagnostic) ───────────────────────────────────────────────────────

def apply_adasyn(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply ADASYN oversampling to the labeled feature matrix.

    Used diagnostically: the oversampled counts are reported but the GNN
    training itself uses class-weighted loss on the original graph structure.
    """
    ada = ADASYN(random_state=ADASYN_SEED, n_neighbors=5)
    try:
        X_res, y_res = ada.fit_resample(X, y)
        return X_res, y_res
    except Exception:
        # ADASYN may fail if minority class is very small; fallback to original
        return X, y


# ── Training & Evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:       nn.Module,
    data:        Data,
    dgi_embeds:  torch.Tensor,
    mask:        torch.Tensor,
    device:      torch.device,
) -> dict:
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device), dgi_embeds)
    probs  = F.softmax(logits, dim=-1)[:, 1]   # P(illicit)

    # Restrict to labeled nodes in the mask
    labeled_mask = mask & (data.y >= 0)
    y_true = data.y[labeled_mask].cpu().numpy()
    y_prob = probs[labeled_mask].cpu().numpy()
    y_pred = (y_prob > 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        return {"f1_illicit": 0.0, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "f1_illicit": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pr_auc":     average_precision_score(y_true, y_prob),
        "precision":  precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall":     recall_score(y_true,    y_pred, pos_label=1, zero_division=0),
    }


def train(
    model:       nn.Module,
    data:        Data,
    dgi_embeds:  torch.Tensor,
    class_weights: torch.Tensor,
    device:      torch.device,
) -> tuple[nn.Module, list[dict]]:
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    train_mask = data.train_mask.to(device)

    # Labeled training nodes
    labeled_train = train_mask & (y >= 0)

    best_val_f1  = 0.0
    patience_cnt = 0
    best_state   = None
    history      = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index, dgi_embeds)
        loss   = criterion(logits[labeled_train], y[labeled_train])
        loss.backward()
        optimizer.step()

        # Validation metrics every 5 epochs
        if epoch % 5 == 0:
            val_metrics  = evaluate(model, data, dgi_embeds, data.val_mask, device)
            train_metrics = evaluate(model, data, dgi_embeds, data.train_mask, device)

            scheduler.step(val_metrics["f1_illicit"])

            record = {
                "epoch":          epoch,
                "train_loss":     float(loss.item()),
                "train_f1":       train_metrics["f1_illicit"],
                "val_f1":         val_metrics["f1_illicit"],
                "val_pr_auc":     val_metrics["pr_auc"],
                "lr":             optimizer.param_groups[0]["lr"],
            }
            history.append(record)

            if val_metrics["f1_illicit"] > best_val_f1:
                best_val_f1  = val_metrics["f1_illicit"]
                patience_cnt = 0
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if epoch % 50 == 0:
                print(f"  Epoch {epoch:>4}  loss={loss.item():.4f}  "
                      f"val_F1={val_metrics['f1_illicit']:.4f}  "
                      f"val_PR-AUC={val_metrics['pr_auc']:.4f}")

            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  "
                      f"(best val F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 9: GAT Fine-tuning")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Load graph ──
    print("  Loading Elliptic graph …")
    data   = torch.load(DATA_PROC / "elliptic_graph.pt", weights_only=False)
    in_dim = data.num_node_features
    print(f"  Nodes={data.num_nodes:,}  Edges={data.num_edges:,}  Features={in_dim}")

    # ── DGI encoder ──
    print("  Loading DGI encoder (frozen) …")
    encoder    = load_dgi_encoder(device)
    dgi_x      = data.x.to(device)
    dgi_ei     = data.edge_index.to(device)
    with torch.no_grad():
        dgi_embeds = encoder(dgi_x, dgi_ei)   # [N, DGI_EMBED_DIM]
    print(f"  DGI embeddings: {list(dgi_embeds.shape)}")

    # ── ADASYN diagnostic ──
    print("\n  ADASYN oversampling diagnostic …")
    X_lab = np.load(DATA_PROC / "elliptic_X_labeled.npy")
    y_lab = np.load(DATA_PROC / "elliptic_y_labeled.npy")
    X_res, y_res = apply_adasyn(X_lab, y_lab)
    print(f"  Before: {len(y_lab):,}  ill={int((y_lab==1).sum())}  "
          f"lic={int((y_lab==0).sum())}")
    print(f"  After : {len(y_res):,}  ill={int((y_res==1).sum())}  "
          f"lic={int((y_res==0).sum())}")

    # ── Class weights ──
    class_weights = compute_class_weights(data.y, data.train_mask)
    print(f"\n  Class weights: licit={class_weights[0]:.3f}  "
          f"illicit={class_weights[1]:.3f}")

    # ── Build MixTrace GAT (defined inline to avoid cross-file import) ──
    class _MixTraceGAT(nn.Module):
        """Inline to avoid module name import issues."""
        def __init__(self, in_dim, hidden_dim, n_heads, n_classes, dgi_embed_dim, dropout):
            super().__init__()
            from torch_geometric.nn import BatchNorm, GATConv
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
                nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x, edge_index, dgi_embeds=None):
            import torch.nn.functional as F2
            h = F2.elu(self.input_proj(x))
            h = F2.dropout(h, p=self.dropout, training=self.training)
            h = F2.elu(self.bn1(self.gat1(h, edge_index)))
            h = F2.dropout(h, p=self.dropout, training=self.training)
            h = F2.elu(self.bn2(self.gat2(h, edge_index)))
            if dgi_embeds is not None:
                skip = F2.elu(self.skip_proj(dgi_embeds))
                h = torch.cat([h, skip], dim=-1)
            else:
                h = torch.cat([h, torch.zeros_like(h)], dim=-1)
            return self.classifier(h)

    model = _MixTraceGAT(
        in_dim=in_dim, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS,
        n_classes=2, dgi_embed_dim=DGI_EMBED_DIM, dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  MixTraceGAT — total parameters: {total_params:,}")
    print(f"  Training for up to {EPOCHS} epochs  (patience={PATIENCE}) …\n")

    model, history = train(model, data, dgi_embeds, class_weights, device)

    # ── Test evaluation ──
    print("\n  Evaluating on test set …")
    test_metrics = evaluate(model, data, dgi_embeds, data.test_mask, device)
    test_metrics["method"] = "DGI+GAT (MixTrace)"

    print(f"  F1 (illicit) : {test_metrics['f1_illicit']:.4f}")
    print(f"  PR-AUC       : {test_metrics['pr_auc']:.4f}")
    print(f"  Precision    : {test_metrics['precision']:.4f}")
    print(f"  Recall       : {test_metrics['recall']:.4f}")

    # ── Save ──
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": in_dim,
         "hidden_dim": HIDDEN_DIM, "n_heads": N_HEADS,
         "dgi_embed_dim": DGI_EMBED_DIM, "dropout": DROPOUT},
        MODELS_DIR / "mixtrace_gat_best.pt",
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / "09_training_curves.csv", index=False)

    metrics_path = RESULTS_DIR / "09_gat_test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print(f"\n  Saved model → models/mixtrace_gat_best.pt")
    print(f"  Saved curves → results/09_training_curves.csv")

    print("\n[OK] Step 9 complete — run 10_taproot_analyzer.py next.\n")


if __name__ == "__main__":
    main()
