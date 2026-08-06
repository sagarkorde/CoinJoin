"""
07_dgi_pretrainer.py  |  MixTrace Pipeline — Step 7: DGI Self-Supervised Pre-training
=======================================================================================
Implements Deep Graph Infomax (DGI) for self-supervised pre-training of a GCN encoder
on the full Elliptic transaction graph (including 157,205 unlabeled nodes).

Algorithm (Velickovic et al., 2019)
------------------------------------
  For each training step:
    1. Encode the uncorrupted graph: z_i = encoder(x_i, A)
    2. Compute a global graph summary: s = σ(MEAN(z_i))
    3. Corrupt node features by row-shuffling: x̃ = shuffle(x)
    4. Encode the corrupted graph: z̃_i = encoder(x̃_i, A)
    5. Score positives (z_i, s) and negatives (z̃_i, s) using a bilinear discriminator
    6. Minimise binary cross-entropy: L = -E[log D(z, s)] - E[log(1 - D(z̃, s))]

The encoder is a two-layer GCN.  After pre-training, encoder weights are saved
and loaded by the GAT fine-tuning stage (Step 9) as an initialisation advantage.

Run
---
    python 07_dgi_pretrainer.py

Reads   : data/processed/elliptic_graph.pt  (Step 6)
Outputs : models/dgi_encoder.pt
          results/07_dgi_training_loss.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"
MODELS_DIR  = HERE / "models"

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_DIM  = 256
EMBED_DIM   = 128
EPOCHS      = 300
LR          = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT     = 0.3
RANDOM_SEED = 42


# ── Model Components ──────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """Two-layer GCN encoder for DGI pre-training and GAT initialisation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv1   = GCNConv(in_dim,    hidden_dim)
        self.conv2   = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x   # [N, out_dim]


class Discriminator(nn.Module):
    """
    Bilinear discriminator: scores (node_embedding, global_summary) pairs.

    D(z, s) = z^T · W · s  (learnable weight matrix W ∈ R^{d×d})
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z       : [N, d]  node embeddings
        summary : [d]     global graph summary

        Returns
        -------
        scores  : [N]
        """
        Ws = torch.matmul(self.weight, summary.unsqueeze(-1))   # [d, 1]
        return torch.matmul(z, Ws).squeeze(-1)                   # [N]


class DGIModel(nn.Module):
    """Full DGI model: GCN encoder + bilinear discriminator."""

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.encoder      = GCNEncoder(in_dim, hidden_dim, embed_dim, dropout)
        self.discriminator = Discriminator(embed_dim)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        x_neg:      torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns positive and negative discriminator scores.
        """
        z_pos = self.encoder(x,     edge_index)   # [N, d]
        z_neg = self.encoder(x_neg, edge_index)   # [N, d]

        # Global summary: sigmoid-activated mean of positive embeddings
        summary = torch.sigmoid(z_pos.mean(dim=0))   # [d]

        pos_scores = self.discriminator(z_pos, summary)   # [N]
        neg_scores = self.discriminator(z_neg, summary)   # [N]

        return pos_scores, neg_scores


# ── Corruption Function ───────────────────────────────────────────────────────

def corrupt_features(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Row-shuffle node feature matrix to break graph structure."""
    perm = torch.randperm(x.size(0), generator=rng)
    return x[perm]


# ── Training ──────────────────────────────────────────────────────────────────

def train_dgi(
    data: Data,
    device: torch.device,
) -> tuple[GCNEncoder, list[float]]:
    torch.manual_seed(RANDOM_SEED)
    rng = torch.Generator().manual_seed(RANDOM_SEED)

    in_dim = data.num_node_features
    model  = DGIModel(in_dim, HIDDEN_DIM, EMBED_DIM, DROPOUT).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    N          = x.size(0)

    # BCE labels: positives → 1, negatives → 0
    pos_labels = torch.ones(N,  device=device)
    neg_labels = torch.zeros(N, device=device)
    criterion  = nn.BCEWithLogitsLoss()

    loss_history = []
    best_loss    = float("inf")
    best_state   = None

    for epoch in tqdm(range(1, EPOCHS + 1), desc="  DGI training"):
        model.train()
        optim.zero_grad()

        x_neg = corrupt_features(x, rng)
        pos_scores, neg_scores = model(x, edge_index, x_neg)

        loss = criterion(pos_scores, pos_labels) + criterion(neg_scores, neg_labels)
        loss.backward()
        optim.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = {k: v.clone() for k, v in model.encoder.state_dict().items()}

        if epoch % 50 == 0:
            tqdm.write(f"  Epoch {epoch:>4}  Loss: {loss_val:.6f}")

    model.encoder.load_state_dict(best_state)
    print(f"\n  Best DGI loss: {best_loss:.6f}")
    return model.encoder, loss_history


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 7: DGI Self-Supervised Pre-training")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("  Loading Elliptic graph …")
    data = torch.load(DATA_PROC / "elliptic_graph.pt", weights_only=False)
    print(f"  Nodes: {data.num_nodes:,}  Edges: {data.num_edges:,}  "
          f"Features: {data.num_node_features}")

    print(f"\n  Training DGI for {EPOCHS} epochs …")
    print(f"  hidden={HIDDEN_DIM}  embed={EMBED_DIM}  lr={LR}  dropout={DROPOUT}")

    encoder, loss_history = train_dgi(data, device)

    # Save encoder weights
    encoder_path = MODELS_DIR / "dgi_encoder.pt"
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "in_dim":     data.num_node_features,
            "hidden_dim": HIDDEN_DIM,
            "embed_dim":  EMBED_DIM,
            "dropout":    DROPOUT,
        },
        encoder_path,
    )
    print(f"  Saved encoder → {encoder_path.relative_to(HERE)}")

    # Save loss curve
    loss_df = pd.DataFrame({"epoch": range(1, EPOCHS + 1), "loss": loss_history})
    loss_path = RESULTS_DIR / "07_dgi_training_loss.csv"
    loss_df.to_csv(loss_path, index=False)
    print(f"  Saved loss curve → {loss_path.name}")

    print("\n[OK] Step 7 complete — run 08_gat_model.py next.\n")


if __name__ == "__main__":
    main()
