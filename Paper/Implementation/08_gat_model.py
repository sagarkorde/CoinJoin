"""
08_gat_model.py  |  MixTrace Pipeline — Step 8: GAT Model Architecture
=======================================================================
Defines the MixTrace Graph Attention Network (GAT) classifier.

Architecture
------------
  Input  → Pre-proj FC  →  GAT Layer 1 (8 heads)  →  GAT Layer 2 (1 head)
         → Skip connection from DGI encoder output → Classifier head

Key Design Choices
------------------
  1. Multi-head attention (8 heads in hidden layer) captures diverse neighbourhood
     aggregation patterns — particularly important for heterogeneous transaction graphs
     where CoinJoin nodes co-occur with normal P2PKH/P2WPKH transactions.

  2. Residual skip connection from the DGI pre-trained encoder output to the final
     classification layer preserves self-supervised representations and mitigates
     catastrophic forgetting during supervised fine-tuning.

  3. BatchNorm after each GAT layer stabilises training under severe class imbalance.

  4. ELU activation (instead of ReLU) avoids dead neurons in sparse financial graphs.

Usage (as imported by 09_training_pipeline.py)
----------------------------------------------
    from 08_gat_model import MixTraceGAT, load_pretrained_encoder
    from 07_dgi_pretrainer import GCNEncoder

    encoder = load_pretrained_encoder(device)
    model   = MixTraceGAT(in_dim=169, hidden_dim=128, n_heads=8, n_classes=2,
                           dgi_embed_dim=128, dropout=0.5)

Run standalone to print a model summary:
    python 08_gat_model.py
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DATA_PROC  = HERE / "data" / "processed"


# ── DGI Encoder (import-compatible re-export) ─────────────────────────────────

def load_pretrained_encoder(device: torch.device):
    """Load the GCN encoder trained in Step 7."""
    checkpoint = torch.load(MODELS_DIR / "dgi_encoder.pt", map_location=device, weights_only=False)
    encoder = _build_gcn_encoder(
        in_dim=checkpoint["in_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        out_dim=checkpoint["embed_dim"],
        dropout=checkpoint["dropout"],
    )
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()
    return encoder.to(device)


def _build_gcn_encoder(in_dim, hidden_dim, out_dim, dropout):
    """Inline GCNEncoder definition to avoid circular import."""
    from torch_geometric.nn import GCNConv

    class _GCNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1   = GCNConv(in_dim,    hidden_dim)
            self.conv2   = GCNConv(hidden_dim, out_dim)
            self._dropout = dropout

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self._dropout, training=self.training)
            return self.conv2(x, edge_index)

    return _GCNEncoder()


# ── MixTrace GAT Model ────────────────────────────────────────────────────────

class MixTraceGAT(nn.Module):
    """
    Graph Attention Network with DGI skip connection for illicit transaction detection.

    Parameters
    ----------
    in_dim       : input feature dimension (165 base + 4 novel = 169)
    hidden_dim   : hidden embedding dimension for GAT layers
    n_heads      : number of attention heads in the first GAT layer
    n_classes    : number of output classes (2: licit / illicit)
    dgi_embed_dim: output dimension of the DGI encoder (for skip connection)
    dropout      : dropout probability
    """

    def __init__(
        self,
        in_dim:       int,
        hidden_dim:   int   = 128,
        n_heads:      int   = 8,
        n_classes:    int   = 2,
        dgi_embed_dim: int  = 128,
        dropout:      float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GAT Layer 1: multi-head, concat → hidden_dim * n_heads
        self.gat1  = GATConv(hidden_dim, hidden_dim, heads=n_heads,
                              dropout=dropout, concat=True)
        self.bn1   = BatchNorm(hidden_dim * n_heads)

        # GAT Layer 2: single head, mean-pooled → hidden_dim
        self.gat2  = GATConv(hidden_dim * n_heads, hidden_dim, heads=1,
                              dropout=dropout, concat=False)
        self.bn2   = BatchNorm(hidden_dim)

        # Skip connection projection (DGI embed → hidden_dim)
        self.skip_proj = nn.Linear(dgi_embed_dim, hidden_dim)

        # Classification head: (GAT output + skip) → n_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        dgi_embeds:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [N, in_dim]   node feature matrix
        edge_index : [2, E]        edge index (bidirectional)
        dgi_embeds : [N, d_dgi]    frozen DGI encoder output (optional)

        Returns
        -------
        logits : [N, n_classes]
        """
        # Input projection
        h = F.elu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GAT Layer 1
        h = self.gat1(h, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GAT Layer 2
        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)

        # DGI skip connection
        if dgi_embeds is not None:
            skip = F.elu(self.skip_proj(dgi_embeds))
            h = torch.cat([h, skip], dim=-1)   # [N, hidden_dim * 2]
        else:
            # Zero-pad skip connection when DGI embeddings are unavailable
            h = torch.cat([h, torch.zeros_like(h)], dim=-1)

        return self.classifier(h)   # [N, n_classes]

    def get_embeddings(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return post-GAT2 node embeddings (before classifier head)."""
        h = F.elu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn1(self.gat1(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.elu(self.bn2(self.gat2(h, edge_index)))   # [N, hidden_dim]


# ── Model Summary ─────────────────────────────────────────────────────────────

def print_model_summary(model: nn.Module, in_dim: int) -> None:
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  MixTraceGAT Architecture")
    print(f"  {'─'*40}")
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, GATConv)):
            params = sum(p.numel() for p in mod.parameters())
            print(f"  {name:<30} {type(mod).__name__:<12} params={params:,}")
    print(f"  {'─'*40}")
    print(f"  Total parameters   : {total:,}")
    print(f"  Trainable params   : {trainable:,}")
    print(f"  Input feature dim  : {in_dim}")


# ── Standalone Test ───────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 8: GAT Model Definition")
    print("=" * 60)

    # Determine input dimension from the saved graph
    graph_path = DATA_PROC / "elliptic_graph.pt"
    if graph_path.exists():
        data   = torch.load(graph_path, map_location="cpu", weights_only=False)
        in_dim = data.num_node_features
    else:
        in_dim = 169   # default: 165 base + 4 novel features
        print("  ⚠  Graph not found — using default in_dim=169")

    model = MixTraceGAT(
        in_dim       = in_dim,
        hidden_dim   = 128,
        n_heads      = 8,
        n_classes    = 2,
        dgi_embed_dim = 128,
        dropout      = 0.5,
    )
    print_model_summary(model, in_dim)

    # Forward pass smoke test
    if graph_path.exists():
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        print(f"\n  Smoke test — logits shape: {list(logits.shape)}")
        print(f"  Expected               : [{data.num_nodes}, 2]")
        print("\n[OK] Model definition verified.\n")
    else:
        print("\n  Run 06_graph_builder.py first to enable smoke test.\n")


if __name__ == "__main__":
    main()
