"""
Graph models for the MixTrace revision.

Naming note (Reviewer 1, comment 10). The round-1 manuscript described the
architecture as "DGI pre-training followed by GAT fine-tuning". That is not
what the code does. The DGI encoder is trained self-supervised, then frozen;
the GAT is trained separately and supervised; the two representations are
concatenated before the classifier. No DGI weight is updated by the supervised
objective and the GAT is not initialised from DGI. The accurate description is
*fusion of a frozen self-supervised representation with a supervised one*, and
the class is named accordingly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class DGIEncoder(nn.Module):
    """Two-layer GCN encoder used as the DGI backbone."""

    def __init__(self, in_dim: int, hid_dim: int = 128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.act = nn.PReLU(hid_dim)

    def forward(self, x, edge_index):
        h = self.act(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


class DGI(nn.Module):
    """
    Deep Graph Infomax.

    Positive samples are the real node embeddings; negatives come from a
    corrupted graph in which the feature rows are permuted while the topology
    is held fixed. The bilinear discriminator scores each node against a
    summary vector of the whole graph.
    """

    def __init__(self, in_dim: int, hid_dim: int = 128):
        super().__init__()
        self.encoder = DGIEncoder(in_dim, hid_dim)
        self.weight = nn.Parameter(torch.empty(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.weight)

    def summary(self, h):
        return torch.sigmoid(h.mean(dim=0))

    def discriminate(self, h, s):
        return h @ (self.weight @ s)

    def forward(self, x, edge_index):
        h_pos = self.encoder(x, edge_index)
        perm = torch.randperm(x.size(0), device=x.device)
        h_neg = self.encoder(x[perm], edge_index)
        s = self.summary(h_pos)
        return h_pos, h_neg, s

    def loss(self, x, edge_index):
        h_pos, h_neg, s = self.forward(x, edge_index)
        logits_pos = self.discriminate(h_pos, s)
        logits_neg = self.discriminate(h_neg, s)
        ones = torch.ones_like(logits_pos)
        zeros = torch.zeros_like(logits_neg)
        return (F.binary_cross_entropy_with_logits(logits_pos, ones)
                + F.binary_cross_entropy_with_logits(logits_neg, zeros)) * 0.5


class GAT(nn.Module):
    """Plain supervised two-layer GAT classifier."""

    def __init__(self, in_dim: int, hid_dim: int = 64, heads: int = 8,
                 dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, hid_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hid_dim * heads, hid_dim, heads=1, concat=False,
                             dropout=dropout)
        self.classifier = nn.Linear(hid_dim, n_classes)

    def embed(self, x, edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.elu(self.conv2(h, edge_index))

    def forward(self, x, edge_index):
        return self.classifier(self.embed(x, edge_index))


class FrozenDGIGATFusion(nn.Module):
    """
    Fusion of a frozen DGI representation with a supervised GAT representation.

    The DGI embedding is supplied precomputed and detached; only the GAT and
    the classifier head carry gradients. `zero_dgi` replaces the frozen
    representation with zeros from the start of training, which is the correct
    way to ablate the DGI contribution (Reviewer 1, comment 9): the model is
    genuinely trained without it rather than having it removed at inference.
    """

    def __init__(self, in_dim: int, dgi_dim: int, hid_dim: int = 64,
                 heads: int = 8, dropout: float = 0.3, n_classes: int = 2,
                 zero_dgi: bool = False):
        super().__init__()
        self.zero_dgi = zero_dgi
        self.dropout = dropout
        self.gat = GAT(in_dim, hid_dim, heads, dropout, n_classes)
        fused = hid_dim + (0 if zero_dgi else dgi_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused, hid_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, n_classes),
        )

    def forward(self, x, edge_index, dgi_emb):
        h_gat = self.gat.embed(x, edge_index)
        if self.zero_dgi:
            fused = h_gat
        else:
            fused = torch.cat([h_gat, dgi_emb.detach()], dim=1)
        return self.classifier(fused)


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE reference point for the model-family comparison."""

    def __init__(self, in_dim: int, hid_dim: int = 64, dropout: float = 0.3,
                 n_classes: int = 2):
        from torch_geometric.nn import SAGEConv
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.classifier = nn.Linear(hid_dim, n_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.classifier(h)


class GCNBaseline(nn.Module):
    """GCN reference point for the model-family comparison."""

    def __init__(self, in_dim: int, hid_dim: int = 64, dropout: float = 0.3,
                 n_classes: int = 2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.classifier = nn.Linear(hid_dim, n_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.classifier(h)
