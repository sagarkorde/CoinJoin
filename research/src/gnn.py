"""
Graph construction and training loops for the GNN experiments.

Two DGI regimes are supported, because Reviewer 1 (comment 2) noted that
pre-training on all 203,769 nodes is transductive and is inconsistent with a
claim that the chronological protocol prevents temporal leakage:

  transductive : DGI sees the whole graph, including validation and test nodes.
  inductive    : DGI sees only the training-period subgraph. Embeddings for
                 later nodes are produced afterwards by a forward pass of the
                 frozen encoder, so no future node influences the weights.

Both are run and reported side by side rather than one being presented as the
protocol.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from config import TRAIN_END, VAL_END
from models import (DGI, GAT, FrozenDGIGATFusion, GCNBaseline,
                    GraphSAGEBaseline)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame,
                feature_cols: list[str]) -> Data:
    """
    PyG graph over all Elliptic nodes.

    Features are standardised with statistics from labelled training nodes
    only. Edges are made bidirectional for message passing. Masks mark
    labelled nodes within each chronological period.
    """
    idx = {int(t): i for i, t in enumerate(nodes["txId"].to_numpy())}
    X = nodes[feature_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = nodes["label"].to_numpy()
    ts = nodes["time_step"].to_numpy()
    has_label = y != -1

    train_mask = (ts <= TRAIN_END) & has_label
    val_mask = (ts > TRAIN_END) & (ts <= VAL_END) & has_label
    test_mask = (ts > VAL_END) & has_label

    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    src = edges["txId1"].map(idx).to_numpy()
    dst = edges["txId2"].map(idx).to_numpy()
    ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])

    return Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(np.where(y == -1, 0, y), dtype=torch.long),
        edge_index=torch.tensor(ei, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
        time_step=torch.tensor(ts, dtype=torch.long),
    )


def train_dgi(data: Data, regime: str, device: str, seed: int,
              hid_dim: int = 128, epochs: int = 200, lr: float = 1e-3,
              patience: int = 30, verbose: bool = False) -> tuple[torch.Tensor, dict]:
    """
    Train DGI and return (frozen embeddings for all nodes, run info).

    regime == "inductive" restricts pre-training to the training-period
    subgraph; embeddings for later nodes come from a forward pass afterwards.
    """
    set_seed(seed)
    t0 = time.time()

    if regime == "inductive":
        keep = (data.time_step <= TRAIN_END)
        sub_idx = keep.nonzero(as_tuple=True)[0]
        remap = -torch.ones(data.num_nodes, dtype=torch.long)
        remap[sub_idx] = torch.arange(sub_idx.numel())
        ei = data.edge_index
        emask = keep[ei[0]] & keep[ei[1]]
        ei_sub = remap[ei[:, emask]]
        x_fit, ei_fit = data.x[sub_idx].to(device), ei_sub.to(device)
    elif regime == "transductive":
        x_fit, ei_fit = data.x.to(device), data.edge_index.to(device)
    else:
        raise ValueError(regime)

    model = DGI(data.num_node_features, hid_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, bad, best_state = float("inf"), 0, None
    losses = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = model.loss(x_fit, ei_fit)
        loss.backward()
        opt.step()
        lv = float(loss.item())
        losses.append(lv)
        if lv < best - 1e-5:
            best, bad = lv, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"      dgi[{regime}] ep{ep:3d} loss {lv:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        emb = model.encoder(data.x.to(device), data.edge_index.to(device)).detach()

    return emb, {"regime": regime, "epochs_run": len(losses),
                 "best_loss": best, "seconds": time.time() - t0}


def _class_weights(y: torch.Tensor, mask: torch.Tensor, device: str):
    """Inverse-frequency weights; the illicit class is ~10 % of labelled nodes."""
    yt = y[mask]
    n_pos = float((yt == 1).sum())
    n_neg = float((yt == 0).sum())
    total = n_pos + n_neg
    w = torch.tensor([total / (2 * max(n_neg, 1)), total / (2 * max(n_pos, 1))],
                     dtype=torch.float, device=device)
    return w


def train_classifier(data: Data, kind: str, device: str, seed: int,
                     dgi_emb: torch.Tensor | None = None,
                     hid_dim: int = 64, heads: int = 8, dropout: float = 0.3,
                     epochs: int = 300, lr: float = 5e-3, weight_decay: float = 5e-4,
                     patience: int = 40, verbose: bool = False,
                     return_model: bool = False) -> dict:
    """
    Train a node classifier and return scores plus timing.

    kind is one of: "fusion", "gat_zero_dgi", "gat", "gcn", "sage".
    Model selection uses validation PR-AUC, which is the appropriate criterion
    under heavy class imbalance; the decision threshold is chosen later, on
    validation scores, by the caller.
    """
    from sklearn.metrics import average_precision_score

    set_seed(seed)
    t0 = time.time()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    y = data.y.to(device)
    trm = data.train_mask.to(device)
    vam = data.val_mask.to(device)

    in_dim = data.num_node_features
    if kind == "fusion":
        model = FrozenDGIGATFusion(in_dim, dgi_emb.size(1), hid_dim, heads,
                                   dropout, zero_dgi=False).to(device)
    elif kind == "gat_zero_dgi":
        model = FrozenDGIGATFusion(in_dim, 0, hid_dim, heads, dropout,
                                   zero_dgi=True).to(device)
    elif kind == "gat":
        model = GAT(in_dim, hid_dim, heads, dropout).to(device)
    elif kind == "gcn":
        model = GCNBaseline(in_dim, hid_dim, dropout).to(device)
    elif kind == "sage":
        model = GraphSAGEBaseline(in_dim, hid_dim, dropout).to(device)
    else:
        raise ValueError(kind)

    needs_emb = kind in ("fusion", "gat_zero_dgi")
    emb = dgi_emb.to(device) if (needs_emb and dgi_emb is not None) else None

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = _class_weights(data.y, data.train_mask, device)

    def fwd():
        return model(x, ei, emb) if needs_emb else model(x, ei)

    best_ap, bad, best_state = -1.0, 0, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = fwd()
        loss = F.cross_entropy(out[trm], y[trm], weight=w)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = F.softmax(fwd(), dim=1)[:, 1]
            ap = average_precision_score(y[vam].cpu().numpy(),
                                         p[vam].cpu().numpy())
        if ap > best_ap + 1e-5:
            best_ap, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 50 == 0:
            print(f"      {kind} ep{ep:3d} loss {loss.item():.4f} val-AP {ap:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = F.softmax(fwd(), dim=1)[:, 1].cpu().numpy()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.startswith("cuda") else float("nan"))
    out = {"scores": scores, "best_val_ap": float(best_ap),
           "seconds": time.time() - t0, "n_params": int(n_params),
           "peak_mem_mb": float(peak), "epochs_run": ep + 1}
    if return_model:
        out["model"] = model
    return out
