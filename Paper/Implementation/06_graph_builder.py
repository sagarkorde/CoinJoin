"""
06_graph_builder.py  |  MixTrace Pipeline — Step 6: Graph Construction
=======================================================================
Builds a PyTorch Geometric (PyG) Data object from the Elliptic Dataset.

Graph Structure
---------------
  Nodes : all 203,769 transactions with 165-dimensional scaled features
          (local features + aggregated neighbourhood features)
  Edges : directed edges from the Elliptic edge list (234,355 edges)
          converted to bi-directional for message passing
  Labels: -1 = unknown, 0 = licit, 1 = illicit

Additionally, if the novel Elliptic features from Step 3 are available
(illicit_neighbor_ratio, shortest_path_to_illicit, graph_in_degree,
graph_out_degree), they are appended to the feature matrix to form a
169-dimensional augmented feature vector.

Outputs
-------
  data/processed/elliptic_graph.pt   — serialised PyG Data object
  data/processed/elliptic_txid_to_idx.json — txId → node-index mapping
  results/06_graph_stats.txt

Run
---
    python 06_graph_builder.py

Reads   : data/processed/  (Steps 2–3 output)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"

NOVEL_FEAT_COLS = [
    "illicit_neighbor_ratio",
    "shortest_path_to_illicit",
    "graph_in_degree",
    "graph_out_degree",
]


# ── Node Features & Labels ────────────────────────────────────────────────────

def build_node_tensors(
    nodes_raw: pd.DataFrame,
    X_all: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """
    Assemble the node feature matrix and label tensor.

    If the engineered Elliptic features exist (from Step 3), append them
    to the pre-scaled feature matrix to form an augmented representation.

    Returns
    -------
    x         : FloatTensor [N, F]
    y         : LongTensor  [N]   (-1 = unknown)
    txid_map  : {txId: node_index}
    """
    eng_path = DATA_PROC / "elliptic_features_engineered.parquet"
    if eng_path.exists():
        eng = pd.read_parquet(eng_path)[["txId"] + NOVEL_FEAT_COLS].copy()
        eng[NOVEL_FEAT_COLS] = eng[NOVEL_FEAT_COLS].fillna(0).astype(np.float32)

        # Align with nodes_raw ordering
        eng = nodes_raw[["txId"]].merge(eng, on="txId", how="left").fillna(0)
        extra = eng[NOVEL_FEAT_COLS].values.astype(np.float32)
        X = np.hstack([X_all, extra])
        print(f"  Feature dim (with novel): {X.shape[1]}")
    else:
        X = X_all
        print(f"  Feature dim (base only) : {X.shape[1]}")
        print("  ⚠  Run 03_feature_engineer.py first for augmented features.")

    # Label tensor (-1 for unlabeled)
    y_all   = np.load(DATA_PROC / "elliptic_y_all.npy")
    txid_to_idx = {int(txid): i for i, txid in enumerate(nodes_raw["txId"].values)}

    x = torch.tensor(X,     dtype=torch.float)
    y = torch.tensor(y_all, dtype=torch.long)

    return x, y, txid_to_idx


# ── Edge Index ────────────────────────────────────────────────────────────────

def build_edge_index(
    edges: pd.DataFrame,
    txid_to_idx: dict[int, int],
) -> torch.Tensor:
    """
    Convert txId-based edge list to sequential node-index edge_index.

    Edges are made bidirectional (undirected message passing) by adding
    reverse edges, following the convention in Elliptic GNN literature.
    """
    src, dst = [], []
    skipped  = 0
    for row in edges.itertuples(index=False):
        u = txid_to_idx.get(int(row.txId1))
        v = txid_to_idx.get(int(row.txId2))
        if u is None or v is None:
            skipped += 1
            continue
        src.append(u); dst.append(v)   # forward
        src.append(v); dst.append(u)   # reverse (bidirectional)

    if skipped:
        print(f"  Skipped {skipped:,} edges (txId not in node list)")

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


# ── Masks ─────────────────────────────────────────────────────────────────────

def build_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_train = np.load(DATA_PROC / "elliptic_mask_train.npy")
    mask_val   = np.load(DATA_PROC / "elliptic_mask_val.npy")
    mask_test  = np.load(DATA_PROC / "elliptic_mask_test.npy")
    return (
        torch.tensor(mask_train, dtype=torch.bool),
        torch.tensor(mask_val,   dtype=torch.bool),
        torch.tensor(mask_test,  dtype=torch.bool),
    )


# ── Statistics ────────────────────────────────────────────────────────────────

def save_stats(data: Data, txid_to_idx: dict) -> None:
    n = data.num_nodes
    e = data.num_edges
    f = data.num_node_features
    ill  = int((data.y ==  1).sum())
    lic  = int((data.y ==  0).sum())
    unk  = int((data.y == -1).sum())
    tr   = int(data.train_mask.sum())
    val  = int(data.val_mask.sum())
    te   = int(data.test_mask.sum())

    lines = [
        "=" * 65,
        "MixTrace — Graph Statistics  (Step 6)",
        "=" * 65, "",
        f"  Nodes          : {n:,}",
        f"  Edges (bidir.) : {e:,}",
        f"  Feature dim    : {f}",
        f"  Illicit        : {ill:,}",
        f"  Licit          : {lic:,}",
        f"  Unknown        : {unk:,}",
        "",
        f"  Train mask     : {tr:,}  nodes",
        f"  Val   mask     : {val:,}  nodes",
        f"  Test  mask     : {te:,}  nodes",
        "",
        f"  Avg node degree: {e / n:.2f}",
    ]
    path = RESULTS_DIR / "06_graph_stats.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {path.relative_to(HERE)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 6: Graph Construction")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Loading processed Elliptic data …")
    nodes = pd.read_parquet(DATA_PROC / "elliptic_nodes.parquet")
    edges = pd.read_parquet(DATA_PROC / "elliptic_edges.parquet")
    X_all = np.load(DATA_PROC / "elliptic_X_all.npy")

    print(f"  Nodes: {len(nodes):,}   Edges (raw): {len(edges):,}")

    print("\n  Building node tensors …")
    x, y, txid_to_idx = build_node_tensors(nodes, X_all)

    print("  Building edge index (bidirectional) …")
    edge_index = build_edge_index(edges, txid_to_idx)
    print(f"  Edge index shape: {list(edge_index.shape)}")

    print("  Loading chronological masks …")
    mask_train, mask_val, mask_test = build_masks()

    # Assemble PyG Data object
    data = Data(
        x          = x,
        y          = y,
        edge_index = edge_index,
        train_mask = mask_train,
        val_mask   = mask_val,
        test_mask  = mask_test,
    )

    print(f"\n  Graph summary: {data}")

    # Persist
    graph_path = DATA_PROC / "elliptic_graph.pt"
    torch.save(data, graph_path)
    print(f"  Saved graph → {graph_path.relative_to(HERE)}")

    # Persist txId → node index mapping
    map_path = DATA_PROC / "elliptic_txid_to_idx.json"
    map_path.write_text(
        json.dumps({str(k): v for k, v in txid_to_idx.items()}),
        encoding="utf-8",
    )
    print(f"  Saved index map → {map_path.name}")

    save_stats(data, txid_to_idx)

    print("\n[OK] Step 6 complete — run 07_dgi_pretrainer.py next.\n")


if __name__ == "__main__":
    main()
