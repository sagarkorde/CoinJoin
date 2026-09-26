"""
Elliptic dataset loading and the chronological split.

Loads from the raw distributed CSVs so that every downstream artefact is
reproducible from source. Nothing here consults a label outside the training
period; feature construction lives in features.py and is audited separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (RAW_ELLIPTIC, CACHE, TRAIN_END, VAL_END, TEST_START,
                    LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN)

_NODE_CACHE = CACHE / "elliptic_nodes.parquet"
_EDGE_CACHE = CACHE / "elliptic_edges.parquet"


def load_elliptic(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (nodes, edges).

    nodes : txId, time_step, label, f_0 .. f_164
            label is 1 illicit / 0 licit / -1 unknown.
    edges : txId1, txId2  (directed, as distributed)
    """
    if not force and _NODE_CACHE.exists() and _EDGE_CACHE.exists():
        return pd.read_parquet(_NODE_CACHE), pd.read_parquet(_EDGE_CACHE)

    feats = pd.read_csv(RAW_ELLIPTIC / "elliptic_txs_features.csv", header=None)
    # Column 0 = txId, column 1 = time step, columns 2.. = 165 features.
    feats.columns = ["txId", "time_step"] + [f"f_{i}" for i in range(feats.shape[1] - 2)]

    classes = pd.read_csv(RAW_ELLIPTIC / "elliptic_txs_classes.csv")
    classes["label"] = classes["class"].map(
        {"1": LABEL_ILLICIT, "2": LABEL_LICIT, "unknown": LABEL_UNKNOWN}
    )
    nodes = feats.merge(classes[["txId", "label"]], on="txId", how="left")
    nodes["label"] = nodes["label"].fillna(LABEL_UNKNOWN).astype(int)

    edges = pd.read_csv(RAW_ELLIPTIC / "elliptic_txs_edgelist.csv")

    nodes.to_parquet(_NODE_CACHE, index=False)
    edges.to_parquet(_EDGE_CACHE, index=False)
    return nodes, edges


def chronological_masks(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over the node table, in its current row order."""
    ts = nodes["time_step"].to_numpy()
    return {
        "train": ts <= TRAIN_END,
        "val":   (ts > TRAIN_END) & (ts <= VAL_END),
        "test":  ts >= TEST_START,
    }


def labelled_masks(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Chronological masks intersected with 'has a ground-truth label'."""
    has_label = nodes["label"].to_numpy() != LABEL_UNKNOWN
    return {k: m & has_label for k, m in chronological_masks(nodes).items()}


def base_feature_columns(nodes: pd.DataFrame) -> list[str]:
    """The 165 distributed Elliptic features (time_step excluded)."""
    return [c for c in nodes.columns if c.startswith("f_")]


def split_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    """Per-split node and class counts, for the manuscript's dataset table."""
    lm = labelled_masks(nodes)
    cm = chronological_masks(nodes)
    y = nodes["label"].to_numpy()
    rows = []
    for name in ("train", "val", "test"):
        rows.append({
            "split": name,
            "time_steps": {"train": f"1-{TRAIN_END}",
                           "val": f"{TRAIN_END+1}-{VAL_END}",
                           "test": f"{TEST_START}-49"}[name],
            "nodes_total": int(cm[name].sum()),
            "nodes_labelled": int(lm[name].sum()),
            "illicit": int((y[lm[name]] == LABEL_ILLICIT).sum()),
            "licit": int((y[lm[name]] == LABEL_LICIT).sum()),
            "unknown": int(cm[name].sum() - lm[name].sum()),
        })
    df = pd.DataFrame(rows)
    df["illicit_rate_%"] = (df["illicit"] / df["nodes_labelled"] * 100).round(3)
    return df
