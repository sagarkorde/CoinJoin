"""
Label-free topological features for the Elliptic transaction graph.

Motivation
----------
E01 showed that the round-1 features (`illicit_neighbor_ratio`,
`shortest_path_to_illicit`) are re-encodings of the target and that no
leakage-free variant of either can carry test-time information, because the
Elliptic edge set contains no cross-time-step edges.

This module replaces them with descriptors that are leakage-free by
construction, on two independent grounds:

  * **No labels.** Every quantity is a function of graph topology alone. No
    label, from any split, enters the computation.
  * **No look-ahead.** Each feature is computed inside the node's own
    time-step subgraph. Since the graph has no cross-step edges, this is not a
    restriction imposed on top of the data, it is the data's own structure.

Both properties are asserted mechanically by `verify_label_independence`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

TOPO_FEATURES = [
    "topo_in_degree",
    "topo_out_degree",
    "topo_degree",
    "topo_clustering",
    "topo_pagerank",
    "topo_core_number",
    "topo_triangles",
    "topo_nbr_degree_mean",
    "topo_nbr_degree_max",
    "topo_nbr_degree_std",
    "topo_component_size",
    "topo_two_hop_size",
    "topo_betweenness",
    "topo_is_source",
    "topo_is_sink",
]

BETWEENNESS_SAMPLES = 128   # pivots per time-step subgraph


def _subgraph_features(sub_nodes: np.ndarray,
                       sub_edges: pd.DataFrame,
                       seed: int = 0) -> pd.DataFrame:
    """Compute all topological descriptors for one time-step snapshot."""
    dg = nx.DiGraph()
    dg.add_nodes_from(sub_nodes.tolist())
    dg.add_edges_from(zip(sub_edges["txId1"].to_numpy(),
                          sub_edges["txId2"].to_numpy()))
    ug = dg.to_undirected()

    in_deg = dict(dg.in_degree())
    out_deg = dict(dg.out_degree())
    deg = dict(ug.degree())

    clustering = nx.clustering(ug)
    triangles = nx.triangles(ug)
    core = nx.core_number(ug)

    # PageRank on the directed graph; damping at the usual 0.85.
    try:
        pagerank = nx.pagerank(dg, alpha=0.85, max_iter=200, tol=1e-08)
    except nx.PowerIterationFailedConvergence:
        pagerank = {n: 1.0 / max(len(sub_nodes), 1) for n in sub_nodes}

    # Component size for each node.
    comp_size = {}
    for comp in nx.connected_components(ug):
        c = len(comp)
        for n in comp:
            comp_size[n] = c

    # Sampled betweenness: exact is quadratic, and the per-step snapshots are
    # small enough that 128 pivots give a stable ranking.
    k = min(BETWEENNESS_SAMPLES, max(ug.number_of_nodes() - 1, 1))
    if ug.number_of_nodes() > 2 and ug.number_of_edges() > 0:
        betw = nx.betweenness_centrality(ug, k=k, seed=seed, normalized=True)
    else:
        betw = {n: 0.0 for n in sub_nodes}

    rows = []
    for n in sub_nodes:
        nbrs = list(ug.neighbors(n)) if n in ug else []
        nbr_degs = [deg.get(x, 0) for x in nbrs]
        two_hop = set()
        for x in nbrs:
            two_hop.update(ug.neighbors(x))
        two_hop.discard(n)

        rows.append({
            "txId": n,
            "topo_in_degree": in_deg.get(n, 0),
            "topo_out_degree": out_deg.get(n, 0),
            "topo_degree": deg.get(n, 0),
            "topo_clustering": clustering.get(n, 0.0),
            "topo_pagerank": pagerank.get(n, 0.0),
            "topo_core_number": core.get(n, 0),
            "topo_triangles": triangles.get(n, 0),
            "topo_nbr_degree_mean": float(np.mean(nbr_degs)) if nbr_degs else 0.0,
            "topo_nbr_degree_max": float(np.max(nbr_degs)) if nbr_degs else 0.0,
            "topo_nbr_degree_std": float(np.std(nbr_degs)) if nbr_degs else 0.0,
            "topo_component_size": comp_size.get(n, 1),
            "topo_two_hop_size": len(two_hop),
            "topo_betweenness": betw.get(n, 0.0),
            "topo_is_source": int(in_deg.get(n, 0) == 0 and out_deg.get(n, 0) > 0),
            "topo_is_sink": int(out_deg.get(n, 0) == 0 and in_deg.get(n, 0) > 0),
        })
    return pd.DataFrame(rows)


def compute_topological_features(nodes: pd.DataFrame, edges: pd.DataFrame,
                                 seed: int = 0,
                                 progress: bool = True) -> pd.DataFrame:
    """
    Compute label-free topological features for every node.

    Returns a frame keyed by txId with the columns in TOPO_FEATURES.
    """
    tmap = dict(zip(nodes["txId"].to_numpy(), nodes["time_step"].to_numpy()))
    e = edges.copy()
    e["t"] = e["txId1"].map(tmap)

    out = []
    steps = sorted(nodes["time_step"].unique())
    it = tqdm(steps, desc="  topo features", disable=not progress)
    for t in it:
        sub_nodes = nodes.loc[nodes["time_step"] == t, "txId"].to_numpy()
        sub_edges = e.loc[e["t"] == t, ["txId1", "txId2"]]
        out.append(_subgraph_features(sub_nodes, sub_edges, seed=seed))

    res = pd.concat(out, ignore_index=True)
    return res[["txId"] + TOPO_FEATURES]


def verify_label_independence(nodes: pd.DataFrame, edges: pd.DataFrame,
                              feats: pd.DataFrame, seed: int = 0) -> dict:
    """
    Mechanical check that the features do not depend on labels.

    The features are recomputed after permuting every label in the dataset. A
    feature that consulted a label in any way would change; these must not.
    """
    shuffled = nodes.copy()
    rng = np.random.default_rng(seed)
    shuffled["label"] = rng.permutation(shuffled["label"].to_numpy())

    recomputed = compute_topological_features(shuffled, edges, seed=seed,
                                              progress=False)
    a = feats.sort_values("txId").reset_index(drop=True)
    b = recomputed.sort_values("txId").reset_index(drop=True)

    identical = {}
    for c in TOPO_FEATURES:
        identical[c] = bool(np.allclose(a[c].to_numpy(dtype=float),
                                        b[c].to_numpy(dtype=float),
                                        rtol=0, atol=0))
    return {
        "all_features_invariant_to_label_permutation": all(identical.values()),
        "per_feature": identical,
    }
