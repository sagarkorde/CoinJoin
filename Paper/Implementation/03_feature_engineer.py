"""
03_feature_engineer.py  |  MixTrace Pipeline — Step 3: Novel Feature Engineering
==================================================================================
Computes five novel features for the Author Dataset and three for the Elliptic
Dataset that serve as key discriminative signals in the MixTrace framework.

Novel Features — Author Dataset
--------------------------------
  1. denomination_entropy (DE)
       Shannon entropy approximation over output value distribution.
       High entropy → well-mixed, arbitrary-value transactions.
       Proxy: computed from value_concentration_ratio and avg_output_value.

  2. mixing_index (MI)
       Normalised product of input and output counts relative to total addresses.
       MI = (input_count × output_count) / total_addresses
       Captures the combinatorial complexity of collaborative transactions.

  3. temporal_irregularity (TI)
       Z-score of fee_rate_sat_per_vbyte within the same (hour, day_of_week) bin.
       CoinJoin transactions exhibit atypical fee patterns at certain times.

  4. utxo_concentration (UC)
       Herfindahl–Hirschman Index approximation over output values.
       UC = value_concentration_ratio² (bounded in [0, 1]).
       Low UC → evenly distributed outputs (equal-denomination mixing).

  5. coinjoin_proximity_score (CPS_author)
       Rule-based composite score from structural CoinJoin indicators:
         - input_count > 2  (collaborative inputs)
         - output_count > 2  (multiple recipients)
         - address_reuse == 0  (no address reuse)
         - is_batch_payment   (batch structure)
         - low utxo_concentration  (equal denominations)

Novel Features — Elliptic Dataset
-----------------------------------
  6. illicit_neighbor_ratio (INR)
       Fraction of 1-hop neighbours with a known illicit label.
       Strong proximity signal for guilt-by-association.

  7. graph_in_degree  /  graph_out_degree
       Raw degree counts from the directed transaction edge list.

  8. shortest_path_to_illicit (SPTI)
       BFS distance (capped at 6) to the nearest known illicit node.
       Captures how far removed a node is from confirmed malicious activity.
       Nodes beyond hop-5 are assigned distance 6; labeled illicit nodes → 0.

Run
---
    python 03_feature_engineer.py

Reads   : data/processed/  (Steps 1–2 output)
Outputs : data/processed/
    author_features_engineered.parquet
    elliptic_features_engineered.parquet
    results/03_feature_stats.txt
"""

import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
DATA_PROC   = HERE / "data" / "processed"
RESULTS_DIR = HERE / "results"

SPTI_CAP = 6   # distances beyond this cap are set to CAP value


# ── Author Dataset Features ───────────────────────────────────────────────────

def compute_denomination_entropy(df: pd.DataFrame) -> pd.Series:
    """
    Shannon entropy approximation over the output value distribution.

    Uses value_concentration_ratio (VCR) as a proxy for the Herfindahl index.
    Entropy ≈ -VCR * log(VCR) - (1 - VCR) * log(1 - VCR + eps)
    """
    vcr = df["value_concentration_ratio"].clip(1e-9, 1 - 1e-9)
    return -(vcr * np.log(vcr) + (1 - vcr) * np.log(1 - vcr)).rename(
        "denomination_entropy"
    )


def compute_mixing_index(df: pd.DataFrame) -> pd.Series:
    """Normalised combinatorial complexity: (input_count × output_count) / total_addresses."""
    denom = df["total_addresses"].replace(0, 1)
    return (df["input_count"] * df["output_count"] / denom).rename("mixing_index")


def compute_temporal_irregularity(df: pd.DataFrame) -> pd.Series:
    """
    Z-score of fee_rate_sat_per_vbyte within each (hour, day_of_week) bin.

    Captures fee behaviour that deviates from the temporal norm — CoinJoin
    transactions are often submitted with atypical fees at specific times.
    """
    col = "fee_rate_sat_per_vbyte"
    grouped = df.groupby(["hour", "day_of_week"])[col]
    z = (df[col] - grouped.transform("mean")) / (grouped.transform("std") + 1e-9)
    return z.rename("temporal_irregularity")


def compute_utxo_concentration(df: pd.DataFrame) -> pd.Series:
    """HHI approximation: vcr² ∈ [0, 1]. Low → equal-denomination outputs."""
    vcr = df["value_concentration_ratio"].clip(0, 1)
    return (vcr ** 2).rename("utxo_concentration")


def compute_cps_author(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based CoinJoin Proximity Score (CPS) for the Author Dataset.

    Each structural indicator contributes equally (weight = 0.2).
    """
    s  = (df["input_count"]  > 2).astype(float) * 0.25
    s += (df["output_count"] > 2).astype(float) * 0.25
    s += (df["address_reuse"] == 0).astype(float) * 0.20
    s += df["is_batch_payment"].astype(float) * 0.15
    uc = compute_utxo_concentration(df)
    s += (uc < uc.median()).astype(float) * 0.15
    return s.rename("coinjoin_proximity_score")


def engineer_author_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["denomination_entropy"]   = compute_denomination_entropy(df)
    df["mixing_index"]           = compute_mixing_index(df)
    df["temporal_irregularity"]  = compute_temporal_irregularity(df)
    df["utxo_concentration"]     = compute_utxo_concentration(df)
    df["coinjoin_proximity_score"] = compute_cps_author(df)
    return df


# ── Elliptic Dataset Features ─────────────────────────────────────────────────

def compute_degree_features(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    """In-degree and out-degree from directed edge list."""
    out_deg = edges.groupby("txId1").size().rename("graph_out_degree")
    in_deg  = edges.groupby("txId2").size().rename("graph_in_degree")
    nodes = nodes.merge(out_deg, left_on="txId", right_index=True, how="left")
    nodes = nodes.merge(in_deg,  left_on="txId", right_index=True, how="left")
    nodes["graph_out_degree"].fillna(0, inplace=True)
    nodes["graph_in_degree"].fillna(0,  inplace=True)
    return nodes


def compute_illicit_neighbor_ratio(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    """Fraction of 1-hop neighbours with a confirmed illicit label."""
    illicit_set = set(nodes.loc[nodes["label"] == 1, "txId"].values)

    # Build neighbour lookup
    src_to_dst: dict[int, list[int]] = {}
    for row in edges.itertuples(index=False):
        src_to_dst.setdefault(row.txId1, []).append(row.txId2)
        src_to_dst.setdefault(row.txId2, []).append(row.txId1)

    def _ratio(tx_id: int) -> float:
        neighbours = src_to_dst.get(tx_id, [])
        if not neighbours:
            return 0.0
        ill = sum(1 for n in neighbours if n in illicit_set)
        return ill / len(neighbours)

    tqdm.pandas(desc="  INR")
    nodes["illicit_neighbor_ratio"] = nodes["txId"].progress_apply(_ratio)
    return nodes


def compute_shortest_path_to_illicit(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    """
    BFS distance to the nearest confirmed illicit node, capped at SPTI_CAP.

    Efficiently computed by running BFS from every illicit source node on
    the reversed graph, then taking the minimum distance for each target.
    """
    print("  Building NetworkX graph for SPTI …")
    G = nx.DiGraph()
    G.add_edges_from(zip(edges["txId1"], edges["txId2"]))

    illicit_nodes = nodes.loc[nodes["label"] == 1, "txId"].tolist()
    all_node_ids  = set(nodes["txId"].values)

    # dist_map[node] = minimum BFS distance from any illicit source
    dist_map: dict[int, int] = {n: SPTI_CAP for n in all_node_ids}
    for ill_node in illicit_nodes:
        dist_map[ill_node] = 0

    G_rev = G.reverse(copy=False)
    print(f"  Running BFS from {len(illicit_nodes):,} illicit sources (cutoff={SPTI_CAP}) …")
    for source in tqdm(illicit_nodes, desc="  SPTI BFS"):
        lengths = nx.single_source_shortest_path_length(
            G_rev, source, cutoff=SPTI_CAP - 1
        )
        for node, dist in lengths.items():
            if node in dist_map and dist < dist_map[node]:
                dist_map[node] = dist

    nodes["shortest_path_to_illicit"] = nodes["txId"].map(dist_map).fillna(SPTI_CAP)
    return nodes


def engineer_elliptic_features(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    print("  Computing degree features …")
    nodes = compute_degree_features(nodes, edges)
    print("  Computing illicit-neighbour ratio …")
    nodes = compute_illicit_neighbor_ratio(nodes, edges)
    print("  Computing shortest-path-to-illicit …")
    nodes = compute_shortest_path_to_illicit(nodes, edges)
    return nodes


# ── Reporting ─────────────────────────────────────────────────────────────────

def save_stats(
    author: pd.DataFrame, nodes: pd.DataFrame
) -> None:
    author_novel = [
        "denomination_entropy", "mixing_index", "temporal_irregularity",
        "utxo_concentration", "coinjoin_proximity_score",
    ]
    elliptic_novel = [
        "graph_in_degree", "graph_out_degree",
        "illicit_neighbor_ratio", "shortest_path_to_illicit",
    ]

    lines = [
        "=" * 65,
        "MixTrace — Novel Feature Statistics  (Step 3)",
        "=" * 65, "",
        "── Author Dataset Novel Features",
    ]
    for feat in author_novel:
        s = author[feat]
        lines.append(
            f"   {feat:<30} mean={s.mean():.4f}  std={s.std():.4f}  "
            f"[{s.min():.4f}, {s.max():.4f}]"
        )

    lines += ["", "── Elliptic Dataset Novel Features"]
    for feat in elliptic_novel:
        s = nodes[feat]
        lines.append(
            f"   {feat:<30} mean={s.mean():.4f}  std={s.std():.4f}  "
            f"[{s.min():.4f}, {s.max():.4f}]"
        )

    # Correlation with labels for Elliptic
    labeled = nodes[nodes["label"] >= 0]
    lines += ["", "── Elliptic Novel Feature Correlation with Label (labeled nodes)"]
    for feat in elliptic_novel:
        corr = labeled[feat].corr(labeled["label"])
        lines.append(f"   {feat:<30} r = {corr:+.4f}")

    path = RESULTS_DIR / "03_feature_stats.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {path.relative_to(HERE)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 3: Novel Feature Engineering")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Author Dataset ──
    print("\n[A] Engineering Author Dataset features …")
    author = pd.read_parquet(DATA_PROC / "author_raw.parquet")
    author = engineer_author_features(author)
    out_path = DATA_PROC / "author_features_engineered.parquet"
    author.to_parquet(out_path, index=False)
    print(f"  Saved {len(author):,} rows → {out_path.name}")

    # ── Elliptic Dataset ──
    print("\n[B] Engineering Elliptic Dataset features …")
    nodes = pd.read_parquet(DATA_PROC / "elliptic_nodes.parquet")
    edges = pd.read_parquet(DATA_PROC / "elliptic_edges.parquet")
    nodes = engineer_elliptic_features(nodes, edges)
    out_path = DATA_PROC / "elliptic_features_engineered.parquet"
    nodes.to_parquet(out_path, index=False)
    print(f"  Saved {len(nodes):,} nodes → {out_path.name}")

    save_stats(author, nodes)

    print("\n[OK] Step 3 complete — run 04_baseline_chamfer.py next.\n")


if __name__ == "__main__":
    main()
