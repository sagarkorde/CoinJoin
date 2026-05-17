"""
01_data_loader.py  |  MixTrace Pipeline — Step 1: Data Loading & Validation
============================================================================
Loads and validates all three source datasets, prints summary statistics,
and persists cleaned versions to data/processed/ for downstream steps.

Datasets
--------
  A  Author Dataset    — Dataset.parquet   (5.88 M transactions, 53 features)
  B  Elliptic Dataset  — 3 CSV files       (203,769 nodes, graph structure)
  C  Blockchain Dataset — placeholder      (loaded if files are present)

Run
---
    python 01_data_loader.py

Outputs (data/processed/)
--------------------------
    author_raw.parquet
    elliptic_nodes.parquet   — features merged with class labels
    elliptic_edges.parquet
    results/01_dataset_stats.txt
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE           = Path(__file__).resolve().parent
ROOT           = HERE.parent
DATA_RAW       = ROOT / "Datasets"
DATA_PROC      = HERE / "data" / "processed"
RESULTS_DIR    = HERE / "results"

AUTHOR_FILE    = DATA_RAW / "Author_Dataset"    / "Dataset.parquet"
ELLIPTIC_DIR   = DATA_RAW / "Elliptic_Dataset"
BLOCKCHAIN_DIR = DATA_RAW / "Blockchai_Dataset"

# Elliptic feature column names (raw file has no header)
_LOCAL_COLS = [f"local_{i:02d}" for i in range(1, 94)]   # 93 local node features
_AGG_COLS   = [f"agg_{i:02d}"   for i in range(1, 73)]   # 72 aggregated neighbourhood features
ELLIPTIC_FEATURE_COLS = ["txId", "time_step"] + _LOCAL_COLS + _AGG_COLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dirs() -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _banner(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── Dataset Loaders ───────────────────────────────────────────────────────────

def load_author() -> pd.DataFrame:
    """Load Author Dataset from parquet and print summary statistics."""
    _banner("Dataset A — Author Dataset")
    df  = pd.read_parquet(AUTHOR_FILE)
    cj  = int(df["is_coinjoin_like"].sum())
    tap = int(df["has_taproot"].sum())
    print(f"  Rows            : {len(df):>12,}")
    print(f"  Columns         : {df.shape[1]:>12}")
    print(f"  CoinJoin-like   : {cj:>12,}  ({cj / len(df) * 100:.2f} %)")
    print(f"  Taproot txs     : {tap:>12,}  ({tap / len(df) * 100:.2f} %)")
    print(f"  Date range      : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Null cells      : {df.isnull().sum().sum():>12,}")
    print(f"  Memory (MB)     : {df.memory_usage(deep=True).sum() / 1e6:>11.1f}")
    return df


def load_elliptic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge Elliptic features, class labels, and edge list."""
    _banner("Dataset B — Elliptic Dataset")

    feat  = pd.read_csv(
        ELLIPTIC_DIR / "elliptic_txs_features.csv",
        header=None, names=ELLIPTIC_FEATURE_COLS,
    )
    cls   = pd.read_csv(ELLIPTIC_DIR / "elliptic_txs_classes.csv")
    edges = pd.read_csv(ELLIPTIC_DIR / "elliptic_txs_edgelist.csv")

    nodes = feat.merge(cls, on="txId", how="left")

    # Normalise class label: 1 → illicit, 0 → licit, -1 → unknown
    nodes["label"] = (
        nodes["class"]
        .map({"1": 1, "2": 0, 1: 1, 2: 0})
        .fillna(-1)
        .astype(int)
    )

    labeled = nodes[nodes["label"] >= 0]
    ill     = int((labeled["label"] == 1).sum())
    lic     = int((labeled["label"] == 0).sum())

    print(f"  Total nodes     : {len(nodes):>12,}")
    print(f"  Labeled nodes   : {len(labeled):>12,}")
    print(f"    Illicit (1)   : {ill:>12,}  ({ill / len(labeled) * 100:.2f} %)")
    print(f"    Licit   (0)   : {lic:>12,}  ({lic / len(labeled) * 100:.2f} %)")
    print(f"  Total edges     : {len(edges):>12,}")
    print(f"  Time steps      : {nodes['time_step'].nunique():>12}  "
          f"(1 – {int(nodes['time_step'].max())})")
    print(f"  Feature dim     : {len(_LOCAL_COLS) + len(_AGG_COLS):>12}")

    return nodes, edges


def load_blockchain() -> pd.DataFrame | None:
    """Load Blockchain Dataset if present (placeholder for future data)."""
    _banner("Dataset C — Blockchain Dataset (placeholder)")
    files = [
        f for f in BLOCKCHAIN_DIR.iterdir()
        if f.suffix in {".csv", ".parquet", ".json"}
    ]
    if not files:
        print("  Directory is empty — skipping.")
        return None

    parts = []
    for f in files:
        if f.suffix == ".csv":
            parts.append(pd.read_csv(f))
        elif f.suffix == ".parquet":
            parts.append(pd.read_parquet(f))
        elif f.suffix == ".json":
            parts.append(pd.read_json(f))

    df = pd.concat(parts, ignore_index=True)
    print(f"  Shape           : {df.shape}")
    return df


# ── Statistics Report ─────────────────────────────────────────────────────────

def save_stats(
    author: pd.DataFrame,
    nodes:  pd.DataFrame,
    edges:  pd.DataFrame,
) -> None:
    labeled = nodes[nodes["label"] >= 0]
    ill     = int((labeled["label"] == 1).sum())
    lic     = int((labeled["label"] == 0).sum())

    lines = [
        "=" * 65,
        "MixTrace — Dataset Statistics  (Step 1)",
        "=" * 65,
        "",
        "── Author Dataset",
        f"   Rows              : {len(author):,}",
        f"   Columns           : {author.shape[1]}",
        f"   CoinJoin-like     : {int(author['is_coinjoin_like'].sum()):,}"
        f"  ({author['is_coinjoin_like'].mean() * 100:.2f} %)",
        f"   Taproot txs       : {int(author['has_taproot'].sum()):,}"
        f"  ({author['has_taproot'].mean() * 100:.2f} %)",
        f"   P2WPKH txs        : {int(author['has_p2wpkh'].sum()):,}",
        f"   P2SH txs          : {int(author['has_p2sh'].sum()):,}",
        f"   Memory (MB)       : {author.memory_usage(deep=True).sum() / 1e6:.1f}",
        "",
        "── Elliptic Dataset",
        f"   Total nodes       : {len(nodes):,}",
        f"   Labeled nodes     : {len(labeled):,}",
        f"     Illicit (1)     : {ill:,}  ({ill / len(labeled) * 100:.2f} %)",
        f"     Licit   (0)     : {lic:,}  ({lic / len(labeled) * 100:.2f} %)",
        f"     Unknown         : {len(nodes) - len(labeled):,}",
        f"   Total edges       : {len(edges):,}",
        f"   Time steps        : {nodes['time_step'].nunique()}",
        f"   Feature dim       : {len(_LOCAL_COLS) + len(_AGG_COLS)}",
        f"     Local features  : {len(_LOCAL_COLS)}",
        f"     Aggregated      : {len(_AGG_COLS)}",
        "",
        "── Class Imbalance Ratio (Elliptic)",
        f"   Licit : Illicit   : {lic / ill:.1f} : 1",
    ]

    path = RESULTS_DIR / "01_dataset_stats.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved → {path.relative_to(HERE)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MixTrace  |  Step 1: Data Loading & Validation")
    print("=" * 60)

    _make_dirs()

    author           = load_author()
    nodes, edges     = load_elliptic()
    blockchain       = load_blockchain()

    print("\n── Persisting processed datasets …")
    author.to_parquet(DATA_PROC / "author_raw.parquet",      index=False)
    nodes.to_parquet( DATA_PROC / "elliptic_nodes.parquet",  index=False)
    edges.to_parquet( DATA_PROC / "elliptic_edges.parquet",  index=False)
    if blockchain is not None:
        blockchain.to_parquet(DATA_PROC / "blockchain_raw.parquet", index=False)
    print("  All files saved to data/processed/")

    save_stats(author, nodes, edges)

    print("\n[OK] Step 1 complete — run 02_preprocessor.py next.\n")


if __name__ == "__main__":
    main()
