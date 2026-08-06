# MixTrace: Multi-Modal Graph Attention Network for CoinJoin Transaction Deanonymization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3+-orange.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Research Study** — Implementation pipeline for the paper:  
> *"Solving the CoinJoin Mystery: A Multi-Modal Graph Attention Framework for Bitcoin Transaction Deanonymization"*

---

## Overview

**MixTrace** is a novel framework for deanonymizing Bitcoin CoinJoin transactions — a collaborative transaction schema that deliberately obscures payment flows by merging multiple users' UTXOs into a single on-chain transaction.

This study addresses three open research gaps in blockchain privacy analysis:

| Gap | Description | Method |
|-----|-------------|--------|
| **Gap 1** | Taproot-upgrade clustering bypasses and false positives | Taproot-Aware DBSCAN clustering vs. naive CIOH |
| **Gap 2** | Multi-dimensional post-mix spending (CST) deanonymization | Multi-Dimensional Chamfer Distance (MD-CD, d=4) |
| **Gap 3** | ML class imbalances in illicit activity detection | DGI pre-training + GAT fine-tuning + ADASYN |

### Novel Contributions

1. **Multi-Dimensional Chamfer Distance (MD-CD)** — extends the 1-D timestamp-based CST linking metric to a 4-D feature space `[timestamp, fee_rate, input_count, output_count]`
2. **CoinJoin Proximity Score (CPS)** — rule-based structural feature quantifying collaborative transaction indicators
3. **Denomination Entropy (DE)** — Shannon entropy approximation over UTXO output value distributions
4. **DGI + GAT (MixTrace)** — self-supervised Deep Graph Infomax pre-training on the full unlabeled graph, followed by supervised Graph Attention Network fine-tuning
5. **Taproot-Aware Heuristic** — structural DBSCAN clustering that reduces false positives from the Common Input Ownership Heuristic (CIOH) under P2TR transactions

---

## Repository Structure

```
CoinJoin/
├── Datasets/
│   ├── Author_Dataset/          # Author-collected Bitcoin transactions (5.88M rows)
│   │   └── Dataset.parquet      # ⚠ Not tracked (1.7 GB) — download separately
│   ├── Elliptic_Dataset/        # Elliptic Bitcoin Dataset (publicly available)
│   │   ├── elliptic_txs_features.csv   # ⚠ Not tracked (690 MB) — download separately
│   │   ├── elliptic_txs_classes.csv    # ⚠ Not tracked — download separately
│   │   └── elliptic_txs_edgelist.csv   # ⚠ Not tracked — download separately
│   └── Blockchai_Dataset/       # Placeholder for additional datasets
│
├── Reference_Papers/            # 8 reference PDFs (literature)
│
├── Paper/
│   └── Implementation/          # ← All source code
│       ├── requirements.txt
│       ├── 01_data_loader.py        # Step 1 : Load & validate datasets
│       ├── 02_preprocessor.py       # Step 2 : Clean, normalize, split
│       ├── 03_feature_engineer.py   # Step 3 : Novel feature computation
│       ├── 04_baseline_chamfer.py   # Step 4 : 1-D Chamfer Distance baseline
│       ├── 05_baseline_classifiers.py  # Step 5 : RF, KNN, K-Means
│       ├── 06_graph_builder.py      # Step 6 : PyTorch Geometric graph
│       ├── 07_dgi_pretrainer.py     # Step 7 : Deep Graph Infomax pre-training
│       ├── 08_gat_model.py          # Step 8 : GAT model architecture
│       ├── 09_training_pipeline.py  # Step 9 : Fine-tuning with ADASYN
│       ├── 10_taproot_analyzer.py   # Step 10: Gap 1 — Taproot analysis
│       ├── 11_multidim_chamfer.py   # Step 11: Gap 2 — MD-Chamfer Distance
│       ├── 12_coinjoin_detector.py  # Step 12: End-to-end inference
│       ├── 13_evaluator.py          # Step 13: Metrics & comparison tables
│       ├── 14_visualizer.py         # Step 14: Publication figures
│       ├── run_seed_experiment.py       # Five-seed validation harness
│       ├── aggregate_multiseed.py       # Aggregate per-seed results
│       ├── generate_multiseed_report.py # Cross-seed summary report
│       └── results/, results_multiseed/ # Metrics, JSON/CSV outputs
│
└── README.md
```

---

## Datasets

### Elliptic Bitcoin Dataset (Public)
Download from Kaggle: [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Place the three CSV files in `Datasets/Elliptic_Dataset/`:
```
elliptic_txs_features.csv   (203,769 transactions × 167 features)
elliptic_txs_classes.csv    (labels: 1=illicit, 2=licit, unknown)
elliptic_txs_edgelist.csv   (234,355 directed edges)
```

| Statistic | Value |
|-----------|-------|
| Total nodes | 203,769 |
| Labeled (illicit) | 4,545 |
| Labeled (licit) | 42,019 |
| Unlabeled | 157,205 |
| Edges | 234,355 |
| Features per node | 167 |
| Time steps | 49 |

### Author Dataset (Bitcoin Transaction Graph)
A curated dataset of 5.88M Bitcoin transactions with 53 engineered features including `is_coinjoin_like`, script type flags (`has_taproot`, `has_p2wpkh`, etc.), fee rates, and temporal features.

> Contact the authors for access to `Dataset.parquet`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sagarkorde/CoinJoin.git
cd CoinJoin/Paper/Implementation

# Install dependencies
pip install -r requirements.txt

# PyTorch + PyTorch Geometric (CPU)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# PyTorch + PyTorch Geometric (CUDA 11.8)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

**Requirements:** Python 3.10+, 16 GB RAM minimum (64 GB recommended for full dataset)

---

## Execution Pipeline

Run each step in order from the `Paper/Implementation/` directory:

```bash
# Phase 1 — Data Preparation
python 01_data_loader.py        # Load & validate all datasets
python 02_preprocessor.py       # Normalize, split, save numpy arrays
python 03_feature_engineer.py   # Compute 5 Author + 4 Elliptic novel features

# Phase 2 — Baselines
python 04_baseline_chamfer.py   # 1-D Chamfer Distance (CST linking)
python 05_baseline_classifiers.py  # Random Forest, KNN, K-Means

# Phase 3 — Graph Construction
python 06_graph_builder.py      # Build PyG Data object (bidirectional graph)

# Phase 4 — MixTrace Training
python 07_dgi_pretrainer.py     # DGI self-supervised pre-training (300 epochs)
python 08_gat_model.py          # Verify GAT architecture
python 09_training_pipeline.py  # Fine-tune with class weighting + ADASYN

# Phase 5 — Research Gap Analysis
python 10_taproot_analyzer.py   # Gap 1: Taproot-Aware DBSCAN vs. CIOH
python 11_multidim_chamfer.py   # Gap 2: MD-CD vs. 1-D Chamfer

# Phase 6 — Evaluation & Output
python 12_coinjoin_detector.py  # End-to-end inference + threshold optimisation
python 13_evaluator.py          # Comparison table + LaTeX export
python 14_visualizer.py         # 11 publication-quality figures (PDF + PNG)
```

---

## Model Architecture — MixTrace

```
Input Features [N × 169]
        │
   Linear Projection → [N × 128]
        │
   GAT Layer 1 (8 heads, concat) → [N × 1024]
   BatchNorm + ELU + Dropout
        │
   GAT Layer 2 (1 head) → [N × 128]
   BatchNorm + ELU
        │
   ┌────┴────────────────────┐
   │                         │
GAT Output              DGI Skip Connection
[N × 128]          (frozen GCN encoder) [N × 128]
   │                         │
   └──────── Concat ─────────┘
                │
         [N × 256]
                │
         FC → ELU → Dropout → FC
                │
         Logits [N × 2]
         (licit / illicit)
```

**Training protocol:**
- DGI pre-training: 300 epochs, Adam, lr=1e-3, weight decay=5e-4
- GAT fine-tuning: up to 500 epochs, early stopping (patience=20), ReduceLROnPlateau
- Class weighting: inverse frequency (illicit weight ≈ 4.3×)
- Split: chronological by time step (train: 1–34, val: 35–41, test: 42–49)

---

## Novel Features Computed

### Author Dataset (5 features)
| Feature | Description |
|---------|-------------|
| `denomination_entropy` | Shannon entropy of UTXO output value distribution |
| `mixing_index` | `(input_count × output_count) / total_addresses` |
| `temporal_irregularity` | Z-score of fee rate within `(hour, day_of_week)` bin |
| `utxo_concentration` | HHI approximation: `value_concentration_ratio²` |
| `coinjoin_proximity_score` | Weighted rule-based CoinJoin structural score |

### Elliptic Dataset (4 features)
| Feature | Description |
|---------|-------------|
| `illicit_neighbor_ratio` | Fraction of 1-hop neighbours with illicit label |
| `graph_in_degree` | In-degree from directed edge list |
| `graph_out_degree` | Out-degree from directed edge list |
| `shortest_path_to_illicit` | BFS distance to nearest illicit node (capped at 6) |

---

## Results

### Illicit Transaction Detection (Elliptic Dataset, Test Set)

| Method | Precision | Recall | F1 (illicit) | PR-AUC |
|--------|-----------|--------|--------------|--------|
| K-Means | — | — | — | — |
| KNN | — | — | — | — |
| Random Forest | — | — | — | — |
| GAT (no DGI) | — | — | — | — |
| **MixTrace (DGI+GAT)** | — | — | — | — |

> Results populated after running Steps 5, 9, 12, and 13.

### CST Linking — Chamfer Distance

| Method | Dimensions | F1 | Precision | Recall |
|--------|-----------|-----|-----------|--------|
| 1-D Chamfer | timestamp | — | — | — |
| **MD-Chamfer** | timestamp, fee, inputs, outputs | — | — | — |

### Taproot Analysis (Gap 1)

| Method | FPR | FNR | F1 |
|--------|-----|-----|-----|
| Naive CIOH | — | — | — |
| **Taproot-Aware DBSCAN** | — | — | — |

> Run Steps 10 and 11 to populate these tables.

---

## Research Gaps Addressed

### Gap 1 — Taproot-Upgrade Clustering Bypasses
Bitcoin's Taproot soft fork (BIP 340/341/342) makes P2TR CoinJoin transactions indistinguishable from single-user P2TR transactions on-chain, defeating the CIOH. MixTrace introduces a structural DBSCAN clustering approach using multi-dimensional behavioural features.

### Gap 2 — Multi-Dimensional CST Deanonymization
The 1-D Chamfer distance (timestamp-only) misses structural similarities between CoinJoin Spending Transactions. MD-CD extends the metric to `d=4` dimensions, improving linking F1.

### Gap 3 — Class Imbalance in Illicit Detection
The Elliptic dataset has a 9.2:1 licit-to-illicit ratio. MixTrace combines: (a) DGI pre-training on all 203K nodes including unlabeled, (b) class-weighted cross-entropy, and (c) ADASYN oversampling for diagnostic reference.

---

## Reference Papers

| # | Paper |
|---|-------|
| 1 | Reducing Privacy of CoinJoin Transactions — Quantitative Bitcoin Network Analysis |
| 2 | Improving Cryptocurrency Crime Detection — CoinJoin Community Detection Approach |
| 3 | A Similarity Measure for Linking CoinJoin Output Spenders |
| 4 | The Price of Anonymity |
| 5 | Ransomware Payments in Bitcoin |
| 6 | Bitcoin Shared Send Transactions Untangling in Numbers |
| 7 | Block Number-Based Address Clustering for Bitcoin Taproot Upgrade |
| 8 | SoK: Assumptions Underlying Cryptocurrency |

---

## Citation

```bibtex
@article{korde2026mixtrace,
  title   = {Solving the CoinJoin Mystery: A Multi-Modal Graph Attention Framework
             for Bitcoin Transaction Deanonymization},
  author  = {Korde, Sagar},
  journal = {Crime, Law and Social Change},
  year    = {2026},
  note    = {Under Review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Sagar Korde** — sagarkorde04@gmail.com  
GitHub: [@sagarkorde](https://github.com/sagarkorde)
