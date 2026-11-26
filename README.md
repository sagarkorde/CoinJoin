# MixTrace: Auditing Graph-Based Bitcoin Forensics

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.7-orange.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code and machine-readable results for a study of graph machine learning in
Bitcoin forensics. The study builds MixTrace, a framework of three
independently evaluated modules (structural CoinJoin screening, post-mix
spending linkage with a multi-dimensional Chamfer distance, and illicit
transaction classification with a frozen Deep Graph Infomax representation
fused into a Graph Attention Network), and then audits how each module is
evaluated. It uses the public Elliptic Data Set and an author-curated corpus of
5,884,387 Bitcoin transactions.

## Main findings

Each finding is produced by one script in `research/experiments/` and recorded
against its commit in [`research/EXPERIMENT_LOG.md`](research/EXPERIMENT_LOG.md).

| Finding | Experiment |
|---|---|
| A shortest-path-to-illicit graph feature restates the label: it is 0 for all 4,545 illicit nodes and no licit node, and drives test F1, PR-AUC and MCC to at least 0.999 under two learners | E01, E02 |
| No leakage-free variant exists: the Elliptic edge set has no cross-time-step edges | E01 |
| The CoinJoin screening label is exactly `input_count >= 4 AND output_count >= 4`, and the remaining columns reconstruct both counts | E06, E08b |
| All six script-type flags are identically False; the corpus is 24.10 % Taproot by output | E07 |
| Elliptic performance is bimodal: mean F1 0.858 up to time step 42, 0.028 afterwards | E04 |
| Retraining, not threshold recalibration, recovers part of the post-shock loss | E19 |
| Protocol rules applied to block-explorer data confirm 4,491 CoinJoin transactions; the screening label's precision is 4.07 % | E14b |
| Whirlpool detection recovers a serialisation template; WabiSabi detection reaches MCC 0.935 with size, value and count features withheld | E15, E17, E20 |
| The frozen graph representation helps the GAT arm, but an untrained encoder helps equally | E05, E18 |

## Repository layout

```
research/                   audit pipeline (the code behind every reported result)
  src/                      configuration, data loading, models, metrics, figures
  experiments/              E01-E20, one script per experiment
  results/                  machine-readable outputs (CSV, JSON, NPY)
  figures/                  figures generated from those outputs
  EXPERIMENT_LOG.md         experiment -> change -> configuration -> result -> commit
  README.md                 environment, data placement and reproduction steps
  build_zenodo_bundle.py    assembles the Zenodo deposit from results/
Paper/Implementation/       the initial MixTrace pipeline whose evaluation is audited
  01_ ... 14_*.py           original fourteen-step pipeline
  results/                  its original outputs, kept so the audit can be reproduced
```

`Paper/Implementation/` is kept unchanged as the object of the audit. Its
outputs were computed with the label-derived features that E01 shows to be
leaked; they are not results of the study.

## Data

No dataset is redistributed here.

| Source | Where |
|---|---|
| Elliptic Data Set | [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), under its own terms |
| Author-curated corpus, 5,884,387 transactions | IEEE DataPort, <https://doi.org/10.21227/bxmt-mn56> |
| CoinJoin verification records and experiment outputs | Zenodo, <https://doi.org/10.5281/zenodo.24551653> |

Place the data under `Paper/Datasets/` as described in
[`research/README.md`](research/README.md).

## Reproducing

```bash
pip install -r Paper/Implementation/requirements.txt
python research/experiments/exp01_leakage_audit.py
python research/experiments/exp02_feature_ablation.py
# ... E03-E20 in the order given in research/README.md
```

The block-explorer experiments (E14, E14b) need not be re-run: the deposited
responses let them run from cache without issuing any request.

## Licence

MIT, see [LICENSE](LICENSE). The deposited data on Zenodo are CC BY 4.0.

## Contact

Sagar D. Korde, sagarkorde04@gmail.com
