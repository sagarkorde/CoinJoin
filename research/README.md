# MixTrace — Round-2 Revision Pipeline

Reproducible re-implementation of the MixTrace study, rebuilt from the raw data
after the round-1 review. The round-1 code is preserved unchanged in
`Paper/Implementation/` and in Git history; nothing here overwrites it.

## Why this directory exists

The round-1 review raised two methodological concerns. Both were investigated
mechanically and both were confirmed, one of them more severe than the report
suggested. A third item — a stated limitation about Taproot — turned out to be
a data-processing defect rather than a property of the blockchain. The findings
are severe enough that editing the original scripts in place would have
destroyed the evidence, so the revision is a separate, self-contained pipeline.

Headline findings, each reproducible from a single script:

| Finding | Evidence |
|---|---|
| `shortest_path_to_illicit` is an exact re-encoding of the label: it is 0 for all 4,545 illicit nodes and no licit node | `exp01_leakage_audit.py` |
| Adding that feature drives test F1, PR-AUC and MCC to 1.0000 under two model families | `exp02_feature_ablation.py` |
| No leakage-free version can exist: the Elliptic edge set has 0 cross-time-step edges | `exp01_leakage_audit.py` |
| `is_coinjoin_like == (input_count >= 4) AND (output_count >= 4)`, exactly, for all 5,884,387 rows | `exp06_coinjoin_label_recovery.py` |
| All six script-type flags are identically False; the corpus is in fact 24.1 % P2TR by output | `exp07_script_types.py` |
| Performance is bimodal: mean F1 0.858 on steps 35–42, 0.028 on steps 43–49 | `exp04_temporal_drift.py` |

## Environment

| Component | Version |
|---|---|
| Python | 3.10.10 |
| PyTorch | 2.7.1+cu128 |
| PyTorch Geometric | 2.7.0 |
| scikit-learn | 1.6.1 |
| numpy | 1.26.4 |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM |
| CPU / RAM | Intel i9 (13th gen) / 64 GB |
| OS | Windows 11 |

All scripts are run with the absolute interpreter path, because the default
`python` on this machine is 3.14 and has no PyTorch:

```bash
"C:/Program Files/Python310/python.exe" research/experiments/exp01_leakage_audit.py
```

## Data

Neither dataset is redistributed here.

* **Elliptic Data Set** — place `elliptic_txs_features.csv`,
  `elliptic_txs_classes.csv` and `elliptic_txs_edgelist.csv` in
  `Paper/Datasets/Elliptic_Dataset/`. 203,769 nodes, 234,355 edges, 165
  features, 49 time steps.
* **Author-curated corpus** — `Paper/Datasets/Author_Dataset/Dataset.parquet`.
  5,884,387 Bitcoin transactions, 53 columns, 2022-07-13 to 2025-07-01.

`src/data.py` loads the Elliptic CSVs directly and caches to
`research/cache/`, so every downstream artefact is reproducible from source.

### Chronological split (fixed everywhere)

| Split | Time steps | Nodes | Labelled | Illicit | Licit | Illicit rate |
|---|---|---|---|---|---|---|
| Train | 1–34 | 136,265 | 29,894 | 3,462 | 26,432 | 11.58 % |
| Validation | 35–41 | 30,680 | 7,829 | 675 | 7,154 | 8.62 % |
| Test | 42–49 | 36,824 | 8,841 | 408 | 8,433 | 4.62 % |

### Label definitions

* **Elliptic** — as distributed: class 1 illicit, class 2 licit, `unknown`
  unlabelled. Unlabelled nodes take part in message passing but never in loss
  or evaluation.
* **Author corpus** — `is_coinjoin_like` is **not** ground truth. It is the
  structural screening heuristic
  `(input_count >= 4) AND (output_count >= 4)`, recovered exactly in E06 and
  verified against every row. The manuscript states this and does not evaluate
  a detector on variables that the label is computed from without also
  reporting a disjoint-feature regime.

## Protocol rules enforced in code

1. **Thresholds** are selected on validation data and frozen before the test
   split is scored. `src/metrics.py::select_threshold` is the only sanctioned
   path to a threshold.
2. **Scalers** are fitted on training rows only.
3. **Threshold-free metrics** (ROC-AUC, PR-AUC) accompany every
   threshold-dependent one.
4. **Ablations** retrain from scratch; nothing is removed at inference only.
5. **Seeds** — 20 for the GNN experiments, 5 for the tabular sweeps, all fixed
   and listed in `src/config.py`.
6. **Uncertainty** — seed variance and test-set bootstrap are reported as the
   distinct quantities they are.

## Layout

```
research/
  src/
    config.py      paths, split boundaries, seeds, hardware budget
    data.py        Elliptic loading and chronological masks
    features.py    label-free topological features + permutation verification
    metrics.py     metric suite, threshold selection, bootstrap CIs
    models.py      DGI, GAT, FrozenDGIGATFusion, GCN, GraphSAGE
    gnn.py         graph construction, DGI and classifier training loops
    plots.py       publication figures
  experiments/
    exp01_leakage_audit.py          leakage audit of the round-1 features
    exp02_feature_ablation.py       four-way ablation requested by Reviewer 1
    exp03_topological_features.py   leakage-free replacements (negative result)
    exp04_temporal_drift.py         per-time-step analysis, protocol comparison
    exp05_gnn_main.py               main GNN experiment, both DGI regimes
    exp06_coinjoin_label_recovery.py  recovers the is_coinjoin_like rule
    exp07_script_types.py           script-type reconstruction, Taproot
    exp08_coinjoin_detection.py     non-circular CoinJoin detection
  results/    machine-readable outputs (CSV / JSON)
  figures/    publication figures
  reports/    response to reviewers
  EXPERIMENT_LOG.md   experiment ID -> change -> config -> result -> commit
```

## Reproducing

Run in order; each script is independent apart from the noted dependencies.

```bash
PY="C:/Program Files/Python310/python.exe"
$PY research/experiments/exp01_leakage_audit.py          # writes features used by E02
$PY research/experiments/exp02_feature_ablation.py       # needs E01
$PY research/experiments/exp03_topological_features.py
$PY research/experiments/exp04_temporal_drift.py
$PY research/experiments/exp05_gnn_main.py               # GPU, ~2-3 h for 20 seeds
$PY research/experiments/exp06_coinjoin_label_recovery.py
$PY research/experiments/exp07_script_types.py
$PY research/experiments/exp08_coinjoin_detection.py     # needs E06 context
```

`exp05_gnn_main.py --seeds 42` runs a single-seed smoke test in a few minutes.

## Traceability

`EXPERIMENT_LOG.md` links every experiment ID to its modification, data
version, configuration, headline result and Git commit hash. No prior version
is overwritten; superseded results remain reachable through Git history.
