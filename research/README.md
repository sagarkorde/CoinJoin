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
$PY research/experiments/exp08b_label_proxy_gradient.py
$PY research/experiments/exp09_explainability.py         # GPU
$PY research/experiments/exp10_statistical_analysis.py   # needs E05
$PY research/experiments/exp11_compute_cost.py           # idle GPU required
$PY research/experiments/exp12_address_columns.py
$PY research/experiments/exp13_training_diagnostics.py   # GPU
$PY research/experiments/exp14_external_ground_truth.py  # network; see below
$PY research/experiments/exp14b_verify_all.py            # network; see below
$PY research/experiments/exp15_confirmed_detection.py    # needs E14b
$PY research/experiments/exp16_detection_uncertainty.py  # needs E14b
```

`exp11_compute_cost.py` refuses to start if another process holds the GPU,
since contention invalidates its timings.

`exp05_gnn_main.py --seeds 42` runs a single-seed smoke test in a few minutes.

## Traceability

`EXPERIMENT_LOG.md` links every experiment ID to its modification, data
version, configuration, headline result and Git commit hash. No prior version
is overwritten; superseded results remain reachable through Git history.

## Network experiments, and how to avoid repeating them

E14 and E14b retrieve per-output values from public block explorers. Running
them from scratch issues roughly six thousand requests; doing so the first
time triggered rate limiting, which is why `src/explorer.py` spreads requests
across endpoints, rests one that refuses, and caps the aggregate rate.

**Do not re-run them unless the retrieval itself is what you are checking.**
The responses are deposited:

- Verification records, version DOI <https://doi.org/10.5281/zenodo.23275812>
- All versions, concept DOI <https://doi.org/10.5281/zenodo.23275811>

Download `d1_raw_api_responses_part*.jsonl.gz`, expand them into
`research/results/e14_verified/` as one JSON file per transaction id, and both
experiments will read from cache and issue no requests. `classify()` in
`exp14_external_ground_truth.py` reproduces the verdicts from those responses
alone.

## Data sources and identifiers

| Source | Identifier | Notes |
|---|---|---|
| Author-curated corpus, 5,884,387 transactions | <https://doi.org/10.21227/bxmt-mn56> | IEEE DataPort, Academic Use licence |
| Verification records, 6,269 responses | <https://doi.org/10.5281/zenodo.23275812> | CC BY 4.0 |
| Elliptic Data Set | [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | Distributed under its own terms |
| Analysis code | <https://github.com/sagarkorde/CoinJoin> | MIT |

Neither the corpus nor the Elliptic data is redistributed in the deposit. The
corpus already carries its own DOI, and re-hosting it would split citations;
Elliptic is not ours to license.
