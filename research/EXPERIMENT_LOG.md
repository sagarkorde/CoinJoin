# MixTrace — Experiment Log (Round-2 Revision)

Traceable record of every experimental milestone. Each row links an experiment
ID to the modification, data version, configuration, headline result, and the
Git commit that contains the code and outputs.

**Environment (fixed for all runs).** Python 3.10.10 · torch 2.7.1+cu128 ·
PyG 2.7.0 · scikit-learn 1.6.1 · numpy 1.26.4 · NVIDIA RTX 4060 Laptop (8 GB
VRAM) · 64 GB RAM (50 GB ceiling) · Windows 11.
Interpreter: `C:/Program Files/Python310/python.exe`.

**Data version.** Elliptic Data Set as distributed (203,769 nodes · 234,355
edges · 165 features · 49 time steps). Chronological split held fixed at
train = steps 1–34, validation = 35–41, test = 42–49.

| Split | Time steps | Nodes | Labelled | Illicit | Licit | Illicit rate |
|---|---|---|---|---|---|---|
| Train | 1–34 | 136,265 | 29,894 | 3,462 | 26,432 | 11.58 % |
| Val | 35–41 | 30,680 | 7,829 | 675 | 7,154 | 8.62 % |
| Test | 42–49 | 36,824 | 8,841 | 408 | 8,433 | 4.62 % |

---

## Experiments

| ID | Modification | Data / version | Configuration | Main result | Commit |
|---|---|---|---|---|---|
| E01 | Leakage audit of round-1 engineered features (INR, SPTI) | Elliptic raw CSVs | Directed-reverse BFS, cap 6, reproducing round-1 semantics | **Leak confirmed.** SPTI == 0 for 4,545/4,545 illicit and 0/42,019 licit nodes (precision 1.000); r = −0.9182 reproduced bit-identically. Zero cross-time-step edges, so strict train-only-seed variant is constant on val/test (SPTI ≡ 6, INR ≡ 0). Both features are unusable. | `7415004` |
| E02 | Four-way feature ablation requested by Reviewer 1 | Elliptic raw CSVs | RandomForest + HistGradientBoosting, 5 seeds, val-selected frozen threshold | **Leak quantified.** Base 165 features: F1 0.621 (RF) / 0.613 (HGB). Adding SPTI drives F1, PR-AUC and MCC to **1.0000** under both model families. INR adds +0.012 (RF) / +0.069 (HGB). Leakage-free tabular ceiling is F1 ~ 0.62. | `60dca79` |
| E03 | Label-free topological features as a replacement for the withdrawn ones | Elliptic raw CSVs | 15 descriptors computed inside each node's own time-step subgraph; permutation verification; 5 seeds x 2 models | **Clean but uninformative (negative result).** All 15 features bit-identical under full label permutation; max \|r\| with label 0.1223 (vs 0.9182 for SPTI). Adding them to the 165 base features changes F1 by -0.004 (RF) / +0.001 (HGB). Topology alone: F1 0.03, ROC-AUC 0.38 - *below chance*, indicating the topology-label relation inverts between train and test periods. | `65b137a` |
| E04 | Per-time-step evaluation and protocol comparison | Elliptic raw CSVs | RandomForest, 165 base features, 5 seeds, frozen threshold | **Baseline validated; collapse localised.** Literature protocol (train 1-25, test 35-49) gives F1 0.7634, consistent with published Elliptic RF results, so the leakage-free baseline is sound. Per step: mean F1 0.8582 on steps 35-42 vs 0.0283 on steps 43-49; recall falls 0.789 to 0.000 at step 43 (dark-market shutdown). The round-1 test window 42-49 lies almost entirely inside the collapsed regime. | `88365ed` |
| E06 | Recover the undocumented `is_coinjoin_like` rule | Author corpus, 5,884,387 tx | Decision tree fitted to structural columns | **Rule recovered exactly.** Depth-3 tree reproduces the label with accuracy 1.00000000: `is_coinjoin_like == (input_count >= 4) AND (output_count >= 4)`. The label is a deterministic function of two columns that the round-1 DBSCAN detector also clusters on, confirming the circularity all three reviewers raised. | `d55d942` |
| E07 | Reconstruct script types; test the Taproot premise | Author corpus, 5,884,387 tx | Parse `input/output_script_types` node descriptors | **Taproot premise overturned.** All six declared script flags are identically False for every row, so nothing could be confirmed through them. Parsing the descriptor columns shows P2TR is 24.10% of outputs; 2,494,168 tx (42.39%) touch Taproot, and 79,386 / 110,352 (71.94%) of label-positive CoinJoin-like candidates do. The corpus spans 2022-07 to 2025-07 and is post-Taproot throughout. | `d55d942` |
| E08 | Non-circular CoinJoin screening: chronological split, frozen cluster map | Author corpus, 600k train / 300k test | DBSCAN eps 0.6 frozen positive-rate map; RandomForest reference; 5 seeds | **Circularity reproduced and measured past.** RF with the label's own input columns: F1 0.9978. RF column-disjoint: F1 0.9140. DBSCAN frozen, all columns: F1 0.4832 (close to the round-1 figure of 0.441). DBSCAN column-disjoint: F1 0.1649. | `1b04e03` |
| E08b | Leakage gradient: how far the screening label survives feature removal | Author corpus, 600k train / 300k test | RandomForest, 6 regimes of increasing strictness, 5 seeds | **Circularity is arithmetic, not incidental.** `total_input_value / avg_input_value` recovers `input_count` for 99.96 % of rows and the reconstructed rule matches the distributed label for **100.0000 %** of sampled transactions. F1 stays at 0.90-1.00 for every regime that retains an arithmetic route, then collapses to **0.2678** once the average-value columns are removed, and to 0.1592 with temporal columns alone. | `868130c` |
| E05 | Main GNN experiment, leakage-free features, both DGI regimes | Elliptic raw CSVs, 165 distributed features | 20 seeds x 7 model/regime arms (140 runs), val-selected frozen threshold, model selection on validation PR-AUC | **Fusion helps its own architecture but is not the best model.** GraphSAGE 0.5467 > fusion-inductive 0.5197 > fusion-transductive 0.5186 > GCN 0.4997 > GAT-zero-DGI 0.4711 > GAT 0.4620. Fusion beats its own zero-DGI ablation by +0.06 F1 and halves cross-seed SD (0.030 vs 0.059). Inductive and transductive are indistinguishable. All GNNs trail the tabular baseline (0.6212). | `0db03f6` |
| E09 | Attribution and failure analysis | Elliptic, leakage-free features | Permutation importance (10 repeats) and integrated gradients (32 steps), primary seed | **Failure is a regime change, not diffuse decay.** Detected illicit transactions sit at mean time step 42.10, missed ones at 45.52, with missed cases scored 0.1882 against a 0.8292 threshold. False positives skew to hubs (mean degree 6.28 against 3.28). Tabular top-15 features are 15/15 local; graph top-15 is 9 local and 6 aggregated. | `654f1c5` |
| E10 | Uncertainty quantification | E05 outputs, Elliptic test split | 20-seed paired t-tests (df = 19) and 2,000-resample test-set bootstrap | **Two analyses, stated separately.** Seed level: GraphSAGE beats fusion (p = 0.0012); fusion beats every attention variant (p < 0.005); regimes indistinguishable (p = 0.8148). Bootstrap on the primary seed puts fusion ahead of GraphSAGE by 0.0430; intervals are ~0.09 wide, so sub-0.05 differences are not resolvable. The disagreement is reported and explained. | `654f1c5` |
| E11 | Computational cost on an idle device | Elliptic graph | Single process, refuses to run if another CUDA process is present | **The cheapest model is also the most accurate.** GraphSAGE: 8.1 s training, 947 MB peak, 29,570 params, 20.1 ms full-graph inference. Fusion: 73.1 s, 6,296 MB, 131,588 params, 78.7 ms. Attention dominates memory at ~6.3 GB against under 1 GB for the other encoders. | `654f1c5` |
| E12 | Address-count columns and the Mixing Index | Author corpus, 5,884,387 tx | Parse address strings; compare against declared counts | **Third broken column family.** `input_address_count`, `output_address_count`, `total_addresses` sit at 1, 1, 2 for almost every row and agree with parsed addresses for only 54.34 %. MI from them is identically 0.5, label correlation 0.0031. Recomputed: mean 0.678, sd 1.402, max 250.5, skew 69.2; log1p cuts skew to 7.9 and raises label correlation to **0.541**. The reported 8.79 / 689.27 matches neither computation. | `da0c952` |
| E13 | Training diagnostics regenerated, leakage-free | Elliptic, 165 features | DGI loss both regimes; supervised curves; t-SNE + linear probe | **Earlier embedding claim narrowed.** Illicit nodes form local concentrations, not distinct regions. Linear probe on the frozen representation: ROC-AUC **0.756 chronological** vs 0.939 random-split. Self-supervision does encode label-relevant structure without labels, but the in-distribution figure overstates transfer. | `da0c952` |
| E14 | External CoinJoin ground truth, pilot | Author corpus + blockstream.info | Dumplings rules, 120 verified | **Route established.** 39/40 Whirlpool candidates confirmed, 7/20 Wasabi2, 0/40 random label-positive. | `d14bf42` |
| E14b | Verify every protocol-shaped candidate | Author corpus + mempool.space / blockstream.info | 6,209 transactions, Dumplings rules on real per-output values | **4,491 confirmed CoinJoins.** Whirlpool 3,839/4,137 (92.8 %), Wasabi2 652/1,522 (42.8 %), random label-positive 0/400. Screening label precision against confirmed CoinJoins: **4.07 %** (95 % upper bound 4.97 %) at near-total recall. | `d14bf42` |

---

## E01 — Leakage audit

**Question.** Do `illicit_neighbor_ratio` and `shortest_path_to_illicit` leak
validation/test labels or look-ahead graph information, and can either be
repaired?

**Method.** `research/experiments/exp01_leakage_audit.py` rebuilds both features
with the round-1 semantics (NetworkX `DiGraph`, reversed, multi-source BFS with
`cutoff = cap − 1`), then re-derives them under a strict protocol in which the
illicit seed set is restricted to training-period nodes.

**Findings.**

1. **SPTI is an exact label re-encoding.** Every illicit node is its own BFS
   source, so SPTI = 0 for all 4,545 illicit nodes and for no licit node. The
   decision rule "SPTI == 0 ⇒ illicit" has precision 1.000 on labelled data and
   applies to the test split, where the label is supposed to be unavailable.
   The manuscript's reported r = −0.9182 is reproduced exactly, confirming the
   audit targets the same computation.

2. **Excluding self-seeding does not repair it.** Distance to the nearest
   *other* illicit seed still correlates at r = −0.7269, because the remaining
   signal comes from validation and test labels of same-step neighbours.

3. **The Elliptic graph has no cross-time-step edges** (0 of 234,355). The graph
   is a disjoint union of 49 per-step snapshots.

4. **Consequently no leakage-free variant exists.** Restricting seeds to the
   training period leaves SPTI ≡ 6 (cap) and INR ≡ 0 on *every* validation and
   test node, because no path connects the training period to a later one. The
   features are not merely contaminated; they are structurally incapable of
   carrying test-time information.

**Decision.** Both features are removed from the model. E02 quantifies the
apparent performance they contributed, and label-free topological features are
developed as a defensible replacement.

---

## E02 — Four-way feature ablation

**Question.** How much apparent test performance do the contaminated features
contribute?

**Method.** `research/experiments/exp02_feature_ablation.py`. Chronological
split; scaler fitted on training nodes only; decision threshold maximises
validation F1 and is frozen before the test split is scored; five seeds per
cell; two model families so the result does not depend on one inductive bias.

**Results (test split, mean over 5 seeds).**

| Model | Feature set | F1 | Precision | Recall | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| RandomForest | (i) base 165 | 0.6212 | 0.9231 | 0.4681 | 0.5431 | 0.8470 | 0.6470 |
| RandomForest | (ii) + INR | 0.6333 | 0.9388 | 0.4779 | 0.6158 | 0.8770 | 0.6599 |
| RandomForest | (iii) + SPTI | 0.9990 | 1.0000 | 0.9980 | 1.0000 | 1.0000 | 0.9990 |
| RandomForest | (iv) + both | 0.9953 | 1.0000 | 0.9907 | 1.0000 | 1.0000 | 0.9951 |
| HistGradientBoosting | (i) base 165 | 0.6126 | 0.8741 | 0.4716 | 0.5561 | 0.8650 | 0.6305 |
| HistGradientBoosting | (ii) + INR | 0.6811 | 0.9023 | 0.5475 | 0.6794 | 0.9103 | 0.6924 |
| HistGradientBoosting | (iii) + SPTI | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| HistGradientBoosting | (iv) + both | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Interpretation.** A perfect test score under two unrelated model families is
the signature of target leakage rather than of a strong feature. SPTI alone
saturates every metric; the model only has to learn the rule "SPTI == 0"
identified in E01. INR contributes a smaller but still illegitimate gain.

**Consequence for the manuscript.** The round-1 headline numbers are not
comparable to the literature and both features must be withdrawn. The
leakage-free ceiling for a strong tabular baseline on the 165 distributed
features is F1 ~ 0.62, which is the reference point every later experiment is
measured against.

---

## E03 — Label-free topological features

**Question.** Can the withdrawn features be replaced by descriptors that are
leakage-free by construction, and do they carry legitimate signal?

**Method.** `research/src/features.py` computes 15 topological descriptors
(degree family, clustering, PageRank, k-core, triangles, neighbour-degree
statistics, component size, two-hop size, sampled betweenness, source/sink
indicators) inside each node's own time-step subgraph. Two independent
guarantees apply: no label of any split enters the computation, and no
cross-step edge exists for a path to traverse.

**Verification.** Every label in the dataset was permuted and all features
recomputed. All 15 are bit-identical (`atol = 0`), so the features are
mechanically proven label-independent. Maximum absolute correlation with the
target is 0.1223 (`topo_component_size`), against 0.9182 for the withdrawn
SPTI. Compute cost: 47.6 s for all 203,769 nodes.

**Ablation (test split, mean over 5 seeds).**

| Model | Feature set | F1 | Precision | Recall | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| RandomForest | base 165 | 0.6212 | 0.9231 | 0.4681 | 0.5431 | 0.8470 | 0.6470 |
| RandomForest | base + topological | 0.6173 | 0.9154 | 0.4657 | 0.5412 | 0.8441 | 0.6424 |
| RandomForest | topological only | 0.0280 | 0.0167 | 0.0868 | 0.0350 | 0.3778 | −0.0788 |
| HistGradientBoosting | base 165 | 0.6126 | 0.8741 | 0.4716 | 0.5561 | 0.8650 | 0.6305 |
| HistGradientBoosting | base + topological | 0.6133 | 0.8771 | 0.4716 | 0.5547 | 0.8643 | 0.6317 |
| HistGradientBoosting | topological only | 0.0380 | 0.0230 | 0.1103 | 0.0358 | 0.3994 | −0.0570 |

**Interpretation (negative result, reported as such).** The replacement
features are clean but do not help: they move F1 by less than half a point in
either direction. More informative is the topology-only row, where ROC-AUC
falls *below* 0.5 on the test split. A below-chance ranking means the
association between topology and illicit status present in the training period
is reversed in the test period. This is direct evidence of concept drift and
motivates E04.

**Decision.** The topological features are retained in the released code and
reported as a negative control, not promoted as a contribution. The manuscript
will not claim a novel feature contribution on Elliptic.

---

## E04 — Temporal drift and the dark-market shock

**Question.** Is the leakage-free baseline weak, or is the evaluation window
unrepresentative? And how does performance behave across individual test time
steps (Reviewer 2)?

**Method.** `research/experiments/exp04_temporal_drift.py`. The same Random
Forest is evaluated under two protocols. Protocol A is this study's split
(train 1–34, val 35–41, test 42–49). Protocol B reproduces the split used in
the Elliptic literature (train 1–34, test 35–49); because a threshold may
never be taken from the test split, the last portion of the training period is
held out for that purpose, giving train 1–25, val 26–34.

**Protocol comparison (5 seeds).**

| Protocol | F1 | Precision | Recall | PR-AUC | ROC-AUC | MCC | Test n |
|---|---|---|---|---|---|---|---|
| A — this study (test 42–49) | 0.6212 ± 0.0042 | 0.9231 | 0.4681 | 0.5431 | 0.8470 | 0.6470 | 8,841 |
| B — literature (test 35–49) | 0.7634 ± 0.0259 | 0.8114 | 0.7224 | 0.7813 | 0.9128 | 0.7498 | 16,670 |

Protocol B is consistent with the Random Forest figures reported for Elliptic
in the literature, which establishes that the leakage-free baseline is sound
and that the lower protocol-A number is a property of the evaluation window.

**Per-time-step test performance (protocol B).**

| Step | Nodes | Illicit | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| 35 | 1,341 | 182 | 0.9566 | 0.9465 | 0.9670 | 0.9956 |
| 36 | 1,708 | 33 | 0.7991 | 0.6691 | 1.0000 | 0.9998 |
| 37 | 498 | 40 | 0.7666 | 0.9926 | 0.6250 | 0.9156 |
| 38 | 756 | 111 | 0.9210 | 0.9401 | 0.9027 | 0.9637 |
| 39 | 1,183 | 81 | 0.8876 | 0.8540 | 0.9259 | 0.9876 |
| 40 | 1,211 | 112 | 0.7388 | 0.8466 | 0.6571 | 0.8922 |
| 41 | 1,132 | 116 | 0.9425 | 0.9546 | 0.9310 | 0.9829 |
| 42 | 2,154 | 239 | 0.8535 | 0.9298 | 0.7891 | 0.9529 |
| **43** | 1,370 | 24 | **0.0000** | 0.0000 | **0.0000** | 0.7909 |
| 44 | 1,591 | 24 | 0.0316 | 0.0269 | 0.0417 | 0.6852 |
| 45 | 1,221 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.6392 |
| 46 | 712 | 2 | 0.1362 | 0.0810 | 0.5000 | 0.8645 |
| 47 | 846 | 22 | 0.0000 | 0.0000 | 0.0000 | 0.5727 |
| 48 | 471 | 36 | 0.0000 | 0.0000 | 0.0000 | 0.6975 |
| 49 | 476 | 56 | 0.0303 | 0.1055 | 0.0179 | 0.7032 |

**Interpretation.** Performance is bimodal, not gradually degrading. Through
step 42 the model is strong (mean F1 0.8582). At step 43 recall falls from
0.789 to 0.000 and never recovers (mean F1 0.0283 thereafter). This is the
dark-market shutdown documented in the Elliptic literature: the illicit
population after the shock is generated by a different process from the one
the model was fitted to. ROC-AUC stays well above chance (0.57–0.86) while F1
is zero, showing the ranking retains some signal but the calibrated operating
point transfers not at all.

**Consequences for the manuscript.**

1. Pooled test metrics over steps 42–49 are dominated by a regime in which the
   training distribution no longer holds. Any single aggregate number for this
   window conceals a total collapse and must be reported per step.
2. This explains the appeal of the leaked feature. After step 43 nothing
   learned from the training period still works, so a feature that re-encodes
   the label is the only thing that keeps the metric high. Leakage and drift
   are the same story told twice.
3. Threshold transfer, not ranking, is the dominant failure mode. A forensic
   deployment would need recalibration rather than retraining alone.

---

## E06 — The CoinJoin label rule

**Question.** What rule produced `is_coinjoin_like`, and does it share
variables with the detector evaluated against it?

**Method.** The rule was never documented, so it was recovered empirically:
decision trees of increasing depth were fitted to predict the label from 34
structural and behavioural columns over all 5,884,387 transactions.

**Result.** A depth-3 tree reproduces the label with training accuracy
`1.00000000`. Read off the tree, and verified directly against the column:

```
is_coinjoin_like  ==  (input_count >= 4) AND (output_count >= 4)
```

The equality holds for every one of the 5,884,387 rows. Positive rate 1.88 %
(110,352 transactions).

**Interpretation.** The label is a deterministic threshold on two count
variables, and those same two variables are among the structural features the
round-1 DBSCAN detector clusters on. The round-1 experiment therefore measured
how well clustering recovers a two-variable threshold rule, not how well it
detects CoinJoin transactions. This confirms mechanically the circularity
raised by R1 (comment 3), R2 and R3 (comment 2).

A further defect: `value_concentration_ratio`, one of the clustering features,
is degenerate. Its median, 25th and 75th percentiles are all exactly 1.0
(mean 0.99984, sd 0.0126), so it carries almost no information.

**Decision.** The label cannot be called ground truth. The manuscript must
state the rule explicitly, and the clustering experiment must be rebuilt
against a target that does not share its inputs.

---

## E07 — Script types and the Taproot claim

**Question.** Does the corpus really contain no confirmed P2TR transactions?

**Why it matters.** Round-1 stated that it does not, and used multi-input
P2WPKH as a Taproot proxy. R1 (comment 5), R2 and R3 (comment 3) all asked for
the Taproot claims to be moderated on that basis, R3 calling a pre-Taproot
validation "a substantial limitation for a paper centered on modern CoinJoin
detection".

**Finding.** The premise is an artefact of broken columns. All six declared
script-type booleans (`has_p2pk`, `has_p2pkh`, `has_p2sh`, `has_p2wpkh`,
`has_p2wsh`, `has_taproot`) are identically `False` for all 5,884,387 rows, so
no script type could ever be confirmed through them. The node-reported
descriptors in `input_script_types` / `output_script_types` carry the real
information.

**Reconstructed composition (output side).**

| Script type | Share of outputs |
|---|---|
| P2WPKH | 38.66 % |
| **P2TR (Taproot)** | **24.10 %** |
| P2SH | 15.61 % |
| P2PKH | 12.39 % |
| OP_RETURN | 6.84 % |
| P2WSH | 2.25 % |
| bare multisig | 0.15 % |

| Quantity | Value |
|---|---|
| Transactions touching Taproot (either side) | 2,494,168 (42.39 %) |
| Label-positive CoinJoin-like candidates | 110,352 |
| …of those, touching Taproot | 79,386 (**71.94 %**) |
| Corpus time range | 2022-07-13 → 2025-07-01 |

**Interpretation.** The corpus is not pre-Taproot and contains no proxy
problem. It begins eight months after Taproot activation, is roughly a quarter
Taproot by output, and nearly three quarters of its CoinJoin-like candidates
involve a Taproot script. The round-1 limitation was a data-processing defect
misread as a property of the blockchain.

**Consequence.** Rather than moderating the Taproot claims as the reviewers
proposed, the study can now support them with direct evidence. This converts
the manuscript's largest stated limitation into a genuine contribution:
script-aware CoinJoin analysis on a modern, majority-post-Taproot corpus.

---

## E08 — Non-circular CoinJoin screening

**Question.** What does the structural screening module achieve once the
cluster-to-class map is frozen on training data and the evaluation is
chronologically separated?

**Corrections applied.**

1. *Chronological split.* Round-1 had no temporal separation. The corpus is
   dense from 2022-07 to 2024-09, so the split is train up to 2023-12,
   validate to 2024-04, test from 2024-05, giving 600,000 training and 300,000
   test transactions at a 1.18 % positive rate.
2. *Frozen cluster map* (R1 comment 4). Clusters are fitted on training rows
   only, each cluster is scored by its training positive rate, the cut-off is
   selected on validation and frozen, and test rows are assigned by nearest
   fitted centroid without contributing to the map.
3. *Scoring rule.* Round-1 used majority vote per cluster. Under a 1–2 %
   positive class majority vote makes almost every cluster negative, which
   reflects the imbalance rather than the clustering, so a validation-selected
   threshold on the cluster positive rate is used instead.

**Results (test split, mean over 5 seeds).**

| Method | Feature regime | F1 | Precision | Recall | FPR | FNR | MCC |
|---|---|---|---|---|---|---|---|
| RandomForest | all columns | 0.9978 | 1.0000 | 0.9955 | 0.0000 | 0.0045 | 0.9977 |
| RandomForest | column-disjoint | 0.9140 | 0.9167 | 0.9114 | 0.0010 | 0.0886 | 0.9130 |
| DBSCAN (frozen) | all columns | 0.4832 | 0.3997 | 0.6109 | 0.0110 | 0.3891 | 0.4868 |
| DBSCAN (frozen) | column-disjoint | 0.1649 | 0.1156 | 0.6725 | 0.0879 | 0.3275 | 0.2296 |

**Interpretation.** With the label's own input columns present, a supervised
learner reaches F1 0.9978, which is simply the learner recovering the
threshold rule identified in E06. The frozen DBSCAN figure of 0.4832 is close
to the 0.441 reported in round-1, so the round-1 number is reproduced under a
corrected protocol; it is a conservative operating point, not superior
detection. The column-disjoint regime appeared at first to show genuine
residual signal at F1 0.9140. E08b shows that it does not.

---

## E08b — The leakage gradient

**Question.** E08 removed the columns the screening label is computed from and
a supervised learner still reached F1 0.914. Is that residual signal genuine
detection, or does the label survive through a proxy?

**Finding: two arithmetic routes back to the label.**

*Route 1, serialised size.* A Bitcoin transaction's size is an affine function
of its input and output counts. Regressing the counts on the size family
recovers the protocol constants directly:

| Target | Fit | R² |
|---|---|---|
| `vsize` | 20.5 + 90.75·n_in + 35.15·n_out | 0.817 |
| `weight` | 80.4 + 363.00·n_in + 140.58·n_out | 0.817 |
| `size` | 71.3 + 163.62·n_in + 36.32·n_out | 0.595 |

`weight` is exactly four times `vsize`, as the consensus rule requires.

*Route 2, value averages.* `avg_input_value` is `total_input_value` divided by
`input_count`, so the quotient of the two columns returns the count itself.
Measured over 500,000 sampled transactions:

| Quantity | Result |
|---|---|
| `total_input_value / avg_input_value` recovers `input_count` | 99.96 % of rows |
| `total_output_value / avg_output_value` recovers `output_count` | 99.98 % of rows |
| Rule rebuilt from those quotients vs the distributed label | **100.0000 % agreement** |

**Gradient (test split, mean over 5 seeds).**

| Regime | Features | F1 | Precision | Recall | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| A all columns | 31 | 0.9977 | 1.0000 | 0.9954 | 1.0000 | 1.0000 | 0.9977 |
| B column-disjoint | 18 | 0.9122 | 0.9145 | 0.9099 | 0.9729 | 0.9991 | 0.9111 |
| C size-free | 14 | 0.9085 | 0.9431 | 0.8764 | 0.9702 | 0.9994 | 0.9081 |
| D value and temporal | 12 | 0.8991 | 0.8908 | 0.9075 | 0.9684 | 0.9994 | 0.8979 |
| **E no averages** | 10 | **0.2678** | 0.3056 | 0.2390 | 0.2065 | 0.9243 | 0.2624 |
| F temporal only | 7 | 0.1592 | 0.0889 | 0.7616 | 0.0819 | 0.8900 | 0.2394 |

**Interpretation.** Performance is flat at 0.90 or above across every regime
that retains an arithmetic path to the counts, and falls by 63 points the
moment the average-value columns are removed. The cliff sits exactly where the
division route closes, which identifies the mechanism rather than merely
suggesting one. Regimes A to D therefore do not measure detection; they measure
reconstruction of a threshold rule.

**Consequence for the manuscript.** The author-curated corpus cannot support a
non-circular benchmark for structural CoinJoin screening, because the label is
a deterministic function of two counts that nearly every remaining column
encodes. The honest residual figure, once every identified route is closed, is
F1 0.2678. The screening module is reported as a characterisation study against
a stated heuristic target, and the gradient itself is reported as a
transferable diagnostic: a metric that stays flat as features are removed and
then falls off a cliff localises the leak to the columns removed at the cliff.

---

## Incident note: duplicate E05 process (2026-09-27)

A first attempt to launch E05 in the background used a `nohup ... &` form that
the task runner reported as failed. The shell reported failure only because a
subsequent `ls` ran in a different working directory; the detached Python
process itself survived and continued running.

A second, intentional launch was then started. Both processes ran the same
script against the same output paths for roughly 67 minutes, sharing one GPU.
The duplicate (PID 5420, started 00:36:55) was identified through its command
line and terminated; the intended run (PID 26444, started 00:37:30) was allowed
to finish.

**Effect on results.** None on correctness. Both processes write their CSV and
JSON outputs only after all seeds complete, so only the surviving process wrote
them. The per-seed score arrays written during the run are produced from fixed
seeds under the same configuration.

**Effect on measurements.** The per-model wall-clock timings and peak-memory
figures recorded inside E05 are contaminated for every seed that ran during the
overlap, because two processes contended for the same device. Those figures are
therefore **not reported**. Computational cost is measured separately in E11,
which runs a single process on an idle device and refuses to start if any other
CUDA process is detected.

---

## E05 — Main GNN experiment

**Question.** With the leaked features withdrawn, how do the graph models
compare, does the frozen self-supervised representation help, and does the
transductive pre-training regime account for any of the result?

**Protocol.** 165 distributed Elliptic features only. Chronological split.
Scaler statistics from labelled training nodes only. Model selection on
validation PR-AUC. Decision threshold selected on validation and frozen before
the test split is scored. Twenty seeds per arm, 140 runs in total. The no-DGI
ablation is trained from scratch with the fused slot zeroed from the first
epoch, not ablated at inference.

**Results (test split, mean over 20 seeds).**

| Model | DGI regime | F1 | SD | Precision | Recall | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| **GraphSAGE** | n/a | **0.5467** | 0.0123 | 0.7686 | 0.4257 | 0.4985 | 0.8533 | 0.5569 |
| Fusion | inductive | 0.5197 | 0.0303 | 0.7290 | 0.4064 | 0.4943 | 0.8505 | 0.5275 |
| Fusion | transductive | 0.5186 | 0.0296 | 0.7168 | 0.4091 | 0.4917 | 0.8503 | 0.5243 |
| GCN | n/a | 0.4997 | 0.0195 | 0.7549 | 0.3748 | 0.4839 | 0.8358 | 0.5165 |
| GAT, zero DGI | transductive | 0.4711 | 0.0593 | 0.5812 | 0.4069 | 0.4568 | 0.8221 | 0.4610 |
| GAT | n/a | 0.4620 | 0.0253 | 0.6015 | 0.3805 | 0.4517 | 0.8314 | 0.4564 |
| GAT, zero DGI | inductive | 0.4568 | 0.0594 | 0.5486 | 0.4048 | 0.4473 | 0.8226 | 0.4438 |

**Findings.**

1. *The frozen self-supervised representation helps the architecture it is
   fused into.* Fusion exceeds its own zero-DGI ablation by 0.049 to 0.063 F1
   and halves the cross-seed standard deviation, 0.0303 against 0.0594. The
   round-1 stability claim therefore survives the correction, restated against
   the proper ablation.

2. *The fused model is nevertheless not the best model.* GraphSAGE reaches
   0.5467 with 29,570 parameters, against 131,588 for the fusion model, and
   with the lowest variance of any arm. The round-1 claim that self-supervised
   pre-training is the primary performance driver is withdrawn and replaced by
   the narrower claim that it improves the GAT arm specifically.

3. *The transductive regime does not account for the result.* Inductive and
   transductive fusion differ by 0.0011 F1, well inside one standard deviation.
   Restricting pre-training to the training-period subgraph costs essentially
   nothing, which settles R1 comment 2: the reported gain is not an artefact of
   letting the encoder observe later nodes.

4. *Every graph model trails the tabular baseline.* The best GNN reaches
   0.5467 against 0.6212 for a Random Forest on the same features. Conclusions
   about model ordering are scoped to this benchmark and this evaluation window.

**Timings not reported.** The per-model wall-clock and peak-memory figures in
this run are contaminated by a duplicate process that shared the GPU for part
of it, as recorded in the incident note. Computational cost is measured in E11.

---

## E09 — Attribution and failure analysis

**Question.** Which features do the leakage-free models rely on, and what
characterises the cases they get wrong?

**Method.** Permutation importance on the tabular reference, shuffling each of
the 165 features in turn on the test split over ten repetitions and measuring
the loss in average precision. Integrated gradients for the fused graph model
with respect to node features, all-zero baseline, 32 steps, averaged over 400
sampled test nodes.

**Feature reliance.** Read against the documented Elliptic layout, which
separates local transaction attributes from aggregated neighbourhood features,
the two families behave differently. All fifteen of the tabular model's most
important features are local. The graph model splits nine local against six
aggregated. Five features appear in both top-fifteen lists. The graph model
distributes part of its reliance onto neighbourhood summaries, and still scores
lower, which is consistent with E03: the neighbourhood signal in this benchmark
is weaker than the local signal.

**Failure profile (fused model, threshold 0.8292).**

| Outcome | Count | Mean time step | Mean degree | Mean score |
|---|---|---|---|---|
| True positive | 182 | 42.10 | 2.29 | 0.9724 |
| False negative | 226 | 45.52 | 1.81 | 0.1882 |
| False positive | 109 | 44.00 | 6.28 | 0.9023 |
| True negative | 8,324 | 44.45 | 3.28 | 0.0803 |

**Interpretation.** The errors are distributed in time. Correctly detected
illicit transactions sit at mean step 42.10 and missed ones at 45.52, so the
model finds illicit activity before the shock of E04 and misses it afterwards.
Missed cases are scored 0.1882, far below the threshold rather than marginally
under it, so they are confident errors and no threshold adjustment recovers
them. False positives skew toward hubs, mean degree 6.28 against 3.28, an
intelligible consequence of message passing that argues for a higher evidential
bar on well-connected transactions in operational use.

---

## E10 — Uncertainty quantification

**Question.** What do repeated seeds establish, what does test-set resampling
establish, and do the two agree?

**Seed level (20 seeds, df = 19, against inductive fusion).**

| Compared arm | Mean F1 | Difference | t | p |
|---|---|---|---|---|
| GraphSAGE | 0.5467 | −0.0270 | −3.799 | 0.0012 |
| Fusion, transductive | 0.5186 | +0.0011 | 0.238 | 0.8148 |
| GCN | 0.4997 | +0.0200 | 2.654 | 0.0157 |
| GAT, zeroed, transductive | 0.4711 | +0.0486 | 3.261 | 0.0041 |
| GAT | 0.4620 | +0.0577 | 6.075 | 0.0000 |
| GAT, zeroed, inductive | 0.4568 | +0.0629 | 4.354 | 0.0003 |

**Test-set bootstrap (2,000 resamples, primary seed).** Percentile intervals
span roughly 0.09 F1 for every system, so differences below about 0.05 are not
resolvable on a test split of 8,841 labelled nodes. Fusion is placed above
GraphSAGE by 0.0430 with an interval excluding zero.

**The two disagree, and the manuscript says so.** The bootstrap describes one
trained model across test samples; the paired test averages over twenty
training runs. The primary seed happens to favour fusion and twenty seeds do
not. The twenty-seed result is taken as the conclusion, since the question is
whether an architecture is better in general. This is itself a demonstration of
the single-run hazard R3 raised.

**Bug found and fixed.** E05 wrote the string `n/a` for models with no DGI
regime. That value is in the pandas default NA list, so reading the CSV back
produced NaN and `groupby` silently dropped those rows: GraphSAGE, GCN and GAT
were missing from the first seed-level table. The reader now disables default
NA conversion, and E05 writes a non-NA sentinel.

---

## E11 — Computational cost

Measured with a single process on an idle device. The benchmark refuses to
start if any other process holds the accelerator, because contention
invalidated the timings recorded inside E05.

| Component | Train (s) | Epochs | Peak (MB) | Params | Inference (ms) |
|---|---|---|---|---|---|
| DGI, transductive | 25.36 | 200 | 1,688.5 | 54,272 | n/a |
| DGI, inductive | 15.05 | 200 | 1,312.3 | 54,272 | n/a |
| Fusion | 73.13 | 229 | 6,296.4 | 131,588 | 78.67 |
| GAT, zeroed | 15.46 | 49 | 6,298.0 | 123,396 | 76.66 |
| GAT | 38.41 | 124 | 6,298.3 | 119,106 | 75.80 |
| GCN | 9.99 | 217 | 962.2 | 14,914 | 14.14 |
| GraphSAGE | 8.12 | 152 | 946.8 | 29,570 | 20.13 |

Graph construction over 203,769 nodes and 468,710 directed edges takes 0.47 s.
Inference is a full-graph forward pass, averaged over twenty repetitions after
warm-up.

**Two points for deployment.** Attention dominates memory: the three
attention-based arms each hold about 6.3 GB at peak against under 1 GB for the
convolutional and sampling-based encoders, because per-edge coefficients must
be materialised across all 468,710 directed edges. On an 8 GB device that
leaves little headroom and a larger graph would require neighbourhood sampling.
And the cheapest model is also the most accurate: GraphSAGE trains in 8.12 s
and answers a full-graph query in 20.13 ms, against 73.13 s and 78.67 ms for a
fused model that scores lower.

---

## E12 — Address-count columns and the Mixing Index

**Question.** Reviewer 3 asked how the outliers implied by a Mixing Index of
mean 8.79 and standard deviation 689.27 were handled.

**Finding.** The premise was wrong in a way that matters more than the
question. The columns naming the input and output address sets count elements
of a one-element array holding a semicolon-joined string, not distinct
addresses, so they sit at 1, 1 and 2 for almost every transaction and agree
with the parsed address lists for only 54.34 % of rows.

| Quantity | Declared columns | Parsed addresses |
|---|---|---|
| Mean output addresses | 1.00 | 2.39 |
| Max output addresses | 1 | 3,199 |
| Mixing Index mean | 0.4998 | 0.678 |
| Mixing Index sd | 0.0111 | 1.402 |
| Mixing Index max | 0.5 | 250.5 |
| Correlation with label | 0.0031 | 0.274 |
| Correlation, log1p | 0.0031 | **0.541** |

**Decision.** The statistic is used under a log(1+x) transform, which reduces
skewness from 69.2 to 7.9 and raises label correlation to 0.541. Winsorising
at the 99.9th percentile is reported but not used, since it discards the
high-multiplicity transactions of forensic interest. The previously reported
figures match neither computation and are withdrawn.

This is the third column family in this corpus found to misreport its own
contents, after the six script-type booleans of E07 and the degenerate
`value_concentration_ratio` of E06.

---

## E13 — Training diagnostics regenerated

**Question.** Three figures came from the superseded run. One carried a
substantive claim: that illicit nodes occupy geometrically distinct regions of
the self-supervised embedding before any label is used. With a
label-re-encoding feature in the input, that separation could have been an
artefact.

**Method.** All three regenerated on the 165 distributed features alone.
Because a projection cannot settle a separability question, a linear probe is
fitted on the frozen representation, under two splits.

| Probe | ROC-AUC | What it measures |
|---|---|---|
| Random five-fold | 0.939 | In-distribution separability, mixes periods |
| Chronological | **0.756** | Transfer from training period to test period |

**Interpretation.** The claim survives in narrowed form. The representation
does encode label-relevant structure without ever observing a label, since
0.756 is well above chance on a strictly later period. But the projection does
not show distinct regions, only local concentration, and the in-distribution
figure overstates transfer by 0.18 ROC-AUC. Only the chronological value is
comparable with the rest of the study.

---

## E14 / E14b — External CoinJoin ground truth

**Question.** Every earlier experiment measured the screening heuristic against
a label produced by the same heuristic family. What does it achieve against
transactions confirmed as CoinJoin outputs from blockchain data?

**Rules.** Taken from Dumplings (github.com/nopara73/Dumplings), the reference
tool used in the CoinJoin measurement literature, not invented here.

* *Whirlpool*: native SegWit only; 5 to 10 inputs and outputs with equal
  counts; every output exactly equal; that value one of
  {0.001, 0.01, 0.05, 0.5} BTC; at least one input exactly pool-sized; every
  other input within 0.0011 BTC above the pool size.
* *Wasabi 2.x*: inputs exclusively P2WPKH or Taproot; at least 50 inputs; input
  and output values sorted descending; over 80 % of outputs drawn from the
  WabiSabi denomination set.

**Method.** The corpus records no per-output values, so nothing can be
confirmed locally. Candidates were selected by the necessary conditions the
corpus columns express, then every candidate was fetched from a public
explorer and tested against the full rule using real per-output values.
Confirmation comes from the blockchain, not from the corpus. All 6,209
transactions were verified: 5,608 fetched, 601 from cache, no failures.

**Results.**

| Group | n | Confirmed | Rate |
|---|---|---|---|
| Whirlpool-shaped | 4,137 | **3,839** | 92.8 % |
| Wasabi2-shaped | 1,522 | **652** | 42.8 % |
| `is_coinjoin_like`, random | 400 | 0 | 0.0 % |
| Label-negative, random | 150 | 0 | 0.0 % |

Confirmed Whirlpool transactions by pool: 1,753 at 0.001 BTC, 1,103 at 0.01,
780 at 0.05, 203 at 0.5. The ordering matches reported Whirlpool usage, where
the smallest pool is the busiest.

**The headline.** Stratified over the whole label-positive population, the
corpus label flags 110,352 transactions of which an estimated 4,491 are
confirmed CoinJoin outputs.

| Stratum | n | Confirmed rate |
|---|---|---|
| Whirlpool-shaped | 4,137 | 0.928 |
| Wasabi2-shaped | 1,522 | 0.428 |
| Remainder | 104,693 | 0.000 (95 % CI 0 to 0.0095) |

**Precision 4.07 %**, upper bound 4.97 %. Recall is near total, because every
confirmed transaction satisfies the label rule by construction: Whirlpool is
5-in/5-out and Wasabi2 has at least 50 of each, both of which clear the
threshold of four.

**Interpretation.** `is_coinjoin_like` is a permissive shape filter, not a
CoinJoin detector. It admits roughly twenty-four transactions for every real
CoinJoin it contains. That is a usable first-stage screen in a pipeline whose
second stage does the discriminating work, and it is not a basis for any
attribution. The figure is now measured against blockchain truth rather than
argued from the label's own definition.

**What this changes.** The study is no longer limited to characterisation. A
confirmed positive set of 4,491 transactions exists, together with 298
Whirlpool-shaped and 870 Wasabi2-shaped transactions that failed verification.
Those failures are the valuable negatives: transactions carrying CoinJoin
shape that are not CoinJoins, which is precisely what a forensic tool must
reject. E15 uses them.

**Fetching conduct.** A first pass used ten workers against one explorer, was
rate-limited with HTTP 429, and then made almost no progress while still
issuing requests. It was stopped. `research/src/explorer.py` replaced it:
requests spread across several explorers, a per-endpoint cooldown that rests
one returning 429 or 5xx while others continue, and a shared token bucket
holding the aggregate near 2.5 requests per second. The completed run recorded
27 throttle events, all absorbed without a single failed transaction.
