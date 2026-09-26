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
