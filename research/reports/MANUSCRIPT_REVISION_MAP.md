# Manuscript Revision Map

Maps each round-1 finding onto the specific change required in
`Paper/main.tex` (2,244 lines, 30 bibitems). Style constraints observed
throughout: third-person phrasing only, no em dash or LaTeX `---` in prose, no
"e.g.", no first-person or possessive pronouns.

---

## Title

**Round 1.** *MixTrace: Graph Attention Network-Based Forensic Detection of
Illicit CoinJoin Transactions*

R1 (comment 7) and R3 (comment 1) both object: the GAT pipeline is evaluated on
Elliptic for generic illicit-node classification with no CoinJoin ground truth,
while CoinJoin screening is done by a separate module on a different corpus.

**Revised.** *Leakage-Free Evaluation of Graph-Based Bitcoin Forensics:
CoinJoin Screening and Illicit Transaction Detection Across Two Datasets*

The revised title separates the two tasks, signals the two-corpus structure,
and foregrounds the methodological contribution that the revision produced.

---

## Abstract (rewritten)

> Graph machine learning is increasingly applied to Bitcoin forensics, but the
> evaluation protocols that support the reported gains are rarely audited. The
> present study reports a reproducible re-evaluation of a three-module
> forensic framework covering structural CoinJoin screening, post-mix spending
> linkage, and illicit transaction classification, assessed across the
> Elliptic Data Set and a curated corpus of 5,884,387 Bitcoin transactions
> spanning July 2022 to September 2024. Three defects in the original protocol
> are identified and quantified. First, a graph feature defined as the
> shortest path to a confirmed illicit node re-encodes the target label
> exactly, taking the value zero for all 4,545 illicit nodes and for no licit
> node; supplying it to a classifier drives test F1, PR-AUC and Matthews
> correlation to 1.0000 under two unrelated model families. Because the
> Elliptic edge set contains no cross-time-step edges, no leakage-free variant
> of the feature can carry information about a test node, and the feature is
> withdrawn. Second, the screening label is shown to be the deterministic rule
> requiring at least four inputs and at least four outputs, recovered exactly
> across all transactions, so evaluating a detector that clusters on those same
> counts measures agreement rather than detection. Third, classification
> performance on Elliptic is bimodal rather than gradually degrading: mean F1
> reaches 0.858 through time step 42 and falls to 0.028 thereafter, with recall
> collapsing from 0.789 to zero at the step corresponding to a major
> marketplace closure, so any pooled figure for the later window conceals a
> complete failure. A fourth finding corrects the record on script coverage:
> the corpus was previously reported to contain no confirmed Taproot
> transactions, but all six script-type indicator columns are uniformly false,
> and parsing the node-reported descriptors shows that Taproot accounts for
> 24.10 percent of outputs and appears in 71.94 percent of screening
> candidates. Corrected protocols are supplied for every module, together with
> leakage-free baselines, inductive and transductive self-supervised regimes,
> twenty-seed repetition, test-set bootstrap intervals, feature attribution and
> failure analysis. Code, configurations and per-experiment commit hashes are
> released so that each result can be reproduced or contested.

---

## Section-by-section changes

| Section | Line | Change | Driver |
|---|---|---|---|
| §1 Introduction | 175 | Reframe contributions around the audit; remove any claim of an integrated end-to-end pipeline | R1-6, R3-1 |
| §2 Related Work | 283 | Add reproducibility and leakage literature; correct the DBSCAN citation to Ester et al. (1996) | R1, R3-8 |
| §3.2 Novel Feature Engineering | 492 | **Delete** the Elliptic four-feature subsection as a contribution. Replace with the leakage audit and the four-way ablation | R1-1, R2, R3-10 |
| §3.2.1 Author features | 498 | Retain, but state that `value_concentration_ratio` is degenerate and excluded | R3-5 |
| §3.3 Multi-Dim Chamfer | 650 | Retain the null result; describe as a bounded negative finding | R3-10 (positive) |
| §3.4 Structural DBSCAN | 692 | State the recovered label rule; add the frozen cluster-to-score protocol and the disjoint-feature regime | R1-3, R1-4, R2, R3-2 |
| §3.5 "DGI Pre-Training and GAT Fine-Tuning" | 756 | **Rename** to fusion of a frozen self-supervised representation with a supervised one; correct the description of what the code does | R1-10 |
| §3.5.1 | 759 | Add the inductive regime alongside the transductive one; state the distinction explicitly | R1-2 |
| §3.5.3 Training Protocol | 857 | Single threshold protocol: validation-selected, frozen before test, per run | R1-8 |
| §4.1 Datasets | 993 | Add the split table, class balance per period, and the corrected script-type composition | R2, R3-3 |
| §4.2 Baselines | 1053 | Add GCN, GraphSAGE, HistGradientBoosting; report the literature-protocol comparison | user request |
| §4.3 Illicit Detection Results | 1089 | Replace all numbers; add per-time-step breakdown; add bootstrap intervals | R1-11, R2, R3-4 |
| §4.4 DGI dynamics | 1227 | Regenerate from the retrained zero-DGI ablation; no superseded figure retained | R1-9 |
| §4.5 Structural Clustering | 1287 | Replace with the non-circular protocol; add the operating-point sweep | R1-12, R3-2 |
| §4.7 Statistical Validation | 1455 | Separate seed variance from test-set variance; 20 seeds; bootstrap | R1-11, R3-4 |
| §5.1 Contribution of DGI | 1603 | Fix the corrupted sentence; scope the claim to the evaluated conditions | R2, R3-6 |
| §5.5 Novel Feature Analysis | 1733 | **Replace** with the withdrawal and its quantified justification | R1-1 |
| §5.7 Limitations | 1829 | Rewrite: Taproot is no longer a limitation; add drift, label provenance, single-corpus screening | R3-3 |
| §6 Conclusion | 1898 | Remove overstated DGI-as-driver claim; scope to datasets and conditions | R2, R3 |
| New §5.x | — | **Add** "Threats to Validity in Graph-Based Forensic Evaluation" as a standalone contribution | emergent |

---

## Figures

All round-1 figures drawn from the superseded ablation are discarded. Replacements:

| Figure | Content | Source |
|---|---|---|
| F1 | Multi-module architecture, three parallel modules, no implied end-to-end flow | redrawn |
| F2 | Four-way feature ablation; leaked features saturate at 1.0 | `fig_leakage_ablation.png` |
| F3 | Per-time-step F1, recall and ROC-AUC with the shock annotated | `fig_temporal_collapse.png` |
| F4 | Corrected label distribution; caption matches content | redrawn, R2 |
| F5 | Model comparison, seed-aggregated with error bars | `fig_model_comparison.png` |
| F6 | PR and ROC curves across models | `fig_curves_with_bands.png` |
| F7 | Output script composition; Taproot share | `fig_script_composition.png` |
| F8 | FPR-FNR and precision-recall trade-off across settings | `fig_operating_points.png` |

---

## Claims withdrawn

1. The two engineered Elliptic graph features as a contribution.
2. Any headline metric computed with those features present.
3. "Taproot-aware" performance based on a P2WPKH proxy; replaced by direct measurement.
4. The framework as a jointly optimised end-to-end pipeline.
5. `is_coinjoin_like` as ground truth.
6. DGI as the primary performance driver in general; scoped to the evaluated setting.
7. Structural DBSCAN as superior detection; described as a conservative operating point.
