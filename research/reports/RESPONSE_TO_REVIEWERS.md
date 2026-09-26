# Response to Reviewers — Round 1

We thank all three reviewers. The reports identified two defects that turned
out to be more serious than the reports themselves suggested, and one stated
limitation that turned out not to exist. We re-ran the study from the raw data
rather than editing the manuscript around the comments.

Every claim below is backed by a script in `research/experiments/`, an output
file in `research/results/`, and a Git commit recorded in
`research/EXPERIMENT_LOG.md`. Reviewers can reproduce any number from the
repository.

---

## Summary of what changed

**1. The leakage concern was correct, and worse than described.** Reviewer 1
asked us to demonstrate that validation and test labels never enter the
engineered features. We could not, because they do. `shortest_path_to_illicit`
computes a BFS distance from every confirmed illicit node, and each illicit
node is its own source, so the feature equals zero for all 4,545 illicit nodes
and for no licit node. The decision rule "SPTI == 0 ⇒ illicit" has precision
1.000 and applies on the test split. In a controlled ablation, adding this
feature drives test F1, PR-AUC and MCC to exactly **1.0000** under two
unrelated model families. Both engineered features are withdrawn.

**2. No leakage-free version of those features exists.** We checked whether the
features could be rebuilt using only training-period labels. They cannot. The
Elliptic edge set contains **zero cross-time-step edges** (0 of 234,355), so
the graph is a disjoint union of 49 per-step snapshots and no path connects a
training-period node to a test-period node. Under a strict protocol both
features become constant on validation and test. This is structural, not
incidental.

**3. The CoinJoin label is a two-variable threshold rule.** The rule was never
documented, so we recovered it: a depth-3 decision tree reproduces
`is_coinjoin_like` with accuracy `1.00000000` across all 5,884,387
transactions, and the identity `is_coinjoin_like == (input_count >= 4) AND
(output_count >= 4)` holds for every row. Both counts were among the features
the round-1 detector clustered on. The reviewers' circularity concern is
confirmed mechanically.

**4. The Taproot limitation does not exist.** All three reviewers asked us to
moderate the Taproot claims because the corpus reportedly contained no
confirmed P2TR transactions. That premise was an artefact: all six
script-type boolean columns are identically `False` for every row, so nothing
could be confirmed through them. Parsing the node-reported script descriptors
instead shows P2TR is **24.10 %** of outputs, **42.39 %** of transactions touch
Taproot, and **71.94 %** of CoinJoin-like candidates do. The corpus runs
2022-07-13 to 2025-07-01 and is post-Taproot throughout. We have therefore
strengthened rather than moderated this part of the study.

**5. The reported performance was concealing a total collapse.** Broken out by
time step as Reviewer 2 requested, the model achieves mean F1 0.8582 across
steps 35–42 and mean F1 **0.0283** across steps 43–49, with recall falling
from 0.789 to 0.000 at step 43 — the dark-market shutdown. The round-1 test
window (42–49) lies almost entirely inside the collapsed regime. This also
explains the appeal of the leaked feature: after the shock nothing learned from
the training period still works, so a feature that re-encodes the label is the
only thing holding the metric up.

---

## Reviewer 1

**1. Potential label and temporal leakage in the engineered Elliptic features.**

Confirmed, and the requested ablation is now reported in full. We reproduced
the round-1 computation exactly, recovering the manuscript's r = −0.9182
bit-identically, which establishes that the audit targets the same code path.

The four-way ablation (test split, mean over 5 seeds, validation-selected
frozen threshold):

| Feature set | RF F1 | RF PR-AUC | HGB F1 | HGB PR-AUC |
|---|---|---|---|---|
| (i) base 165 | 0.6212 | 0.5431 | 0.6126 | 0.5561 |
| (ii) + INR | 0.6333 | 0.6158 | 0.6811 | 0.6794 |
| (iii) + SPTI | 0.9990 | 1.0000 | **1.0000** | **1.0000** |
| (iv) + both | 0.9953 | 1.0000 | **1.0000** | **1.0000** |

A perfect score under two unrelated inductive biases is the signature of target
leakage. On the look-ahead question the reviewer raises: the concern cannot
arise in the way described, because there are no cross-time-step edges at all —
but that same fact makes any leakage-free reconstruction degenerate. Both
features are withdrawn from the manuscript. See E01, E02.

**2. The transductive nature of DGI pre-training should be explicitly
acknowledged.**

Accepted. We now run and report both regimes side by side: transductive (DGI
sees the whole graph) and inductive (DGI is fitted on the training-period
subgraph only, with embeddings for later nodes produced by a forward pass of
the frozen encoder). Results for the two are reported together rather than one
being presented as "the" protocol. See E05.

**3. The status of the CoinJoin labels requires clarification.**

Accepted, and the exact rule is now stated in the manuscript:
`is_coinjoin_like == (input_count >= 4) AND (output_count >= 4)`, verified
against all 5,884,387 rows. The reviewer's concern that the same variables feed
both the target and the detector is correct. We no longer describe the label as
ground truth; it is a structural screening heuristic and is named as such
throughout. See E06.

**4. Possible contamination in DBSCAN cluster-to-class assignment.**

Accepted and corrected exactly as proposed. Clusters are now fitted on
training-period transactions only, the majority-vote cluster-to-class map is
frozen, and test transactions are assigned by nearest fitted centroid without
contributing to the map. We additionally introduced a chronological split of
the corpus (train ≤ 2024-06, validate ≤ 2024-12, test 2025), where round-1 had
no temporal separation at all. See E08.

**5. Taproot-related claims should be moderated.**

Respectfully, the premise does not hold, and we ask the reviewer to re-examine
this point. The claim that the corpus contains no confirmed P2TR transactions
came from six boolean columns that are identically `False` for all 5,884,387
rows — they confirm nothing about any script type, Taproot included. The
node-reported descriptors show 24.10 % of outputs are `witness_v1_taproot`.
We have replaced the proxy analysis with direct measurement on real P2TR data
and the claims are now stronger, not weaker. See E07.

**6. The three components are better characterised as a multi-module framework
than an integrated pipeline.**

Accepted without reservation. The contribution statement and conclusion are
rewritten to describe three independently evaluated modules. No claim of joint
optimisation or end-to-end information flow remains.

**7. The title should distinguish CoinJoin analysis from general
illicit-transaction detection.**

Accepted. The title is revised to state the multi-module, two-dataset
structure and no longer implies GAT-based CoinJoin detection.

**8. Threshold-selection protocols are inconsistent across the manuscript.**

Accepted. A single protocol is now enforced in code: the threshold maximises
validation F1, is selected independently for each run, and is frozen before the
test split is scored. `research/src/metrics.py` is the only sanctioned path to
a threshold, so the inconsistency cannot recur.

**9. The GAT-without-DGI ablation should be made fully consistent.**

Accepted. The no-DGI arm is trained from scratch with the fused slot zeroed
from the first epoch, not ablated at inference. Every figure is regenerated
from the final protocol; no figure from the superseded run survives.

**10. The terminology "DGI pre-training followed by GAT fine-tuning" should be
reconsidered.**

Accepted — the reviewer's reading of the code is correct. The DGI encoder is
frozen, the GAT is trained separately, and the two representations are
concatenated. No DGI weight is updated by the supervised objective. We now call
this *fusion of a frozen self-supervised representation with a supervised one*,
and the implementation class is named `FrozenDGIGATFusion` to keep code and
manuscript aligned.

**11. Statistical significance should be interpreted more cautiously.**

Accepted. Seeds are increased from 5 to 20, and we add the test-set bootstrap
the reviewer suggested: paired bootstrap resampling of test nodes, reported as
a difference with a percentile confidence interval. Seed variance and test-set
variance are now reported as the distinct quantities they are, and the
seed-level paired t-tests are described only as evidence of seed-level
consistency.

**12. The DBSCAN result represents a substantial operating-point trade-off.**

Accepted. The result is described throughout as a conservative operating point,
not superior detection, and we now report precision–recall and FPR–FNR curves
across parameter settings rather than a single point.

---

## Reviewer 2

**Informal language.** A full editorial pass has been made; "sort of", "kind
of" and "so yeah" are removed.

**Taproot proxy (lines 76–78).** See Reviewer 1 comment 5 — the proxy is no
longer needed.

**Illicit Neighbour Ratio uses neighbouring labels.** Confirmed as leakage; the
feature is withdrawn. See E01/E02.

**DGI input may contain label-derived features.** Correct, and this is now
moot: the label-derived features are gone, so DGI is fitted on the 165
distributed features only. We also verified our replacement topological
features are label-independent by permuting every label in the dataset and
confirming the features are bit-identical.

**Suggested citation (smart-grid / WSN communication framework).** We have
considered this and respectfully do not include it. The paper concerns
capacity- and spectrum-aware communication for wireless sensor networks in
smart grids; we could not identify a defensible link to Bitcoin transaction
forensics, and adding it would not serve the reader.

**Section 3.4 cluster labelling.** Corrected. See Reviewer 1 comment 4.

**"Strict ground truth" for `is_coinjoin_like`.** Accepted; the term is removed
and the exact rule is stated. See E06.

**Figure 4 inconsistency.** Corrected; the figure and caption now agree.

**More random seeds.** Increased to 20.

**Figure 11 should indicate the validation-selected threshold.** The threshold
is now marked on the curve with the corresponding independent test performance
annotated.

**Computational cost.** Added: per-model training and inference time, parameter
counts, peak GPU memory, and hardware specification (NVIDIA RTX 4060 Laptop,
8 GB VRAM; Intel i9 13th generation; 64 GB RAM; Windows 11), together with
complexity for each module.

**Explainability.** Added: feature attribution for the tabular models and a
GNN attribution analysis, with an error analysis of the failure cases.

**Temporal performance per time step.** Added, and it proved to be one of the
study's main findings — see item 5 of the summary above.

**Conclusion overstates DGI as the primary driver.** Accepted; the claim is
now scoped to the evaluated dataset and conditions.

---

## Reviewer 3

**1. Title/scope mismatch.** Accepted; see Reviewer 1 comment 7.

**2. Circularity in ground truth.** Confirmed mechanically — the label is
exactly `(input_count >= 4) AND (output_count >= 4)`. We now report both a
regime that reproduces the circularity and one whose features are provably
disjoint from the label's inputs, so the reader can see the difference
directly. See E06, E08.

**3. Pre-Taproot limitation.** The limitation was an artefact of broken
columns; the corpus is post-Taproot throughout and is 24.10 % P2TR by output.
See Reviewer 1 comment 5 and E07.

**4. Statistical basis is thin.** Accepted; 20 seeds plus test-set bootstrap
confidence intervals.

**5. Mixing Index inconsistency (mean 8.79, sd 689.27).** Accepted. We also
found a related defect the reviewer did not see: `value_concentration_ratio`,
one of the clustering features, is degenerate — its 25th, 50th and 75th
percentiles are all exactly 1.0. Heavy-tailed quantities are now log- or
rank-transformed and the transformation is stated.

**6. Corrupted text in Section 5.1.** Corrected.

**7. Figure/table consistency.** Accepted; all figures are now seed-aggregated
with variance bands, or explicitly labelled single-run diagnostics.

**8. Reference mismatch for DBSCAN.** Accepted; corrected to Ester, Kriegel,
Sander and Xu (1996).

**9. Writing quality.** A full editorial pass has been made.

**10. Positive contributions.** We are grateful for the encouragement, but we
must correct one item. The reviewer lists the two graph-structural features as
a genuine strength to be retained. Our audit shows they are a label leak, and
they are withdrawn. We would rather lose the contribution than keep a result we
cannot defend. The honest reporting of the Chamfer-distance null result is
retained, and the study now contains several further negative results reported
in the same spirit.
