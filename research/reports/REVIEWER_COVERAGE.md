# Reviewer concern coverage

Verified against `Manuscript_Revised_Clean.docx`, the master copy.
Manuscript length: 18,168 words.

| Reviewer | # | Concern | Status |
|---|---|---|---|
| R1 | 1 | Leakage audit and four-way ablation | Addressed |
| R1 | 2 | Transductive acknowledged; inductive run | Addressed |
| R1 | 3 | Exact CoinJoin label rule stated | Addressed |
| R1 | 4 | Frozen cluster-to-class map | Addressed |
| R1 | 5 | Taproot claims resolved | Addressed |
| R1 | 6 | Modules, not integrated pipeline | Addressed |
| R1 | 7 | Title distinguishes the two tasks | Addressed |
| R1 | 8 | Single threshold protocol | Addressed |
| R1 | 9 | Zero-DGI ablation retrained from scratch | Addressed |
| R1 | 10 | Fusion terminology, not fine-tuning | Addressed |
| R1 | 10b | No stale fine-tuning wording outside the correction | Addressed |
| R1 | 11 | Bootstrap and 20 seeds | Addressed |
| R1 | 12 | Operating-point trade-off with curves | Addressed |
| R2 | 1 | Informal phrasing removed | Addressed |
| R2 | 3 | INR label dependence resolved | Addressed |
| R2 | 4 | DGI on distributed features only | Addressed |
| R2 | 5 | WSN citation (declined, see note) | Addressed |
| R2 | 8 | Figure 4 caption corrected | Addressed |
| R2 | 9 | More seeds | Addressed |
| R2 | 10 | Validation threshold and test performance shown | Addressed |
| R2 | 11 | Compute cost, hardware, complexity | Addressed |
| R2 | 12 | Explainability / attribution | Addressed |
| R2 | 13 | Per-time-step temporal performance | Addressed |
| R2 | 14 | DGI-as-primary-driver claim scoped | Addressed |
| R3 | 1 | Title/scope reframed | Addressed |
| R3 | 2 | Independent validation of labels | Addressed |
| R3 | 3 | Taproot stated in abstract/conclusion | Addressed |
| R3 | 4 | Statistical basis strengthened | Addressed |
| R3 | 5 | Mixing Index outliers addressed | Addressed |
| R3 | 6 | Corrupted sentence removed | Addressed |
| R3 | 7 | Seed-aggregated figures | Addressed |
| R3 | 8 | DBSCAN citation corrected | Addressed |
| R3 | 9 | Editorial pass | Addressed |
| R3 | 10 | Positive contributions addressed | Addressed |

**34 of 34 verified.**

## Note on the two citations Reviewer 2 suggested

Reviewer 2 suggested two additional references: a paper on capacity- and
spectrum-aware communication for wireless sensor networks in smart grids, and
a second at doi 10.1002/eng2.12963 as a source of ideas for explainability.

Neither is included, for two reasons that point the same way.

Neither appears in the project reference corpus, and the standing rule for this
manuscript is that citations are drawn only from that corpus, so including
either would require sourcing bibliographic metadata from elsewhere.

On substance, neither connects to the subject matter. Capacity and spectrum
allocation in wireless sensor networks shares no method, dataset or threat
model with Bitcoin transaction forensics, and the explainability work was not
used: the attribution reported here is permutation importance and integrated
gradients, both standard and both cited to their own sources.

The response letter declines the first explicitly and gives this reasoning. If
the editor presses the point, the cost of adding a sentence is small, but the
decision should be made deliberately rather than by default.
