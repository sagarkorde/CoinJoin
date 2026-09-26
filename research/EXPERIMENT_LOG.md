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
| E01 | Leakage audit of round-1 engineered features (INR, SPTI) | Elliptic raw CSVs | Directed-reverse BFS, cap 6, reproducing round-1 semantics | **Leak confirmed.** SPTI == 0 for 4,545/4,545 illicit and 0/42,019 licit nodes (precision 1.000); r = −0.9182 reproduced bit-identically. Zero cross-time-step edges, so strict train-only-seed variant is constant on val/test (SPTI ≡ 6, INR ≡ 0). Both features are unusable. | `TBD` |

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
