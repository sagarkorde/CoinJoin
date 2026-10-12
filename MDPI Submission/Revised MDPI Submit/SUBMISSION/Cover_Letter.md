---
title: "Cover Letter — Revised Submission"
---

# Cover Letter

**Manuscript ID:** jcp-4517330

**Title:** MixTrace: Auditing Graph-Based Bitcoin Forensics Through Leakage-Free Evaluation and External CoinJoin Validation

**Authors:** Sagar D. Korde, Narendra M. Shekokar, Ashwini Dalvi, Anuprita Korde, Abhijeet Pasi, Irfan Siddavatam

**Submission type:** Major revision

---

Dear Editor,

Please find enclosed the revised version of our manuscript, together with a
separate point-by-point response to each of the three reviewer reports.

Two matters need stating before the substance. The title has changed, at the
request of Reviewers 1 and 3, who asked that CoinJoin screening and
illicit-transaction detection be named as the separate tasks they are; the
manuscript was submitted as "MixTrace: Graph Attention Network-Based Forensic
Detection of Illicit CoinJoin Transactions in the Bitcoin Blockchain". And the
author list has changed, which is addressed below.

We should also say plainly at the outset that this revision withdraws several
of the results reported in the original submission. The reviewers raised two
methodological concerns, and investigating them showed both to be correct and
one of them more serious than the reports suggested. Rather than adjust the
text around the comments, we re-ran the study from the raw data.

**What the audit found.** A graph feature defined as the shortest path to a
confirmed illicit node re-encodes the classification target: it takes the value
zero for all 4,545 illicit nodes and for no licit node, and supplying it to a
classifier drives every test metric to exactly 1.0000 under two unrelated model
families. Because the Elliptic edge set contains no cross-time-step edges, no
leakage-free variant of the feature can carry information about a test node, so
the feature is withdrawn rather than repaired, and with it every figure
computed in its presence.

The screening label was likewise found to be a deterministic threshold on two
count variables, recoverable by arithmetic from almost every other column. And
classification performance on Elliptic proved to be bimodal rather than
gradually degrading, with mean F1 of 0.858 through time step 42 and 0.028
thereafter, so the originally reported pooled figure described no part of the
evaluation window well.

**What was added.** One stated limitation turned out not to exist. The corpus
was reported to contain no confirmed Pay-to-Taproot transactions; in fact all
six script-type indicator columns are uniformly false, and parsing the
node-reported descriptors shows Taproot accounts for 24.10 % of outputs.

More substantially, the reviewers asked for independent validation of the
CoinJoin labels. We have now obtained it. Applying published protocol rules to
per-output values retrieved from public block explorers confirms 4,491
transactions in the corpus as CoinJoin outputs. Against those confirmations the
screening label attains a precision of 4.07 %, and the confirmed transactions
support the first detection result in the study whose labels do not originate
in the heuristic being evaluated.

**What survives.** A leakage-free baseline, a measured account of where and why
these methods fail, four negative results reported as such, and a detection
result that is externally grounded. We also report that a plain GraphSAGE
baseline outperforms the architecture the original manuscript advocated, and we
have withdrawn the claim that self-supervised pre-training is the primary
driver of performance.

The revised manuscript is therefore narrower in its claims than the original
and better supported in each of them. The manuscript has grown from
approximately 12,400 words to approximately 20,300, and every results section
was rewritten.

**Where we do not follow the reviewers.** Two points should be visible to the
editor rather than buried in the responses. The first is the Taproot
limitation, which all three reports asked us to state more prominently: we have
instead removed it, because the premise rested on six corpus columns that are
uniformly false and therefore confirm nothing about any script type. The
measurement that replaces it is given in the manuscript and in our response to
each reviewer. The second is smaller. Reviewer 2 suggested two references, one
on capacity-aware and spectrum-aware communication in wireless sensor networks
and one on explainability. We have not included either, because we could not
identify a claim in the revised manuscript that they support, and we would
rather give that reason than add a citation without one. Both remain open to
your direction.

**Authorship.** The revision adds four authors, and a completed Change of
Authorship Form accompanies this submission. No author is removed and the order
of the two original authors is unchanged. The request is made during peer
review, before acceptance, and the reason is the scope of the work described
above: the study was re-executed from the raw data across eighteen experiments,
and an external validation was built that did not exist in the first
submission. The contributions of each author added are described on the form,
and the CRediT statement in the manuscript matches it. All six authors have
approved the revised manuscript and the change.

**Reproducibility.** Code, configurations, per-experiment commit hashes and a
full experiment log are released at https://github.com/sagarkorde/CoinJoin, and
the verification data, the raw block-explorer responses behind it, the
collection manifest and the checksum list are deposited at
https://doi.org/10.5281/zenodo.23275812. Any figure in the manuscript can
therefore be reproduced or contested.

We are grateful to the reviewers. The two defects they identified would have
survived into print without their scrutiny.

Yours sincerely,

Sagar D. Korde (corresponding author)
Narendra M. Shekokar
Ashwini Dalvi
Anuprita Korde
Abhijeet Pasi
Irfan Siddavatam

---

**Enclosed**

- `Manuscript_Revised_Clean.docx` — the revised manuscript
- `Manuscript_Revised_Highlighted.docx` — the same, with revised text marked
- `Response_to_Reviewer1.docx`, `Response_to_Reviewer2.docx`, `Response_to_Reviewer3.docx`
- `jcp-4517330-authorship-change-form.docx` — Change of Authorship Form
- `Figures/` — all figures at publication resolution
- `Supplementary/` — experiment log, reproduction guide, originality check
