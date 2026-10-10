"""Rewrite the submission README: Word-only, and correct the diff explanation."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\sagar\Desktop\CoinJoin")
OUT = REPO / "MDPI Submission" / "Revised MDPI Submit" / "SUBMISSION"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "_tmp"

TEXT = """# MDPI Resubmission Package

**Manuscript:** Leakage-Free Evaluation of Graph-Based Bitcoin Forensics:
Externally Validated CoinJoin Screening and Illicit Transaction Detection
Across Two Datasets

**Authors:** Sagar D. Korde, Narendra M. Shekokar
**Type:** Major revision, round 2

---

## Files

| File | Purpose |
|---|---|
| `Cover_Letter.docx` | Letter to the editor summarising what changed and why |
| `Manuscript_Revised_Clean.docx` | Revised manuscript, no markup. **Submit this one.** |
| `Manuscript_Revised_Clean.pdf` | Reading copy, 56 pages |
| `Manuscript_Revised_Highlighted.docx` | Same manuscript with revised passages highlighted in yellow |
| `Response_to_Reviewers.docx` | Point-by-point reply to all three reports |
| `Figures/` | 11 figures at publication resolution, 320 dpi |
| `Supplementary/` | Experiment log, reproducibility guide, originality check |

`Manuscript_Revised_Clean.docx` is the master copy of the manuscript from this
point on. Edit it directly; do not regenerate it from anything else.

## How the highlighting was produced

Every paragraph of the revision was compared against every paragraph of the
round-1 manuscript. A paragraph with no close match in round 1 is new or
rewritten and is highlighted. **211 of 301 paragraphs (70 %)** are marked,
which reflects the scale of the revision: the manuscript grew from 10,214 to
19,106 words and most results sections were rewritten.

Unhighlighted text is round-1 prose the revision did not need to change,
concentrated in the introduction, related work and ethics sections. Spot-checks
confirm the marking is accurate: the new title, abstract and findings are
highlighted, while the original introduction and UTXO background are not.

## Before submitting: three things to check

**1. The Word file needs a layout pass.** Content, tables, figures and
citations are all present and the JCP styles are applied, but the MDPI front
matter is not laid out. Open `Manuscript_Revised_Clean.docx` and seat the title
block, author affiliations, ORCIDs, corresponding-author footnote and keywords
into the MDPI template. Check table and figure numbering afterwards.

**2. Similarity.** `Supplementary/S3_Originality_Check.md` reports 26.6 %
twelve-word overlap with the round-1 manuscript. That is self-overlap with your
own unpublished submission and is normal for a revision. It will appear in an
iThenticate report only if that text is already public as a preprint, thesis
chapter or earlier submission. If any of it is, tell the editor rather than
rewriting working prose to defeat the matcher.

**3. One open editorial decision.** Reviewer 2 suggested citing a paper on
wireless sensor network and smart grid cybersecurity. It is not in the
reference list. The topic has no defensible connection to Bitcoin transaction
forensics, and the response letter declines it with that reasoning. If you
would rather add it than argue the point, it can be placed with a one-sentence
framing.

## What this revision withdraws

Stated plainly, because an editor should not have to find it:

- The two engineered graph features, and every figure computed with them
- The reported illicit-class F1 of 0.806 and ROC-AUC of 0.992
- The 79.8 % false-positive reduction and the 0.973 silhouette figure
- The claim that self-supervised pre-training is the primary performance driver
- The description of the corpus as containing no confirmed Taproot transactions
- The Mixing Index statistics of mean 8.79 and standard deviation 689.27

## What it adds

- An external validation confirming 4,491 CoinJoin transactions from
  blockchain data, giving the screening label a measured precision of 4.07 %
- A detection result whose labels do not originate in the heuristic under
  evaluation: MCC 0.9559, bootstrap interval [0.9232, 0.9830]
- A leakage-gradient diagnostic, transferable beyond this study
- Four negative results reported as such
- 18 experiments, each with a released script and a commit hash

## Reproducibility

All code: https://github.com/sagarkorde/CoinJoin

`Supplementary/S1_Experiment_Log` maps every experiment to its modification,
configuration, headline result and commit hash.
"""


def main() -> None:
    (OUT / "README_SUBMISSION.md").write_text(TEXT, encoding="utf-8")
    ref = SCRATCH / "jcp_ref.docx"
    if not ref.exists():
        SCRATCH.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "Paper" / "jcp-template.dot", ref)
    subprocess.run(["pandoc", str(OUT / "README_SUBMISSION.md"), "-o",
                    str(OUT / "README_SUBMISSION.docx"),
                    f"--reference-doc={ref}"],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
    print("README rewritten; diff-tool explanation removed")
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.suffix == ".tex":
            print("  WARNING: tex still present:", p)
    print("  tex files in package:",
          len([p for p in OUT.rglob("*.tex")]))


if __name__ == "__main__":
    main()
