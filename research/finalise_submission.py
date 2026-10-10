"""Finalise the submission folder: cover letter, manifest, change summary."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\sagar\Desktop\CoinJoin")
PAPER = REPO / "Paper"
OUT = REPO / "MDPI Submission" / "Revised MDPI Submit" / "SUBMISSION"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "_tmp"


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def main() -> None:
    ref = SCRATCH / "jcp_ref.docx"
    if not ref.exists():
        shutil.copy(PAPER / "jcp-template.dot", ref)

    # Cover letter to Word.
    run(["pandoc", str(OUT / "Cover_Letter.md"), "-o",
         str(OUT / "Cover_Letter.docx"), f"--reference-doc={ref}"])

    # Originality check into the package.
    oc = REPO / "research" / "reports" / "ORIGINALITY_CHECK.md"
    if oc.exists():
        shutil.copy(oc, OUT / "Supplementary" / "S3_Originality_Check.md")

    # Experiment log as a readable Word document too.
    run(["pandoc", str(REPO / "research" / "EXPERIMENT_LOG.md"), "-o",
         str(OUT / "Supplementary" / "S1_Experiment_Log.docx"),
         f"--reference-doc={ref}"])

    manifest = """# MDPI Resubmission Package

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
| `Manuscript_Revised_Clean.pdf` | Same manuscript as typeset PDF, 56 pages |
| `Manuscript_Revised_Highlighted.docx` | Same manuscript with revised passages highlighted in yellow |
| `Response_to_Reviewers.docx` | Point-by-point reply to all three reports |
| `Figures/` | 11 figures at publication resolution, 320 dpi |
| `Supplementary/` | Experiment log, reproducibility guide, originality check, LaTeX source |

## How the highlighting was produced

Every paragraph of the revision was compared against every paragraph of the
round-1 manuscript. A paragraph with no close match in round 1 is new or
rewritten and is highlighted. **211 of 301 paragraphs (70 %)** are marked,
which reflects the scale of the revision: the manuscript grew from 10,214 to
19,106 words and most results sections were rewritten.

Unhighlighted text is round-1 prose the revision did not need to change,
concentrated in the introduction, related work and ethics sections.

`latexdiff` was tried first. It did not terminate on sources this divergent,
and on a revision of this size its output would underline nearly every line,
which tells an editor less than a paragraph-level mark does.

## Before submitting: three things to check

**1. The Word file needs a layout pass.** The manuscript was written in LaTeX
and converted to Word with the JCP template supplied as the style reference.
Content, tables, figures and citations carry across; the MDPI front matter does
not. Open `Manuscript_Revised_Clean.docx` and seat the title block, author
affiliations, ORCIDs, corresponding-author footnote and keywords into the MDPI
template manually. Check table and figure numbering after doing so.

**2. Similarity.** `Supplementary/S3_Originality_Check.md` reports 26.6 %
twelve-word overlap with the round-1 manuscript. That is self-overlap with your
own unpublished submission and is normal for a revision. It will appear in an
iThenticate report only if that text is already public as a preprint, thesis
chapter or earlier submission. If any of it is, tell the editor rather than
rewriting working prose to defeat the matcher.

**3. One open editorial decision.** Reviewer 2 suggested citing a paper on
wireless sensor network and smart grid cybersecurity. It is not in the
reference list. The topic has no defensible connection to Bitcoin transaction
forensics, and the response letter declines it with that reasoning. If you would
rather add it than argue the point, say so and it can be placed with a
one-sentence framing.

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
    (OUT / "README_SUBMISSION.md").write_text(manifest, encoding="utf-8")
    run(["pandoc", str(OUT / "README_SUBMISSION.md"), "-o",
         str(OUT / "README_SUBMISSION.docx"), f"--reference-doc={ref}"])

    tmp = OUT / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    print("package contents:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            rel = p.relative_to(OUT)
            print(f"  {str(rel):52s} {p.stat().st_size/1024:8.0f} KB")


if __name__ == "__main__":
    main()
