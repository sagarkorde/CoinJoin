"""
Verify the revised Word manuscript against every reviewer concern.

Checks the delivered .docx rather than any intermediate source, since the Word
file is the master copy. Each concern is either confirmed present by evidence
found in the text, or reported as missing. Nothing is assumed.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOCX = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
            r"\SUBMISSION\Manuscript_Revised_Clean.docx")
REPORT = Path(r"C:\Users\sagar\Desktop\CoinJoin\research\reports\REVIEWER_COVERAGE.md")


def doc_text(path: Path) -> str:
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("word/document.xml"))
    parts = []
    for p in root.iter(W + "p"):
        parts.append("".join(t.text or "" for t in p.iter(W + "t")))
    return "\n".join(parts)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


# (reviewer, id, concern, kind, pattern)
#   kind "present": the pattern must appear
#   kind "absent" : the pattern must NOT appear
CHECKS = [
    ("R1", "1", "Leakage audit and four-way ablation", "present",
     r"four-way feature ablation|165 distributed.*neighbour ratio|shortest-path feature"),
    ("R1", "2", "Transductive acknowledged; inductive run", "present",
     r"inductive"),
    ("R1", "3", "Exact CoinJoin label rule stated", "present",
     r"input_count.*>=.*4|at least four inputs and at least"),
    ("R1", "4", "Frozen cluster-to-class map", "present",
     r"frozen|cluster-to-score"),
    ("R1", "5", "Taproot claims resolved", "present",
     r"24\.10"),
    ("R1", "6", "Modules, not integrated pipeline", "present",
     r"evaluated independently|no information flows between"),
    ("R1", "7", "Title distinguishes the two tasks", "present",
     r"CoinJoin Screening and Illicit Transaction"),
    ("R1", "8", "Single threshold protocol", "present",
     r"selected on validation.*frozen|frozen before the test"),
    ("R1", "9", "Zero-DGI ablation retrained from scratch", "present",
     r"zeroed from the first epoch|fused slot zeroed"),
    ("R1", "10", "Fusion terminology, not fine-tuning", "present",
     r"fusion of a frozen"),
    # The phrase survives only inside the passage that quotes and corrects it,
    # which is what R1 asked for, so the check excludes that context.
    ("R1", "10b", "No stale fine-tuning wording outside the correction",
     "absent",
     r"GAT fine-tuning(?!.{0,80}(does not describe|that phrasing))"),
    ("R1", "11", "Bootstrap and 20 seeds", "present",
     r"bootstrap"),
    ("R1", "12", "Operating-point trade-off with curves", "present",
     r"operating point|FPR.{0,3}FNR|false-positive against false-negative"),

    ("R2", "1", "Informal phrasing removed", "absent",
     r"\bsort of\b|\bkind of\b|so yeah|\byeah\b"),
    ("R2", "3", "INR label dependence resolved", "present",
     r"illicit neighbour ratio|neighbour ratio"),
    ("R2", "4", "DGI on distributed features only", "present",
     r"165 distributed features"),
    # Declined: neither paper R2 suggested appears in the project reference
    # corpus, and the standing rule is that citations come only from it.
    ("R2", "5", "WSN citation (declined, see note)", "absent",
     r"wireless sensor|smart grid"),
    ("R2", "8", "Figure 4 caption corrected", "present",
     r"unlabelled nodes are excluded|labelled nodes"),
    ("R2", "9", "More seeds", "present",
     r"twenty seeds|20 seeds|twenty-seed"),
    ("R2", "10", "Validation threshold and test performance shown", "present",
     r"validation-selected|threshold of 0\.8292"),
    ("R2", "11", "Compute cost, hardware, complexity", "present",
     r"peak memory|RTX 4060|inference"),
    ("R2", "12", "Explainability / attribution", "present",
     r"integrated gradients|permutation importance|attribution"),
    ("R2", "13", "Per-time-step temporal performance", "present",
     r"per-time-step|per time step|time step 43"),
    ("R2", "14", "DGI-as-primary-driver claim scoped", "present",
     r"withdrawn and replaced by|not the strongest model|is not the best model"),

    ("R3", "1", "Title/scope reframed", "present",
     r"Across Two Datasets"),
    ("R3", "2", "Independent validation of labels", "present",
     r"4,491|4\{,\}491"),
    ("R3", "3", "Taproot stated in abstract/conclusion", "present",
     r"24\.10 percent of outputs"),
    ("R3", "4", "Statistical basis strengthened", "present",
     r"Matthews correlation"),
    ("R3", "5", "Mixing Index outliers addressed", "present",
     r"log\(1\+x\)|skewness from 69\.2"),
    ("R3", "6", "Corrupted sentence removed", "absent",
     r"ininitializations|initializationsa|from36"),
    ("R3", "7", "Seed-aggregated figures", "present",
     r"mean over twenty seeds|error bars giving one standard deviation"),
    ("R3", "8", "DBSCAN citation corrected", "present",
     r"Ester"),
    ("R3", "9", "Editorial pass", "absent",
     r"\bsort of\b|\bkind of\b"),
    ("R3", "10", "Positive contributions addressed", "present",
     r"Chamfer|negative result"),
]


def main() -> None:
    if not DOCX.exists():
        print("manuscript not found:", DOCX)
        sys.exit(1)
    text = doc_text(DOCX)
    n = norm(text)

    rows, missing = [], []
    for rev, cid, concern, kind, pat in CHECKS:
        hit = re.search(pat, n, re.I) is not None
        ok = hit if kind == "present" else not hit
        rows.append((rev, cid, concern, kind, ok, hit))
        if not ok:
            missing.append((rev, cid, concern, kind, pat))

    print(f"manuscript: {len(text.split()):,} words\n")
    print(f"{'Rev':4} {'#':4} {'Concern':46} {'Result'}")
    print("-" * 78)
    for rev, cid, concern, kind, ok, hit in rows:
        mark = "PASS" if ok else "** CHECK **"
        print(f"{rev:4} {cid:4} {concern[:45]:46} {mark}")

    print(f"\n{len(rows) - len(missing)} / {len(rows)} verified")
    if missing:
        print("\nNeeds attention:")
        for rev, cid, concern, kind, pat in missing:
            want = "expected but not found" if kind == "present" else "should be absent but was found"
            print(f"  {rev}-{cid}  {concern}: {want}")

    lines = ["# Reviewer concern coverage",
             "",
             "Verified against `Manuscript_Revised_Clean.docx`, the master copy.",
             f"Manuscript length: {len(text.split()):,} words.",
             "",
             "| Reviewer | # | Concern | Status |",
             "|---|---|---|---|"]
    for rev, cid, concern, kind, ok, hit in rows:
        lines.append(f"| {rev} | {cid} | {concern} | "
                     f"{'Addressed' if ok else '**Check**'} |")
    lines += ["", f"**{len(rows) - len(missing)} of {len(rows)} verified.**"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
