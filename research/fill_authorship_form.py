"""
Fill MDPI's Change of Authorship Form for jcp-4517330.

Leaves the manuscript title as MDPI pre-filled it. The title identifies the
record in their system, and the revised title is a separate change that belongs
in the revision itself, not in an authorship form.

Signature cells are left empty. Every author signs their own.
"""
from __future__ import annotations

import copy
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\sagar\Downloads\manuscript-id-authorship-change-form.docx")
DST = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
           r"\SUBMISSION\jcp-4517330-authorship-change-form.docx")

DJSCE = ("Department of Computer Engineering, Dwarkadas J. Sanghvi College of "
         "Engineering, Mumbai, Maharashtra 400056, India")
KJSCE = ("K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, "
         "Mumbai, Maharashtra 400077, India")

ORIGINAL = [
    ("Corresponding author", "Sagar D. Korde", f"{DJSCE}; {KJSCE}"),
    ("", "Narendra M. Shekokar", DJSCE),
]

NEW = [
    ("Corresponding author", "Mr.", "Sagar D. Korde", "sagarkorde04@gmail.com",
     f"{DJSCE}; {KJSCE}"),
    ("", "Dr.", "Narendra M. Shekokar", "narendra.shekokar@djsce.ac.in", DJSCE),
    ("", "Dr.", "Ashwini Dalvi", "ashwinidalvi@somaiya.edu", KJSCE),
    ("", "Ms.", "Anuprita Korde", "anuprita.k@somaiya.edu", KJSCE),
    ("", "Mr.", "Abhijeet Pasi", "abhijeet.p@somaiya.edu", KJSCE),
    ("", "Dr.", "Irfan Siddavatam", "irfansiddavatam@somaiya.edu", KJSCE),
]

CONTRIB = [
    ("Sagar D. Korde",
     "Conceptualization; Methodology; Software; Formal analysis; "
     "Investigation; Data curation; Visualization; Writing - original draft; "
     "Writing - review & editing"),
    ("Narendra M. Shekokar",
     "Conceptualization; Supervision; Project administration; "
     "Writing - review & editing"),
    ("Ashwini Dalvi",
     "Methodology; Validation; Writing - review & editing"),
    ("Anuprita Korde",
     "Data curation; Validation; Writing - review & editing"),
    ("Abhijeet Pasi",
     "Software; Investigation; Visualization; Writing - review & editing"),
    ("Irfan Siddavatam",
     "Conceptualization; Methodology; Formal analysis; Supervision; "
     "Writing - review & editing"),
]

DESCRIPTION = (
    "New author(s) have been added. Four authors are added to the list. No "
    "author is removed, and the relative order of the two authors named on the "
    "original submission is unchanged. The corresponding author is unchanged."
)

REASONS = [
    "This request is made during peer review, before acceptance and before "
    "publication. The manuscript is currently under major revision following "
    "three reviewer reports. No Correction notice is therefore required, and "
    "the request falls within the period in which the journal's policy states "
    "that authorship changes should be made.",

    "The reviewer reports prompted a revision considerably larger than the "
    "editorial pass the authors had anticipated. Two of the methodological "
    "concerns raised were investigated and confirmed, and the affected results "
    "were withdrawn rather than defended. The study was then re-executed from "
    "the raw data under corrected protocols. The revision also required work "
    "that did not exist in the original submission, most substantially an "
    "external validation of CoinJoin status in which 6,269 Bitcoin "
    "transactions were retrieved from public block explorers and tested "
    "against published protocol detection rules, confirming 4,491 as CoinJoin "
    "outputs and establishing a measurement against labels that did not "
    "originate in the heuristic being evaluated. The revised manuscript rests "
    "on eighteen experiments, each released with its source code and a "
    "version-control commit hash, and grew from approximately 10,200 to "
    "approximately 19,100 words.",

    "Work of this scope required expertise beyond that of the two original "
    "authors. Ashwini Dalvi contributed domain expertise in blockchain "
    "forensics and transaction-graph analysis, advised on the design of the "
    "external validation procedure, and critically reviewed the methodology "
    "and the revised manuscript. Anuprita Korde contributed to the "
    "construction and curation of the transaction corpus and to the "
    "verification of the derived features on which the screening analysis "
    "depends, including the data-quality checks that identified several "
    "misreported columns. Abhijeet Pasi contributed to the implementation and "
    "execution of the experimental pipeline, to the reproducibility procedures "
    "by which each result is recorded against released code, and to the "
    "preparation of the results and figures. Irfan Siddavatam contributed to "
    "the conception and framing of the revised study and advised on the "
    "evaluation protocol and the interpretation of the statistical results. "
    "These contributions were made during the revision, in response to the "
    "scope of additional work the reviewer reports required.",

    "Each author being added satisfies the ICMJE criteria the journal applies. "
    "Each made a substantial contribution to the conception, execution or "
    "analysis of the work reported; each participated in drafting or "
    "critically revising the manuscript for important intellectual content; "
    "each has approved the version now submitted; and each agrees to be "
    "accountable for the aspects of the work to which they contributed.",

    "All six authors, including both authors named on the original submission, "
    "have reviewed this request and consent to the revised author list. No "
    "author objects and no author is being removed. Signatures are provided "
    "below.",

    "Supporting material. A dataset deposit published on 10 October 2026 "
    "(DOI 10.5281/zenodo.23275812) records the four authors being added as "
    "members of the research group; this record predates the present request "
    "and is independently timestamped. The analysis repository at "
    "https://github.com/sagarkorde/CoinJoin carries a dated commit for each of "
    "the eighteen experiments, with the mapping given in the accompanying "
    "experiment log.",
]


def set_cell(tc, text: str) -> None:
    """Replace a table cell's text, preserving the first paragraph's format."""
    paras = tc.findall(W + "p")
    if not paras:
        return
    first = paras[0]
    for extra in paras[1:]:
        tc.remove(extra)
    tmpl_rpr = first.find(W + "r/" + W + "rPr")
    for r in first.findall(W + "r"):
        first.remove(r)
    r = ET.SubElement(first, W + "r")
    if tmpl_rpr is not None:
        r.append(copy.deepcopy(tmpl_rpr))
    t = ET.SubElement(r, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def para_after(anchor, texts: list[str]) -> list:
    """Build paragraphs modelled on `anchor`, one per text."""
    out = []
    for txt in texts:
        p = copy.deepcopy(anchor)
        for child in list(p):
            if child.tag != W + "pPr":
                p.remove(child)
        r = ET.SubElement(p, W + "r")
        t = ET.SubElement(r, W + "t")
        t.text = txt
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        out.append(p)
    return out


def main() -> None:
    ET.register_namespace("w", NS)
    with zipfile.ZipFile(SRC) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])
    body = root.find(W + "body")
    tables = [el for el in body if el.tag == W + "tbl"]
    print(f"  form has {len(tables)} tables")

    # Table 0 is the free-text box for the reasons.
    reasons_tbl = tables[0]
    tc = reasons_tbl.find(W + "tr/" + W + "tc")
    anchor = tc.find(W + "p")
    for child in list(tc):
        if child.tag == W + "p":
            tc.remove(child)
    for p in para_after(anchor, REASONS):
        tc.append(p)
    print(f"  reasons: {len(REASONS)} paragraphs written")

    # Description of the change sits in the paragraph that names it.
    for p in body.iter(W + "p"):
        t = "".join(n.text or "" for n in p.iter(W + "t"))
        if t.startswith("Description of the change"):
            idx = list(body).index(p)
            body.insert(idx + 1, para_after(p, [DESCRIPTION])[0])
            print("  description written")
            break

    def fill(tbl, rows, start=1, label=""):
        """Write rows into a table, failing loudly if a column is missing.

        An earlier version dropped values silently when a table had fewer
        columns than the row supplied, which is how the contribution roles
        went missing on the first pass.
        """
        trs = tbl.findall(W + "tr")
        for i, values in enumerate(rows):
            if start + i >= len(trs):
                break
            tcs = trs[start + i].findall(W + "tc")
            if len(values) > len(tcs):
                raise ValueError(
                    f"{label or 'table'} row {start+i} has {len(tcs)} column(s) "
                    f"but {len(values)} values were supplied; "
                    f"join them before writing")
            for j, v in enumerate(values):
                set_cell(tcs[j], v)
        return min(len(rows), len(trs) - start)

    n1 = fill(tables[1], ORIGINAL, label="original authorship")
    print(f"  original authorship: {n1} rows")
    n2 = fill(tables[2], NEW, label="new authorship")
    print(f"  new authorship: {n2} rows")

    # Table 3 is removals; none apply.
    trs = tables[3].findall(W + "tr")
    if len(trs) > 1:
        tcs = trs[1].findall(W + "tc")
        if len(tcs) > 1:
            set_cell(tcs[1], "None. No author is being removed.")
    print("  removals: marked none")

    n4 = fill(tables[4], [(f"{n}: {c}",) for n, c in CONTRIB], start=1,
              label="contributions")
    print(f"  contributions: {n4} rows")

    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    DST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DST.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    if DST.exists():
        DST.unlink()
    shutil.move(tmp, DST)
    print(f"\n  written: {DST}")


if __name__ == "__main__":
    main()
