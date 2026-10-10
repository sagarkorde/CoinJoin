"""
Insert the minted Zenodo DOI wherever the deposit is referenced.

Record: https://zenodo.org/records/23275812
  version DOI  10.5281/zenodo.23275812   this exact snapshot
  concept DOI  10.5281/zenodo.23275811   all versions

The manuscript cites the version DOI, because a reader checking a number in
the article needs the files the article was built from, not whatever the
latest version happens to contain. The concept DOI is given alongside for
readers who want to follow the record forward.

Counts, reconciled: the deposit holds 6,269 explorer responses, covering the
6,209 candidates verified in the main experiment plus the earlier pilot batch.
4,491 transactions were confirmed as CoinJoin outputs.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

REPO = Path(r"C:\Users\sagar\Desktop\CoinJoin")
SUB = REPO / "MDPI Submission" / "Revised MDPI Submit" / "SUBMISSION"
BUNDLE = REPO / "MDPI Submission" / "Revised MDPI Submit" / "ZENODO_BUNDLE"

VERSION_DOI = "10.5281/zenodo.23275812"
CONCEPT_DOI = "10.5281/zenodo.23275811"
DATAPORT_DOI = "10.21227/bxmt-mn56"

NEW_DATA = (
    "Data availability. The Elliptic Data Set is publicly available at "
    "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set and is "
    "distributed under its own terms; it is not redistributed here. The "
    "author-curated corpus of 5,884,387 Bitcoin transactions is published on "
    "IEEE DataPort under DOI " + DATAPORT_DOI + ". The blockchain records used "
    "to confirm CoinJoin status are deposited under DOI " + VERSION_DOI + " "
    "(CC BY 4.0), comprising 6,269 raw block-explorer responses covering the "
    "6,209 verified candidates together with the per-transaction verdicts "
    "derived from them, so that the verification can be repeated without "
    "issuing further requests to public infrastructure. That identifier refers "
    "to the exact snapshot used here; " + CONCEPT_DOI + " resolves to all "
    "versions of the record. All transaction data are public ledger records; "
    "no personal, identifying or off-chain information was collected or is "
    "distributed."
)


def set_paragraph_text(p, text: str) -> None:
    """Replace a paragraph's text, keeping the first run's formatting.

    Hyperlinks sit in w:hyperlink elements rather than as direct runs, so they
    are removed explicitly or their text reappears after the replacement.
    """
    for link in p.findall(W + "hyperlink"):
        p.remove(link)
    runs = p.findall(W + "r")
    if not runs:
        return
    first, rest = runs[0], runs[1:]
    for t in first.findall(W + "t"):
        first.remove(t)
    for extra in rest:
        p.remove(extra)
    t = ET.SubElement(first, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def patch_docx(path: Path) -> int:
    ET.register_namespace("w", NS)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])
    n = 0
    for p in root.iter(W + "p"):
        txt = "".join(t.text or "" for t in p.iter(W + "t"))
        if re.match(r"\s*Data availability\.", txt):
            set_paragraph_text(p, NEW_DATA)
            n += 1
    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for nm in names:
            z.writestr(nm, blobs[nm])
    shutil.move(tmp, path)
    return n


def patch_text(path: Path, pairs: list[tuple[str, str]]) -> int:
    if not path.exists():
        return 0
    s = path.read_text(encoding="utf-8")
    n = 0
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new)
            n += 1
    path.write_text(s, encoding="utf-8")
    return n


def main() -> None:
    print("Word manuscripts")
    for name in ("Manuscript_Revised_Clean.docx",
                 "Manuscript_Revised_Highlighted.docx"):
        p = SUB / name
        if p.exists():
            print(f"  {name}: {patch_docx(p)} statement(s) updated")

    print("\nBundle self-reference")
    man_path = BUNDLE / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["doi"] = VERSION_DOI
        man["concept_doi"] = CONCEPT_DOI
        man["record_url"] = "https://zenodo.org/records/23275812"
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
        print("  manifest.json: DOI recorded")

    readme_add = f"""
## This record

Version DOI: <https://doi.org/{VERSION_DOI}> (this snapshot)
Concept DOI: <https://doi.org/{CONCEPT_DOI}> (all versions)

Cite the version DOI when reproducing a figure from the article, since it
resolves to the files the article was built from.
"""
    rp = BUNDLE / "README.md"
    if rp.exists() and VERSION_DOI not in rp.read_text(encoding="utf-8"):
        rp.write_text(rp.read_text(encoding="utf-8") + readme_add,
                      encoding="utf-8")
        print("  README.md: DOI section appended")

    print("\nRepository and package documentation")
    dep_line_old = "All code: https://github.com/sagarkorde/CoinJoin"
    dep_line_new = (
        "All code: https://github.com/sagarkorde/CoinJoin\n\n"
        f"Verification records: https://doi.org/{VERSION_DOI} (CC BY 4.0)\n\n"
        f"Source corpus: https://doi.org/{DATAPORT_DOI}")
    for p in (REPO / "research" / "README.md",
              SUB / "Supplementary" / "S2_Reproducibility_Guide.md"):
        if patch_text(p, [(dep_line_old, dep_line_new)]):
            print(f"  {p.name}: updated")

    sub_readme = SUB / "README_SUBMISSION.md"
    if sub_readme.exists():
        s = sub_readme.read_text(encoding="utf-8")
        old = "All code: https://github.com/sagarkorde/CoinJoin"
        new = (f"Code: https://github.com/sagarkorde/CoinJoin\n\n"
               f"Verification records: https://doi.org/{VERSION_DOI}\n\n"
               f"Source corpus: https://doi.org/{DATAPORT_DOI}")
        if old in s:
            sub_readme.write_text(s.replace(old, new, 1), encoding="utf-8")
            print("  README_SUBMISSION.md: updated")

    print("\nVerification")
    for name in ("Manuscript_Revised_Clean.docx",
                 "Manuscript_Revised_Highlighted.docx"):
        p = SUB / name
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        txt = "\n".join("".join(t.text or "" for t in q.iter(W + "t"))
                        for q in root.iter(W + "p"))
        print(f"  {name}: version DOI {VERSION_DOI in txt}, "
              f"concept DOI {CONCEPT_DOI in txt}, "
              f"stale wording {'reasonable request' in txt}")


if __name__ == "__main__":
    main()
