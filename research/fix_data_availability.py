"""
Correct the Data Availability Statement in the Word manuscript.

The statement said the author corpus was "available on reasonable request".
It is in fact published on IEEE DataPort with a DOI, so the statement
understated what is available and, in a paper arguing for reproducible
evaluation, was the weakest sentence in the document.

Edits the .docx directly. The Word file is the master copy; nothing is
regenerated from another source.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

SUB = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
           r"\SUBMISSION")
TARGETS = ["Manuscript_Revised_Clean.docx", "Manuscript_Revised_Highlighted.docx"]

DATAPORT_DOI = "10.21227/bxmt-mn56"

NEW_DATA = (
    "Data availability. The Elliptic Data Set is publicly available at "
    "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set and is "
    "distributed under its own terms; it is not redistributed here. The "
    "author-curated corpus of 5,884,387 Bitcoin transactions is published on "
    "IEEE DataPort under DOI " + DATAPORT_DOI + " "
    "(https://doi.org/" + DATAPORT_DOI + "). The blockchain records used to "
    "confirm CoinJoin status, comprising the raw block-explorer responses for "
    "all 6,209 retrieved transactions together with the per-transaction "
    "verdicts derived from them, are deposited separately under CC BY 4.0 so "
    "that the verification can be repeated without issuing further requests "
    "to public infrastructure. All transaction data are public ledger records; "
    "no personal, identifying or off-chain information was collected or is "
    "distributed."
)

NEW_CODE = (
    "Code availability. The full pipeline, the experiment log and the commit "
    "hash for every experiment reported here are available at "
    "https://github.com/sagarkorde/CoinJoin under the MIT License. Each "
    "result in this article is produced by a named script recorded against "
    "its commit in that log."
)


def set_paragraph_text(p, text: str) -> None:
    """
    Replace a paragraph's text, keeping its first run's formatting.

    Word wraps a hyperlink in a <w:hyperlink> element carrying its own runs,
    which are not direct <w:r> children of the paragraph. Removing only the
    direct runs therefore leaves the link text behind and it reappears glued
    to the end of the replacement, so hyperlink elements are removed too.
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


def patch(path: Path) -> tuple[int, int]:
    ET.register_namespace("w", NS)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])

    n_data = n_code = 0
    for p in root.iter(W + "p"):
        txt = "".join(t.text or "" for t in p.iter(W + "t"))
        if re.match(r"\s*Data availability\.", txt):
            set_paragraph_text(p, NEW_DATA); n_data += 1
        elif re.match(r"\s*Code availability\.", txt):
            set_paragraph_text(p, NEW_CODE); n_code += 1

    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    shutil.move(tmp, path)
    return n_data, n_code


def main() -> None:
    for name in TARGETS:
        path = SUB / name
        if not path.exists():
            print(f"  {name}: not found"); continue
        d, c = patch(path)
        print(f"  {name}: data availability {d}, code availability {c}")

    # Confirm the correction landed and the old wording is gone.
    for name in TARGETS:
        path = SUB / name
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        txt = "\n".join("".join(t.text or "" for t in p.iter(W + "t"))
                        for p in root.iter(W + "p"))
        stale = "reasonable request" in txt
        doi = DATAPORT_DOI in txt
        print(f"  {name}: DOI present {doi}, stale wording present {stale}")


if __name__ == "__main__":
    main()
