"""
Set the manuscript title across the submission package.

Keeps "MixTrace" and uses a single colon. The two tasks stay visibly separate,
since conflating them is what Reviewer 1 (item 7) and Reviewer 3 (item 1)
objected to in the original title.

The authorship change form is deliberately not touched: its title field
identifies the record in MDPI's system and was pre-filled by them.
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

OLD_FRAGMENT = "Leakage-Free Evaluation of Graph-Based Bitcoin Forensics"
NEW_TITLE = ("MixTrace: Externally Validated CoinJoin Screening and "
             "Leakage-Free Illicit Transaction Detection in Bitcoin")


def set_paragraph_text(p, text: str) -> None:
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
        style = p.find(W + "pPr/" + W + "pStyle")
        is_title = style is not None and style.get(W + "val") == "Title"
        if is_title or (OLD_FRAGMENT in txt and len(txt) < 220):
            set_paragraph_text(p, NEW_TITLE)
            n += 1
    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for nm in names:
            z.writestr(nm, blobs[nm])
    try:
        path.unlink()
        shutil.move(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name} is open in Word; close it and re-run")
    return n


def patch_md(path: Path) -> int:
    if not path.exists():
        return 0
    s = path.read_text(encoding="utf-8")
    # Collapse the old multi-line title to the new single line.
    pattern = re.compile(
        r"Leakage-Free Evaluation of Graph-Based Bitcoin\s+Forensics:\s+"
        r"Externally Validated CoinJoin Screening and\s+Illicit Transaction "
        r"Detection\s+Across Two Datasets", re.S)
    s2, n = pattern.subn(NEW_TITLE, s)
    if n == 0:
        flat = re.sub(r"\s+", " ", s)
        if "Leakage-Free Evaluation of Graph-Based Bitcoin Forensics" in flat:
            print(f"    {path.name}: title present but pattern did not match")
    path.write_text(s2, encoding="utf-8")
    return n


def main() -> None:
    print(f"New title ({len(NEW_TITLE)} chars):\n  {NEW_TITLE}\n")
    for name in ("Manuscript_Revised_Clean.docx",):
        p = SUB / name
        if not p.exists():
            continue
        try:
            print(f"  {name}: {patch_docx(p)} title paragraph(s) set")
        except RuntimeError as exc:
            print(f"  {name}: SKIPPED, {exc}")
    for name in ("Cover_Letter.md", "README_SUBMISSION.md",
                 "Authorship_Change_Justification.md"):
        p = SUB / name
        print(f"  {name}: {patch_md(p)} replacement(s)")


if __name__ == "__main__":
    main()
