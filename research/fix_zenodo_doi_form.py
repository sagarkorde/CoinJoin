"""
Repair the Zenodo DOIs printed in the Data Availability Statement.

Both read 10.5281/zenodo/23275812 rather than 10.5281/zenodo.23275812. The
hyperlink targets are correct, so the links work; it is the printed identifier
that is wrong, which is the one a reader copies. Every other document in the
package has the form right, so the defect is local to the manuscript.

The identifier is split across five runs inside the hyperlink, with the
separator sitting in a run of its own under a different revision id, so a plain
string replacement over the XML finds nothing. The separator run is located
instead: a run holding exactly "/" inside a hyperlink whose target names Zenodo
and whose text up to that point ends in "zenodo".
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"

SUB = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
           r"\SUBMISSION")
TARGET = SUB / "Manuscript_Revised_Clean.docx"


def hyperlink_targets(rels: bytes) -> dict[str, str]:
    root = etree.fromstring(rels)
    return {r.get("Id"): r.get("Target") or ""
            for r in root.findall(PKG + "Relationship")}


def fix_link(link, targets: dict[str, str]) -> bool:
    target = targets.get(link.get(R + "id"), "")
    if "zenodo" not in target.lower():
        return False

    runs = link.findall(W + "r")
    seen = ""
    for run in runs:
        texts = run.findall(W + "t")
        body = "".join(t.text or "" for t in texts)
        if body == "/" and seen.endswith("zenodo"):
            texts[0].text = "."
            return True
        seen += body
    return False


def main() -> None:
    with zipfile.ZipFile(TARGET) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    targets = hyperlink_targets(blobs["word/_rels/document.xml.rels"])
    root = etree.fromstring(blobs["word/document.xml"])

    fixed = 0
    for link in root.iter(W + "hyperlink"):
        if fix_link(link, targets):
            fixed += 1
    print("  %d separator(s) corrected" % fixed)
    if not fixed:
        print("  nothing to do")
        return

    blobs["word/document.xml"] = etree.tostring(
        root, encoding="UTF-8", xml_declaration=True, standalone=True)

    text = " ".join(
        "".join(t.text or "" for t in p.iter(W + "t"))
        for p in etree.fromstring(blobs["word/document.xml"]).iter(W + "p"))
    print("  slash form remaining: %d" % len(re.findall(r"10\.5281/zenodo/\d", text)))
    print("  correct form present: %s" % re.findall(r"10\.5281/zenodo\.\d+", text))

    tmp = TARGET.with_suffix(".tmp.docx")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    try:
        TARGET.unlink()
        shutil.move(str(tmp), str(TARGET))
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("%s is open in Word; close it and re-run"
                           % TARGET.name)
    print("  written to %s" % TARGET.name)


if __name__ == "__main__":
    main()
