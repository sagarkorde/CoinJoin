"""
Regenerate the highlighted manuscript from the current clean copy.

Two improvements over the first pass.

The baseline is now the round-1 Word submission the reviewers actually read,
rather than a converted source. Same format, same content, and it is the
document the editor will compare against.

The highlighted copy is rebuilt from the clean one each time rather than
patched, so the two cannot drift. They had already drifted once: the
six-author block reached the clean copy while the highlighted copy was open in
Word and was skipped.
"""
from __future__ import annotations

import difflib
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

BASE = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit")
SUB = BASE / "SUBMISSION"
ORIGINAL = BASE / "Korde_Shekokar_MixTrace_Manuscript.docx"
CLEAN = SUB / "Manuscript_Revised_Clean.docx"
HIGHLIGHTED = SUB / "Manuscript_Revised_Highlighted.docx"

MATCH_THRESHOLD = 0.88
MIN_LEN = 40


def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    return s.strip().lower()


def paragraphs(path: Path) -> list[str]:
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("word/document.xml"))
    out = []
    for p in root.iter(W + "p"):
        t = "".join(n.text or "" for n in p.iter(W + "t"))
        if len(t.strip()) >= MIN_LEN:
            out.append(norm(t))
    return out


def is_revised(text: str, old: list[str], exact: set[str]) -> bool:
    n = norm(text)
    if len(n) < MIN_LEN:
        return False
    if n in exact:
        return False
    cands = difflib.get_close_matches(n, old, n=3, cutoff=0.6)
    if not cands:
        return True
    best = max(difflib.SequenceMatcher(None, n, c).ratio() for c in cands)
    return best < MATCH_THRESHOLD


def main() -> None:
    ET.register_namespace("w", NS)

    old = paragraphs(ORIGINAL)
    exact = set(old)
    print(f"  baseline: {ORIGINAL.name}, {len(old)} paragraphs")

    # Always rebuild from the clean copy so the two cannot diverge.
    shutil.copy(CLEAN, HIGHLIGHTED)

    with zipfile.ZipFile(HIGHLIGHTED) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])

    total = marked = 0
    for p in root.iter(W + "p"):
        text = "".join(t.text or "" for t in p.iter(W + "t"))
        if len(text.strip()) < MIN_LEN:
            continue
        total += 1
        if not is_revised(text, old, exact):
            continue
        marked += 1
        for r in p.iter(W + "r"):
            rpr = r.find(W + "rPr")
            if rpr is None:
                rpr = ET.Element(W + "rPr")
                r.insert(0, rpr)
            for existing in rpr.findall(W + "highlight"):
                rpr.remove(existing)
            hl = ET.SubElement(rpr, W + "highlight")
            hl.set(W + "val", "yellow")

    # An 86 per cent yellow document is accurate but hard to navigate, so a
    # short note at the top says what the marking means and what survived.
    body = root.find(W + "body")
    first = body.find(W + "p")
    note = ET.Element(W + "p")
    r = ET.SubElement(note, W + "r")
    rpr = ET.SubElement(r, W + "rPr")
    ET.SubElement(rpr, W + "i")
    t = ET.SubElement(r, W + "t")
    t.text = (
        "Marked-changes copy. Yellow marks text that is new or rewritten "
        f"relative to the submitted version (manuscript jcp-4517330). "
        f"{marked} of {total} substantive paragraphs are marked. The "
        "proportion is high because the revision withdrew several results, "
        "re-executed the study and added new sections; the manuscript grew "
        "from roughly 11,000 to roughly 18,500 words. Unmarked text is "
        "largely the related-work and ethics material, which the revision "
        "did not need to change. This copy is provided for review only; the "
        "clean file is the one for publication.")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    body.insert(list(body).index(first), note)

    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    tmp = HIGHLIGHTED.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    try:
        if HIGHLIGHTED.exists():
            HIGHLIGHTED.unlink()
        shutil.move(tmp, HIGHLIGHTED)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{HIGHLIGHTED.name} is open in Word; close and re-run")

    pct = marked / max(total, 1) * 100
    print(f"  highlighted {marked} of {total} paragraphs ({pct:.1f}%)")

    # Confirm the rebuild carries the current front matter.
    with zipfile.ZipFile(HIGHLIGHTED) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = ["".join(t.text or "" for t in q.iter(W + "t"))
             for q in root.iter(W + "p")]
    head = "\n".join(paras[:9])
    whole = "\n".join(paras)
    print(f"  six authors: {'Irfan Siddavatam' in head}")
    print(f"  Zenodo DOI:  {'23275812' in whole}")


if __name__ == "__main__":
    main()
