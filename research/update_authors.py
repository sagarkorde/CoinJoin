"""
Replace the author block in the Word manuscripts with the six-author list.

Writes the names as one Author paragraph in MDPI's superscript-marker style,
then the numbered affiliations and the correspondence line, so the block can be
dropped straight into the MDPI template without rearranging.

One identifier is marked for confirmation. Five ORCIDs were supplied directly
against a name. The sixth was a freshly registered record shown without a
name, and the only author otherwise unaccounted for is Anuprita Korde, so it
is attributed by elimination rather than by evidence. The script writes it
with a visible CONFIRM marker instead of presenting it as settled, because a
wrong ORCID on a published paper attributes someone's work to a stranger and
is awkward to undo.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

SUB = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
           r"\SUBMISSION")
TARGETS = ["Manuscript_Revised_Clean.docx", "Manuscript_Revised_Highlighted.docx"]

AUTHORS = (
    "Sagar D. Korde 1,2,*, Narendra M. Shekokar 1, Ashwini Dalvi 2, "
    "Anuprita Korde 2, Abhijeet Pasi 2 and Irfan Siddavatam 2"
)

AFFIL_1 = (
    "1  Department of Computer Engineering, Dwarkadas J. Sanghvi College of "
    "Engineering, Mumbai, Maharashtra 400056, India; "
    "narendra.shekokar@djsce.ac.in (N.M.S.)"
)

AFFIL_2 = (
    "2  K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, "
    "Mumbai, Maharashtra 400077, India; ashwinidalvi@somaiya.edu (A.D.); "
    "anuprita.k@somaiya.edu (A.K.); abhijeet.p@somaiya.edu (A.P.); "
    "irfansiddavatam@somaiya.edu (I.S.)"
)

CORRESP = "*  Correspondence: sagarkorde04@gmail.com"

ORCIDS = (
    "ORCID: S.D.K. 0000-0002-6289-6741; N.M.S. 0000-0002-2507-4140; "
    "A.D. 0000-0001-9015-457X; A.K. 0009-0005-2650-1414 [CONFIRM]; "
    "A.P. 0009-0002-0841-573X; I.S. 0000-0002-0858-3154"
)

BLOCK = [
    ("Author", AUTHORS),
    ("Author", AFFIL_1),
    ("Author", AFFIL_2),
    ("Author", CORRESP),
    ("Author", ORCIDS),
]


def make_paragraph(style: str, text: str, template):
    """Build a paragraph in `style`, copying run formatting from `template`."""
    p = ET.Element(W + "p")
    ppr = ET.SubElement(p, W + "pPr")
    ps = ET.SubElement(ppr, W + "pStyle")
    ps.set(W + "val", style)
    r = ET.SubElement(p, W + "r")
    tmpl_rpr = template.find(W + "r/" + W + "rPr") if template is not None else None
    if tmpl_rpr is not None:
        import copy
        r.append(copy.deepcopy(tmpl_rpr))
    t = ET.SubElement(r, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def patch(path: Path) -> tuple[int, int]:
    ET.register_namespace("w", NS)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])
    body = root.find(W + "body")

    children = list(body)
    author_idx = [i for i, el in enumerate(children)
                  if el.tag == W + "p"
                  and (el.find(W + "pPr/" + W + "pStyle") is not None)
                  and el.find(W + "pPr/" + W + "pStyle").get(W + "val") == "Author"]
    if not author_idx:
        return 0, 0

    template = children[author_idx[0]]
    first, last = author_idx[0], author_idx[-1]

    new_ps = [make_paragraph(style, text, template) for style, text in BLOCK]
    for el in children[first:last + 1]:
        body.remove(el)
    for offset, p in enumerate(new_ps):
        body.insert(first + offset, p)

    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    try:
        shutil.move(tmp, path)
    except PermissionError:
        # The file is open in Word, which holds an exclusive lock.
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name} is open in Word; close it and re-run")
    return last - first + 1, len(new_ps)


def main() -> None:
    for name in TARGETS:
        p = SUB / name
        if not p.exists():
            print(f"  {name}: not found"); continue
        try:
            removed, added = patch(p)
            print(f"  {name}: {removed} author paragraph(s) replaced by {added}")
        except RuntimeError as exc:
            print(f"  {name}: SKIPPED, {exc}")

    print("\nVerification")
    for name in TARGETS:
        p = SUB / name
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        paras = ["".join(t.text or "" for t in q.iter(W + "t"))
                 for q in root.iter(W + "p")]
        head = "\n".join(paras[:8])
        names_ok = all(n in head for n in
                       ("Sagar D. Korde", "Narendra M. Shekokar", "Ashwini Dalvi",
                        "Anuprita Korde", "Abhijeet Pasi", "Irfan Siddavatam"))
        print(f"  {name}: six authors {names_ok}, "
              f"CONFIRM marker {'[CONFIRM]' in head}")
    print("\n  front matter now reads:")
    with zipfile.ZipFile(SUB / TARGETS[0]) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    for q in list(root.iter(W + "p"))[:7]:
        t = "".join(x.text or "" for x in q.iter(W + "t"))
        if t.strip():
            print("   ", t[:120])


if __name__ == "__main__":
    main()
