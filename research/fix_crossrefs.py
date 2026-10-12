"""
Resolve the unresolved cross-references left in the manuscript by conversion.

Twenty-nine placeholders survive in Manuscript_Revised_Clean.docx, each one a
broken internal hyperlink that prints its own label: "Equation [eq:cjrule]",
"Table [tab:model_comparison]", "Section [sec:class_results]". They carry the
Hyperlink character style, so they render blue and underlined.

Three things happen here.

Equations are numbered. Each label is bound to the first display-equation
paragraph that follows its first reference, which holds for all thirteen; the
numbering then follows document order. The equation paragraph is rewritten
from a centred m:oMathPara into an inline m:oMath between a centre tab and a
right tab, with the number at the right margin, which is how MDPI sets
numbered equations.

The table reference is resolved to Table 4. The table carrying the caption
"Test-split performance on the Elliptic Data Set, mean over twenty seeds" is
fourth in document order, and one surviving sentence elsewhere in the
manuscript already calls it Table 4.

The section reference is resolved to Section 4, which is where the illicit
transaction detection results sit according to the roadmap sentence in the
introduction.

Each placeholder is replaced by a plain run, so the broken hyperlink and its
blue underline go with it.
"""
from __future__ import annotations

import copy
import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

SUB = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
           r"\SUBMISSION")
TARGET = SUB / "Manuscript_Revised_Clean.docx"

# Text width = page width minus both margins, from the section properties.
# A4 at 11906 twips with 720 twip margins leaves 10466.
TEXT_WIDTH = 10466
CENTRE_TAB = TEXT_WIDTH // 2

NON_EQUATION = {
    "tab:model_comparison": "4",
    "sec:class_results": "4",
    # The manuscript contains exactly one algorithm, and Reviewer 1 item 8
    # already refers to it as Algorithm 1.
    "alg:mixtrace": "1",
}


def qn(tag: str) -> str:
    prefix, local = tag.split(":", 1)
    return {"w": W, "m": M}[prefix] + local


def para_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(qn("w:t")))


def is_display_equation(p) -> bool:
    """A paragraph whose only real content is a block-level math paragraph."""
    children = [c for c in p if c.tag != qn("w:pPr")]
    if len(children) != 1 or children[0].tag != qn("m:oMathPara"):
        return False
    return not para_text(p).strip()


def anchors_in(p) -> list[str]:
    out = []
    for link in p.iter(qn("w:hyperlink")):
        anchor = link.get(qn("w:anchor"))
        if anchor and (anchor.startswith("eq:") or anchor.startswith("tab:")
                       or anchor.startswith("sec:")):
            out.append(anchor)
    return out


def bind_equations(body) -> dict[str, int]:
    """Map each eq: label to its number, in document order."""
    flat = list(body.iter(qn("w:p")))
    first_ref: dict[str, int] = {}
    for i, p in enumerate(flat):
        for a in anchors_in(p):
            if a.startswith("eq:") and a not in first_ref:
                first_ref[a] = i

    bound: dict[str, int] = {}          # label -> index of its equation
    used: set[int] = set()
    for label, i in sorted(first_ref.items(), key=lambda kv: kv[1]):
        for j in range(i + 1, len(flat)):
            if j in used or not is_display_equation(flat[j]):
                continue
            bound[label] = j
            used.add(j)
            break
        else:
            raise RuntimeError("no display equation found after %s" % label)

    order = sorted(bound.items(), key=lambda kv: kv[1])
    return {label: n for n, (label, _) in enumerate(order, start=1)}, bound, flat


def make_run(proto, text: str):
    """A plain run carrying `text`, styled like `proto` but never a hyperlink."""
    if proto is not None:
        r = copy.deepcopy(proto)
        for child in list(r):
            if child.tag != qn("w:rPr"):
                r.remove(child)
        rPr = r.find(qn("w:rPr"))
        if rPr is not None:
            for style in rPr.findall(qn("w:rStyle")):
                rPr.remove(style)
    else:
        r = etree.Element(qn("w:r"))
    t = etree.SubElement(r, qn("w:t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r


def replace_links(body, numbers: dict[str, int]) -> int:
    replaced = 0
    for link in list(body.iter(qn("w:hyperlink"))):
        anchor = link.get(qn("w:anchor"))
        if not anchor:
            continue
        if anchor in numbers:
            text = "(%d)" % numbers[anchor]
        elif anchor in NON_EQUATION:
            text = NON_EQUATION[anchor]
        else:
            continue
        inner = link.find(qn("w:r"))
        run = make_run(inner, text)
        parent = link.getparent()
        parent.replace(link, run)
        replaced += 1
    return replaced


def number_equation(p, n: int) -> None:
    """Centre the equation and set its number at the right margin."""
    pPr = p.find(qn("w:pPr"))
    if pPr is None:
        pPr = etree.Element(qn("w:pPr"))
        p.insert(0, pPr)

    for old in pPr.findall(qn("w:tabs")):
        pPr.remove(old)
    tabs = etree.SubElement(pPr, qn("w:tabs"))
    for val, pos in (("center", CENTRE_TAB), ("right", TEXT_WIDTH)):
        tab = etree.SubElement(tabs, qn("w:tab"))
        tab.set(qn("w:val"), val)
        tab.set(qn("w:pos"), str(pos))
    for jc in pPr.findall(qn("w:jc")):
        pPr.remove(jc)

    block = p.find(qn("m:oMathPara"))
    maths = block.findall(qn("m:oMath"))
    index = list(p).index(block)
    p.remove(block)

    lead = etree.Element(qn("w:r"))
    etree.SubElement(lead, qn("w:tab"))
    p.insert(index, lead)
    for offset, math in enumerate(maths, start=1):
        p.insert(index + offset, math)

    tail = etree.Element(qn("w:r"))
    etree.SubElement(tail, qn("w:tab"))
    t = etree.SubElement(tail, qn("w:t"))
    t.text = "(%d)" % n
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    p.append(tail)


def main() -> None:
    with zipfile.ZipFile(TARGET) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    root = etree.fromstring(blobs["word/document.xml"])
    body = root.find(W + "body")

    # Re-runnable: once the equations have been converted to inline form there
    # are no block-level math paragraphs left to number, and numbering them a
    # second time would append a second number to each.
    if body.find(".//" + M + "oMathPara") is None:
        print("  equations already numbered, leaving them alone")
        numbers = {}
    else:
        numbers, bound, flat = bind_equations(body)
        print("  equations numbered in document order:")
        for label, n in sorted(numbers.items(), key=lambda kv: kv[1]):
            print("    (%2d)  %s" % (n, label))
        for label, n in numbers.items():
            number_equation(flat[bound[label]], n)

    replaced = replace_links(body, numbers)
    print("  %d placeholders replaced" % replaced)

    blobs["word/document.xml"] = etree.tostring(
        root, encoding="UTF-8", xml_declaration=True, standalone=True)

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
