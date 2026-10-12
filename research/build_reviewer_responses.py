"""
Build one response letter per reviewer in MDPI's own reply format.

The layout is not recreated by hand. The table is cloned row by row from
MDPI's "Example for author to respond reviewer" file, so the cell widths,
shading, borders, fonts and spacing are the journal's rather than ours. Rows
are deep-copied when a section needs more of them than the example provides.

Section 2 uses the question set that the JCP review report form actually puts
to reviewers, not the generic set in the example file. The two differ, and
only the former can be matched to the ticks the reviewers recorded.
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reviewer_responses_content as C  # noqa: E402

TEMPLATE = Path(r"C:\Users\sagar\Desktop\ZHS_23826\Fintech MDPI"
                r"\Fintech MDPI Revised\MDPI V6"
                r"\Example for author to respond reviewer - MDPI.docx")
OUT = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission"
           r"\Revised MDPI Submit\SUBMISSION")

# Prototype row indices in the example table.
P_TITLE, P_SEC3, P_FULL, P_QHDR, P_QROW = 0, 1, 2, 3, 4
P_SECFULL, P_COMMENT, P_RESPONSE, P_POINT = 10, 11, 12, 16


def runs_of(p):
    return p.findall(qn('w:r'))


def set_bold(run, bold: bool) -> None:
    rPr = run.find(qn('w:rPr'))
    if rPr is None:
        rPr = run.makeelement(qn('w:rPr'), {})
        run.insert(0, rPr)
    for b in rPr.findall(qn('w:b')):
        rPr.remove(b)
    for b in rPr.findall(qn('w:bCs')):
        rPr.remove(b)
    if bold:
        rPr.append(rPr.makeelement(qn('w:b'), {}))
        rPr.append(rPr.makeelement(qn('w:bCs'), {}))


def set_text(run, text: str) -> None:
    for t in run.findall(qn('w:t')):
        run.remove(t)
    for br in run.findall(qn('w:br')):
        run.remove(br)
    t = run.makeelement(qn('w:t'), {})
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    run.append(t)


def fill(tc, blocks) -> None:
    """Replace a cell's content with `blocks`.

    Each block is a string, or a (label, body) pair whose label is bold.
    Paragraph and run formatting is taken from the cell's own first paragraph
    and first run, so the example file's styling survives.
    """
    paras = tc.findall(qn('w:p'))
    if not paras:
        raise ValueError("cell has no paragraph to use as a prototype")
    proto_p = copy.deepcopy(paras[0])
    proto_runs = runs_of(proto_p)
    if not proto_runs:
        # An empty prototype cell: borrow a run from a sibling paragraph.
        for other in paras[1:]:
            if runs_of(other):
                proto_runs = [copy.deepcopy(runs_of(other)[0])]
                break
    proto_r = copy.deepcopy(proto_runs[0]) if proto_runs else None
    for r in runs_of(proto_p):
        proto_p.remove(r)

    for p in paras:
        tc.remove(p)

    if not blocks:
        blocks = [""]
    for block in blocks:
        p = copy.deepcopy(proto_p)
        segments = block if isinstance(block, (list, tuple)) else [(block, False)]
        if isinstance(block, str):
            segments = [(block, False)]
        for text, bold in segments:
            if proto_r is None:
                r = p.makeelement(qn('w:r'), {})
            else:
                r = copy.deepcopy(proto_r)
            set_bold(r, bold)
            set_text(r, text)
            p.append(r)
        tc.append(p)


def clone(proto_rows, idx):
    return copy.deepcopy(proto_rows[idx])


def cells(tr):
    return tr.findall(qn('w:tc'))


def build(n: int) -> Path:
    spec = C.REVIEWERS[n]
    doc = docx.Document(str(TEMPLATE))
    body = doc.element.body

    tbl = doc.tables[0]._tbl
    proto = tbl.findall(qn('w:tr'))
    proto_rows = [copy.deepcopy(tr) for tr in proto]

    # Strip the example file down to the one table we are about to rebuild.
    for child in list(body):
        if child is tbl or child.tag == qn('w:sectPr'):
            continue
        body.remove(child)
    for tr in proto:
        tbl.remove(tr)

    def add(idx):
        tr = clone(proto_rows, idx)
        tbl.append(tr)
        return tr

    # 0. Heading.
    fill(cells(add(P_TITLE))[0],
         [[("Response to Reviewer %d Comments" % n, True)]])

    # 1. Summary.
    tr = add(P_SEC3)
    tc = cells(tr)
    fill(tc[0], [[("1. Summary", True)]])
    fill(tc[1], [""])
    fill(tc[2], [""])
    fill(cells(add(P_FULL))[0], list(spec["summary"]))

    # 2. Questions for General Evaluation.
    tr = add(P_QHDR)
    tc = cells(tr)
    fill(tc[0], [[("2. Questions for General Evaluation", True)]])
    fill(tc[1], [[("Reviewer\u2019s Evaluation", True)]])
    fill(tc[2], [[("Response and Revisions", True)]])
    for question, (verdict, answer) in zip(C.QUESTIONS, spec["general"]):
        tc = cells(add(P_QROW))
        fill(tc[0], [question])
        fill(tc[1], [[(verdict, True)]])
        fill(tc[2], [answer])

    # 3. Point-by-point response.
    fill(cells(add(P_SECFULL))[0],
         [[("3. Point-by-point response to Comments and Suggestions for "
            "Authors", True)]])
    for i, (comment, response) in enumerate(spec["points"], start=1):
        fill(cells(add(P_COMMENT))[0],
             [[("Comments %d: " % i, True), (comment, False)]])
        blocks = [[("Response %d: " % i, True), (response[0], False)]]
        blocks += list(response[1:])
        fill(cells(add(P_RESPONSE))[0], blocks)

    # 4. English language.
    fill(cells(add(P_SECFULL))[0],
         [[("4. Response to Comments on the Quality of English Language",
            True)]])
    point, reply = spec["english"]
    fill(cells(add(P_POINT))[0], [[("Point 1: ", True), (point, False)]])
    blocks = [[("Response 1: ", True), (reply[0], False)]] + list(reply[1:])
    fill(cells(add(P_POINT))[0], blocks)

    # 5. Additional clarifications.
    fill(cells(add(P_SECFULL))[0],
         [[("5. Additional clarifications", True)]])
    fill(cells(add(P_FULL))[0], list(spec["extra"]))

    # Always build beside the target first, so a file left open in Word
    # blocks only that one move and not the rest of the run.
    out = OUT / ("Response_to_Reviewer%d.docx" % n)
    tmp = out.with_suffix(".tmp.docx")
    doc.save(str(tmp))
    try:
        if out.exists():
            out.unlink()
        shutil.move(str(tmp), str(out))
    except PermissionError:
        return None, tmp
    return out, None


def main() -> None:
    blocked = []
    for n in (1, 2, 3):
        path, pending = build(n)
        if path is None:
            blocked.append((n, pending))
            print("  Response_to_Reviewer%d.docx  OPEN IN WORD, built to %s"
                  % (n, pending.name))
            continue
        d = docx.Document(str(path))
        t = d.tables[0]
        words = sum(len(c.text.split()) for r in t.rows for c in r.cells)
        print("  %-28s %2d rows, %d comments, ~%d words"
              % (path.name, len(t.rows), len(C.REVIEWERS[n]["points"]), words))
    if blocked:
        print("\n  Close these in Word and re-run to put them in place:")
        for n, pending in blocked:
            print("    Response_to_Reviewer%d.docx" % n)


if __name__ == "__main__":
    main()
