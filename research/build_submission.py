"""
Assemble the MDPI resubmission package.

Produces a clean manuscript, a copy with revised passages highlighted, the
response to reviewers, figures and supplementary material, in one folder.

Highlighting is computed rather than asserted. Each paragraph of the revised
manuscript is compared against every paragraph of the round-1 manuscript; a
paragraph with no close match in round 1 is new or rewritten and is
highlighted. latexdiff was tried first and did not terminate on sources this
divergent, and on a revision of this size its output would mark nearly every
line, which tells an editor less than a paragraph-level mark does.
"""
from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

REPO = Path(r"C:\Users\sagar\Desktop\CoinJoin")
PAPER = REPO / "Paper"
OUT = REPO / "MDPI Submission" / "Revised MDPI Submit" / "SUBMISSION"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "_tmp"

MATCH_THRESHOLD = 0.88     # below this, a paragraph counts as new or rewritten
MIN_LEN = 40               # ignore very short fragments when diffing


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # pandoc emits UTF-8; the Windows default of cp1252 mangles it.
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', **kw)


def tex_to_paragraphs(tex: Path) -> list[str]:
    """Plain-text paragraphs of a LaTeX source, via pandoc."""
    r = run(["pandoc", str(tex), "-t", "plain", "--wrap=none",
             f"--resource-path={PAPER}{';' if sys.platform == 'win32' else ':'}{PAPER / 'figures'}"])
    if r.returncode != 0:
        print("  pandoc failed on", tex.name, r.stderr[:200])
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", r.stdout)]
    return [normalise(p) for p in paras if len(p.strip()) >= MIN_LEN]


def normalise(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    return s.strip().lower()


def is_revised(para: str, old: list[str], index: dict) -> bool:
    """True when no round-1 paragraph closely matches this one."""
    n = normalise(para)
    if len(n) < MIN_LEN:
        return False
    if n in index:
        return False
    # Compare against the most plausible candidates rather than all of them.
    cands = difflib.get_close_matches(n, old, n=3, cutoff=0.6)
    if not cands:
        return True
    best = max(difflib.SequenceMatcher(None, n, c).ratio() for c in cands)
    return best < MATCH_THRESHOLD


def highlight_docx(src: Path, dst: Path, old_paras: list[str]) -> tuple[int, int]:
    """Copy src to dst, highlighting paragraphs with no round-1 counterpart."""
    import xml.etree.ElementTree as ET
    ET.register_namespace("w", NS)

    old_index = set(old_paras)
    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        blobs = {n: zin.read(n) for n in names}

    root = ET.fromstring(blobs["word/document.xml"])
    total = marked = 0
    for p in root.iter(W + "p"):
        text = "".join(t.text or "" for t in p.iter(W + "t"))
        if len(text.strip()) < MIN_LEN:
            continue
        total += 1
        if not is_revised(text, old_paras, old_index):
            continue
        marked += 1
        for r in p.iter(W + "r"):
            rpr = r.find(W + "rPr")
            if rpr is None:
                rpr = ET.Element(W + "rPr")
                r.insert(0, rpr)
            hl = ET.SubElement(rpr, W + "highlight")
            hl.set(W + "val", "yellow")

    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, blobs[n])
    return total, marked


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Figures").mkdir(exist_ok=True)
    (OUT / "Supplementary").mkdir(exist_ok=True)

    ref = SCRATCH / "jcp_ref.docx"
    shutil.copy(PAPER / "jcp-template.dot", ref)
    rp = f"{PAPER}{';' if sys.platform == 'win32' else ':'}{PAPER / 'figures'}"

    print("  converting revised manuscript to Word ...")
    clean = OUT / "Manuscript_Revised_Clean.docx"
    r = run(["pandoc", str(PAPER / "main_round2.tex"), "-o", str(clean),
             f"--reference-doc={ref}", f"--resource-path={rp}"])
    if r.returncode != 0:
        print("  FAILED:", r.stderr[:300]); return
    print(f"    {clean.name}  {clean.stat().st_size/1e6:.2f} MB")

    print("  reading round-1 manuscript for the diff ...")
    old = tex_to_paragraphs(PAPER / "main.tex")
    print(f"    {len(old)} round-1 paragraphs")

    print("  marking revised passages ...")
    marked_doc = OUT / "Manuscript_Revised_Highlighted.docx"
    total, marked = highlight_docx(clean, marked_doc, old)
    pct = marked / max(total, 1) * 100
    print(f"    {marked} of {total} paragraphs highlighted ({pct:.1f}%)")

    print("  response to reviewers ...")
    resp_md = REPO / "research" / "reports" / "RESPONSE_TO_REVIEWERS.md"
    run(["pandoc", str(resp_md), "-o", str(OUT / "Response_to_Reviewers.docx"),
         f"--reference-doc={ref}"])

    print("  figures ...")
    figs = ["fig_architecture.png", "fig_leakage_ablation.png",
            "fig_temporal_collapse.png", "fig_class_distribution.png",
            "fig_model_comparison.png", "fig_curves.png",
            "fig_operating_points.png", "fig_script_composition.png",
            "fig_dgi_loss.png", "fig_training_curves.png",
            "fig_embedding_tsne.png"]
    n = 0
    for f in figs:
        src = PAPER / "figures" / f
        if src.exists():
            shutil.copy(src, OUT / "Figures" / f); n += 1
    print(f"    {n} figures copied")

    print("  supplementary ...")
    for src, dst in [(REPO / "research" / "EXPERIMENT_LOG.md",
                      "S1_Experiment_Log.md"),
                     (REPO / "research" / "README.md",
                      "S2_Reproducibility_Guide.md")]:
        if src.exists():
            shutil.copy(src, OUT / "Supplementary" / dst)
    pdf = PAPER / "main_round2.pdf"
    if pdf.exists():
        shutil.copy(pdf, OUT / "Manuscript_Revised_Clean.pdf")
    # The LaTeX source is deliberately not shipped. This work is Word only by
    # standing instruction, and the Word manuscript is the master copy.

    print("\n  package written to", OUT)


if __name__ == "__main__":
    main()
