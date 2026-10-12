"""
Correct three figures on the Change of Authorship Form and its draft.

Two errors, both found by checking the form against the stored experiment
outputs and the manuscripts' own document properties.

The form says 6,269 transactions were retrieved and tested. 6,269 is the number
of raw explorer responses in the Zenodo deposit; 6,209 is the number of
candidates tested, per e14b_summary.json. The manuscript keeps the two apart
correctly, so only the form conflates them.

The form says the manuscript grew from approximately 10,200 to approximately
19,100 words. Word's own counts, held in each file's document properties, are
12,433 and 20,300. The earlier figures appear to have been token counts of the
body text, which exclude what Word counts.

The form is submitted with the revision and is read against the manuscript, so
a figure that disagrees with it invites exactly the query the form exists to
avoid.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

SUB = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
           r"\SUBMISSION")

EDITS = [
    ("6,269 Bitcoin transactions were retrieved",
     "6,209 candidate transactions were retrieved"),
    ("approximately 10,200 to approximately 19,100 words",
     "approximately 12,400 to approximately 20,300 words"),
]

TARGETS = ["jcp-4517330-authorship-change-form.docx",
           "Authorship_Change_Form_Draft.docx"]


def patch(path: Path) -> int:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    root = etree.fromstring(blobs["word/document.xml"])
    changed = 0
    # Each passage was written as a single run, so a run-local replacement is
    # enough and leaves every other run untouched.
    for t in root.iter(W + "t"):
        if not t.text:
            continue
        new = t.text
        for old, replacement in EDITS:
            new = new.replace(old, replacement)
        if new != t.text:
            t.text = new
            changed += 1
    if not changed:
        return 0

    blobs["word/document.xml"] = etree.tostring(
        root, encoding="UTF-8", xml_declaration=True, standalone=True)
    tmp = path.with_suffix(".tmp.docx")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    try:
        path.unlink()
        shutil.move(str(tmp), str(path))
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("%s is open in Word; close it and re-run" % path.name)
    return changed


def main() -> None:
    for name in TARGETS:
        path = SUB / name
        if not path.exists():
            print("  %-44s missing" % name)
            continue
        n = patch(path)
        print("  %-44s %d run(s) corrected" % (name, n))

    md = SUB / "Authorship_Change_Form_Draft.md"
    if md.exists():
        s = md.read_text(encoding="utf-8")
        before = s
        for old, replacement in EDITS:
            s = s.replace(old, replacement)
        md.write_text(s, encoding="utf-8")
        print("  %-44s %s" % (md.name, "corrected" if s != before else "no match"))


if __name__ == "__main__":
    main()
