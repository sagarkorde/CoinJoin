"""
Clean the accessibility caption on Table 14.

Every table carries a w:tblCaption attribute holding its caption as alt text.
Sixteen of the seventeen are clean; the one on the Chamfer linking table still
holds the conversion's own source, a LaTeX control sequence and the unresolved
equation label. The attribute is not rendered in the body, but a screen reader
reads it and MDPI's production step may pick it up.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

TARGET = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission"
              r"\Revised MDPI Submit\SUBMISSION\Manuscript_Revised_Clean.docx")

# The label is preceded by a non-breaking space, so match on the label alone.
EDITS = [
    (re.compile(r"threshold \\theta\^\*,"), "threshold,"),
    (re.compile(r"Equation[\s\u00a0]*\[eq:chamfer\]"), "Equation\u00a0(6)"),
]


def main() -> None:
    with zipfile.ZipFile(TARGET) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    raw = blobs["word/document.xml"].decode("utf-8")
    for pattern, replacement in EDITS:
        raw, n = pattern.subn(replacement, raw)
        print("  %-34s %d replacement(s)" % (pattern.pattern, n))
    blobs["word/document.xml"] = raw.encode("utf-8")

    leftover = re.findall(r"\[(?:eq|tab|sec|fig):[^\]]{1,30}\]", raw)
    print("  placeholders remaining: %d" % len(leftover))

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
