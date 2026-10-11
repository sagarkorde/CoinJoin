"""
Add the CRediT Author Contributions statement to the manuscript.

Inverts the author-to-roles mapping used on the authorship change form into
the role-to-authors order MDPI's template expects, so the two documents agree.
A mismatch between them is a common reason an authorship request is returned.

Written with commas rather than the em dash MDPI's own template uses
("writing-original draft preparation"), because the em dash is banned in this
manuscript by a standing style rule. The comma form is used widely in MDPI
articles and carries the same meaning.
"""
from __future__ import annotations

import copy
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

SUB = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
           r"\SUBMISSION")
TARGET = SUB / "Manuscript_Revised_Clean.docx"

# As assigned on the authorship change form.
ROLES = {
    "Sagar D. Korde": ["Conceptualization", "Methodology", "Software",
                       "Formal analysis", "Investigation", "Data curation",
                       "Visualization", "Writing, original draft preparation",
                       "Writing, review and editing"],
    "Narendra M. Shekokar": ["Conceptualization", "Supervision",
                             "Project administration",
                             "Writing, review and editing"],
    "Ashwini Dalvi": ["Methodology", "Validation",
                      "Writing, review and editing"],
    "Anuprita Korde": ["Data curation", "Validation",
                       "Writing, review and editing"],
    "Abhijeet Pasi": ["Software", "Investigation", "Visualization",
                      "Writing, review and editing"],
    "Irfan Siddavatam": ["Conceptualization", "Methodology", "Formal analysis",
                         "Supervision", "Writing, review and editing"],
}

INITIALS = {
    "Sagar D. Korde": "S.D.K.",
    "Narendra M. Shekokar": "N.M.S.",
    "Ashwini Dalvi": "A.D.",
    "Anuprita Korde": "A.K.",
    "Abhijeet Pasi": "A.P.",
    "Irfan Siddavatam": "I.S.",
}

# MDPI lists roles in taxonomy order; resources and funding acquisition do not
# apply, the latter because no external funding was received.
ORDER = ["Conceptualization", "Methodology", "Software", "Validation",
         "Formal analysis", "Investigation", "Data curation",
         "Writing, original draft preparation", "Writing, review and editing",
         "Visualization", "Supervision", "Project administration"]


def join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def build() -> str:
    parts = []
    for role in ORDER:
        who = [INITIALS[a] for a, rs in ROLES.items() if role in rs]
        if not who:
            continue
        label = role[0].lower() + role[1:] if role != "Conceptualization" else role
        parts.append(f"{label}, {join(who)}")
    body = "; ".join(parts)
    # Initials already end in a full stop, so do not add a second one.
    sep = " " if body.endswith(".") else ". "
    return ("Author Contributions. " + body + sep + "All authors have read "
            "and agreed to the published version of the manuscript.")


def main() -> None:
    statement = build()
    print(statement)
    print(f"\n({len(statement.split())} words)\n")

    ET.register_namespace("w", NS)
    with zipfile.ZipFile(TARGET) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    root = ET.fromstring(blobs["word/document.xml"])
    body = root.find(W + "body")

    anchor = None
    for p in body.iter(W + "p"):
        t = "".join(x.text or "" for x in p.iter(W + "t"))
        if re.match(r"\s*Conflict of interest", t):
            anchor = p
            break
    if anchor is None:
        print("  anchor not found")
        return

    # Remove any statement added by an earlier run.
    for p in list(body.iter(W + "p")):
        t = "".join(x.text or "" for x in p.iter(W + "t"))
        if t.strip().startswith("Author Contributions."):
            body.remove(p)

    new = copy.deepcopy(anchor)
    for child in list(new):
        if child.tag != W + "pPr":
            new.remove(child)
    r = ET.SubElement(new, W + "r")
    t = ET.SubElement(r, W + "t")
    t.text = statement
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    body.insert(list(body).index(anchor), new)

    blobs["word/document.xml"] = ET.tostring(root, encoding="utf-8",
                                             xml_declaration=True)
    tmp = TARGET.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    try:
        TARGET.unlink()
        shutil.move(tmp, TARGET)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{TARGET.name} is open in Word; close it and re-run")
    print(f"  inserted into {TARGET.name}")

    (SUB / "Author_Contributions.txt").write_text(statement, encoding="utf-8")
    print(f"  also written to Author_Contributions.txt")


if __name__ == "__main__":
    main()
