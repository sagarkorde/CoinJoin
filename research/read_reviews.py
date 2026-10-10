"""Extract full text and any tracked comments from the three reviewer files."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
BASE = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")


def paras(xml: bytes) -> list[str]:
    root = ET.fromstring(xml)
    out = []
    for p in root.iter(W + "p"):
        t = "".join(n.text or "" for n in p.iter(W + "t"))
        if t.strip():
            out.append(t.strip())
    return out


def main() -> None:
    for name in ("Reviewer 1.docx", "Reviewer 2.docx", "Reviewer 3.docx"):
        path = BASE / name
        print("=" * 78)
        print(name)
        print("=" * 78)
        if not path.exists():
            print("  NOT FOUND")
            continue
        z = zipfile.ZipFile(path)
        body = paras(z.read("word/document.xml"))
        print(f"[body: {len(body)} paragraphs]")
        for p in body:
            print(p)
        for extra in ("word/comments.xml", "word/footnotes.xml",
                      "word/endnotes.xml"):
            if extra in z.namelist():
                ex = paras(z.read(extra))
                ex = [e for e in ex if len(e) > 3]
                if ex:
                    print(f"\n--- {extra} ({len(ex)}) ---")
                    for e in ex:
                        print("  *", e)
        print()


if __name__ == "__main__":
    main()
