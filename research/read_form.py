"""Dump the structure of a .docx form passed as an argument."""
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def cell_text(c):
    return "".join(n.text or "" for n in c.iter(W + "t")).strip()


def main() -> None:
    z = zipfile.ZipFile(sys.argv[1])
    root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W + "body")
    pi = ti = 0
    for el in body:
        if el.tag == W + "p":
            t = "".join(n.text or "" for n in el.iter(W + "t"))
            if t.strip():
                print(f"[P{pi:02d}] {t[:120]}")
            pi += 1
        elif el.tag == W + "tbl":
            rows = el.findall(W + "tr")
            print(f"[TBL{ti}] rows={len(rows)}")
            for ri, r in enumerate(rows):
                cells = [cell_text(c) for c in r.findall(W + "tc")]
                print(f"    r{ri}: " + " | ".join(c[:38] for c in cells))
            ti += 1


if __name__ == "__main__":
    main()
