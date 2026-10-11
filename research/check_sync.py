"""Report whether the clean and highlighted manuscripts are in sync."""
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
BASE = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
            r"\SUBMISSION")


def inspect(path: Path) -> dict:
    z = zipfile.ZipFile(path)
    raw = z.read("word/document.xml").decode("utf-8")
    root = ET.fromstring(raw)
    paras = ["".join(n.text or "" for n in q.iter(W + "t"))
             for q in root.iter(W + "p")]
    head = "\n".join(paras[:9])
    whole = "\n".join(paras)
    return {
        "six_authors": "Irfan Siddavatam" in head,
        "zenodo_doi": "23275812" in whole,
        "dataport_doi": "10.21227" in whole,
        "highlight_runs": raw.count("w:highlight"),
        "words": len(whole.split()),
    }


def main() -> None:
    for name in ("Manuscript_Revised_Clean.docx",
                 "Manuscript_Revised_Highlighted.docx"):
        p = BASE / name
        if not p.exists():
            print(f"{name}: missing")
            continue
        d = inspect(p)
        print(f"{name}")
        for k, v in d.items():
            print(f"    {k:16s} {v}")
    locks = list(BASE.glob("~$*"))
    if locks:
        print("\nOpen in Word (locked):")
        for l in locks:
            print("   ", l.name.replace("~$", "", 1))


if __name__ == "__main__":
    main()
