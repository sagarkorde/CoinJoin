"""
Manuscript hygiene checks.

Run against the LaTeX source before any submission build. Covers the house
style constraints and the citation integrity problems raised in round 1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BANNED = {
    "em dash (U+2014)": "\u2014",
    "en dash (U+2013)": "\u2013",
    "LaTeX triple hyphen": None,      # handled by regex below
    "e.g.": None,
    "i.e.": None,
    "we": None,
    "our": None,
    "my": None,
    "their": None,
    "they": None,
    "this project": None,
}

WORD_PATTERNS = {
    "e.g.": r"\be\.g\.",
    "i.e.": r"\bi\.e\.",
    "we": r"\bwe\b",
    "our": r"\bour\b",
    "my": r"\bmy\b",
    "their": r"\btheir\b",
    "they": r"\bthey\b",
    "this project": r"this project",
    "LaTeX triple hyphen": r"(?<!-)---(?!-)",
}


def check(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    problems = 0

    print(f"  {path.name}: {len(src.splitlines()):,} lines")

    for name, ch in (("em dash (U+2014)", "\u2014"), ("en dash (U+2013)", "\u2013")):
        n = src.count(ch)
        if n:
            problems += n
            print(f"  [FAIL] {name}: {n} occurrence(s)")

    for name, pat in WORD_PATTERNS.items():
        hits = list(re.finditer(pat, src, re.IGNORECASE))
        if hits:
            problems += len(hits)
            lines = sorted({src[:h.start()].count("\n") + 1 for h in hits})
            print(f"  [FAIL] {name}: {len(hits)} occurrence(s) at lines "
                  f"{lines[:10]}{' ...' if len(lines) > 10 else ''}")

    # Citation integrity.
    keys = re.findall(r"\\bibitem\{([^}]+)\}", src)
    cited: set[str] = set()
    for m in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", src):
        cited.update(k.strip() for k in m.split(","))
    dupes = {k for k in keys if keys.count(k) > 1}
    orphans = [k for k in keys if k not in cited]
    missing = sorted(k for k in cited if k not in keys)

    print(f"  bibitems: {len(keys)}  |  distinct keys cited: {len(cited)}")
    if dupes:
        problems += len(dupes)
        print(f"  [FAIL] duplicate bibitem keys: {sorted(dupes)}")
    if missing:
        problems += len(missing)
        print(f"  [FAIL] cited without a bibitem: {missing}")
    if orphans:
        print(f"  [WARN] bibitems never cited: {orphans}")

    # Undefined references and labels.
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs: set[str] = set()
    for m in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", src):
        refs.add(m.strip())
    dangling = sorted(r for r in refs if r not in labels)
    if dangling:
        problems += len(dangling)
        print(f"  [FAIL] references with no matching label: {dangling}")

    print(f"  -> {'CLEAN' if problems == 0 else str(problems) + ' problem(s)'}")
    return problems


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Paper/main_round2.tex")
    sys.exit(1 if check(target) else 0)
