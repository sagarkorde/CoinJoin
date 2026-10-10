"""
Bring the reproducibility guide up to date.

Two gaps: the run list stopped at exp08, omitting the eight experiments added
since, and neither the Zenodo deposit nor the DataPort corpus was referenced.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(r"C:\Users\sagar\Desktop\CoinJoin")
GUIDE = REPO / "research" / "README.md"
S2 = (REPO / "MDPI Submission" / "Revised MDPI Submit" / "SUBMISSION"
      / "Supplementary" / "S2_Reproducibility_Guide.md")

VERSION_DOI = "10.5281/zenodo.23275812"
CONCEPT_DOI = "10.5281/zenodo.23275811"
DATAPORT_DOI = "10.21227/bxmt-mn56"

OLD_RUNS = """$PY research/experiments/exp06_coinjoin_label_recovery.py
$PY research/experiments/exp07_script_types.py
$PY research/experiments/exp08_coinjoin_detection.py     # needs E06 context
```"""

NEW_RUNS = """$PY research/experiments/exp06_coinjoin_label_recovery.py
$PY research/experiments/exp07_script_types.py
$PY research/experiments/exp08_coinjoin_detection.py     # needs E06 context
$PY research/experiments/exp08b_label_proxy_gradient.py
$PY research/experiments/exp09_explainability.py         # GPU
$PY research/experiments/exp10_statistical_analysis.py   # needs E05
$PY research/experiments/exp11_compute_cost.py           # idle GPU required
$PY research/experiments/exp12_address_columns.py
$PY research/experiments/exp13_training_diagnostics.py   # GPU
$PY research/experiments/exp14_external_ground_truth.py  # network; see below
$PY research/experiments/exp14b_verify_all.py            # network; see below
$PY research/experiments/exp15_confirmed_detection.py    # needs E14b
$PY research/experiments/exp16_detection_uncertainty.py  # needs E14b
```

`exp11_compute_cost.py` refuses to start if another process holds the GPU,
since contention invalidates its timings."""

SECTION = f"""

## Network experiments, and how to avoid repeating them

E14 and E14b retrieve per-output values from public block explorers. Running
them from scratch issues roughly six thousand requests; doing so the first
time triggered rate limiting, which is why `src/explorer.py` spreads requests
across endpoints, rests one that refuses, and caps the aggregate rate.

**Do not re-run them unless the retrieval itself is what you are checking.**
The responses are deposited:

- Verification records, version DOI <https://doi.org/{VERSION_DOI}>
- All versions, concept DOI <https://doi.org/{CONCEPT_DOI}>

Download `d1_raw_api_responses_part*.jsonl.gz`, expand them into
`research/results/e14_verified/` as one JSON file per transaction id, and both
experiments will read from cache and issue no requests. `classify()` in
`exp14_external_ground_truth.py` reproduces the verdicts from those responses
alone.

## Data sources and identifiers

| Source | Identifier | Notes |
|---|---|---|
| Author-curated corpus, 5,884,387 transactions | <https://doi.org/{DATAPORT_DOI}> | IEEE DataPort, Academic Use licence |
| Verification records, 6,269 responses | <https://doi.org/{VERSION_DOI}> | CC BY 4.0 |
| Elliptic Data Set | [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | Distributed under its own terms |
| Analysis code | <https://github.com/sagarkorde/CoinJoin> | MIT |

Neither the corpus nor the Elliptic data is redistributed in the deposit. The
corpus already carries its own DOI, and re-hosting it would split citations;
Elliptic is not ours to license.
"""


def main() -> None:
    s = GUIDE.read_text(encoding="utf-8")
    if OLD_RUNS in s:
        s = s.replace(OLD_RUNS, NEW_RUNS, 1)
        print("  run list extended to exp16")
    else:
        print("  run list anchor not found")
    if VERSION_DOI not in s:
        s = s.rstrip() + SECTION
        print("  data sources and network section added")
    GUIDE.write_text(s, encoding="utf-8")

    S2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(GUIDE, S2)
    print(f"  copied to {S2.name}")

    txt = GUIDE.read_text(encoding="utf-8")
    print(f"\n  exp16 referenced: {'exp16' in txt}")
    print(f"  version DOI present: {VERSION_DOI in txt}")
    print(f"  DataPort DOI present: {DATAPORT_DOI in txt}")


if __name__ == "__main__":
    main()
