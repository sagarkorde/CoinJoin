"""
Central configuration for the MixTrace revision pipeline.

All experiment scripts import paths, split boundaries and seeds from here so
that a change to the protocol propagates everywhere instead of drifting
between scripts (a defect identified in the round-1 review).
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
RESEARCH    = REPO_ROOT / "research"
RAW_ELLIPTIC = REPO_ROOT / "Paper" / "Datasets" / "Elliptic_Dataset"
RAW_AUTHOR   = REPO_ROOT / "Paper" / "Datasets" / "Author_Dataset" / "Dataset.parquet"

CACHE       = RESEARCH / "cache"
RESULTS     = RESEARCH / "results"
REPORTS     = RESEARCH / "reports"
FIGURES     = RESEARCH / "figures"

for _d in (CACHE, RESULTS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ── Elliptic chronological split ──────────────────────────────────────────────
# Elliptic has 49 time steps. The split below is the standard chronological
# protocol used in the literature and is held fixed across every experiment.
TRAIN_END  = 34   # train  = steps  1..34  (inclusive)
VAL_END    = 41   # val    = steps 35..41  (inclusive)
TEST_START = 42   # test   = steps 42..49

# ── Elliptic label encoding ───────────────────────────────────────────────────
# Raw file uses "1" = illicit, "2" = licit, "unknown" = unlabelled.
LABEL_ILLICIT  = 1
LABEL_LICIT    = 0
LABEL_UNKNOWN  = -1

# ── Reproducibility ───────────────────────────────────────────────────────────
# Round-1 reviewers asked for more than five seeds; we use twenty.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
         52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
PRIMARY_SEED = 42

# ── Hardware budget (RTX 4060 Laptop, 8 GB VRAM; 50 GB RAM ceiling) ───────────
MAX_RAM_GB  = 50
DEVICE      = "cuda"
PYTHON      = r"C:/Program Files/Python310/python.exe"
