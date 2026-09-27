"""
E12 - Address-count columns and the Mixing Index.

Reviewer 3 questioned a derived statistic reported with a mean of 8.79 and a
standard deviation of 689.27, asking how the implied outliers were handled.
Investigating that question uncovered a defect rather than a heavy tail.

The Mixing Index is defined as

    MI(t) = |A| * |B| / |A union B|

over the input and output address sets. Computed from the corpus columns that
name those quantities, `input_address_count`, `output_address_count` and
`total_addresses`, the statistic is degenerate: those columns are pinned at 1,
1 and 2 for almost every row, because they count elements of a one-element
array holding a semicolon-joined string rather than distinct addresses. MI is
then identically 0.5 and carries no information.

This experiment quantifies the defect, recomputes the address counts by
parsing the address strings, and reports the Mixing Index as it behaves when
computed correctly, together with the transform used to handle its tail.

This is the third column family in this corpus found to misreport its own
contents, after the six script-type booleans of E07 and the degenerate
`value_concentration_ratio` noted in E06.

Outputs
-------
  results/e12_address_columns.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR

BATCH = 400_000


def count_addrs(cell) -> int:
    """Distinct addresses in one cell of an address column."""
    if cell is None:
        return 0
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, list):
        cell = ";".join(str(x) for x in cell if x is not None)
    s = str(cell)
    if not s or s == "None":
        return 0
    return len({t for t in s.split(";") if t})


def main() -> None:
    pf = pq.ParquetFile(RAW_AUTHOR)
    cols = ["input_addresses", "output_addresses", "input_address_count",
            "output_address_count", "total_addresses", "input_count",
            "output_count", "is_coinjoin_like"]

    parts = []
    n = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        df = batch.to_pandas()
        n += len(df)
        a = df["input_addresses"].map(count_addrs).to_numpy()
        b = df["output_addresses"].map(count_addrs).to_numpy()
        parts.append(pd.DataFrame({
            "a_real": a, "b_real": b,
            "a_decl": df["input_address_count"].to_numpy(),
            "b_decl": df["output_address_count"].to_numpy(),
            "tot_decl": df["total_addresses"].to_numpy(),
            "in_ct": df["input_count"].to_numpy(),
            "out_ct": df["output_count"].to_numpy(),
            "y": df["is_coinjoin_like"].astype(int).to_numpy(),
        }))
        print(f"    parsed {n:,} rows", end="\r")
    d = pd.concat(parts, ignore_index=True)
    print(f"\n  parsed {len(d):,} transactions")

    out: dict = {"n_transactions": int(len(d))}

    # -- the defect --------------------------------------------------------
    out["declared_columns"] = {
        "input_address_count_unique_values": sorted(
            int(v) for v in pd.unique(d["a_decl"])[:10]),
        "output_address_count_max": int(d["b_decl"].max()),
        "total_addresses_modal_value": int(d["tot_decl"].mode().iloc[0]),
        "fraction_output_count_equal_1": round(float((d["b_decl"] == 1).mean()), 6),
        "interpretation": (
            "The address-count columns count elements of a one-element array "
            "holding a semicolon-joined address string, not distinct "
            "addresses, so they are pinned at 1, 1 and 2."),
    }
    out["parsed_columns"] = {
        "output_addresses_mean": round(float(d["b_real"].mean()), 4),
        "output_addresses_max": int(d["b_real"].max()),
        "input_addresses_mean": round(float(d["a_real"].mean()), 4),
        "input_addresses_max": int(d["a_real"].max()),
        "agreement_with_declared_pct": round(float(
            (d["b_real"] == d["b_decl"]).mean() * 100), 2),
    }

    # -- Mixing Index, declared versus parsed ------------------------------
    union_decl = d["tot_decl"].replace(0, np.nan).astype(float)
    mi_decl = (d["a_decl"].astype(float) * d["b_decl"].astype(float)) / union_decl

    union_real = (d["a_real"] + d["b_real"]).replace(0, np.nan).astype(float)
    mi_real = (d["a_real"].astype(float) * d["b_real"].astype(float)) / union_real

    def describe(x: pd.Series) -> dict:
        x = x.dropna()
        return {
            "mean": round(float(x.mean()), 4),
            "sd": round(float(x.std()), 4),
            "median": round(float(x.median()), 4),
            "p99": round(float(x.quantile(0.99)), 4),
            "p999": round(float(x.quantile(0.999)), 4),
            "max": round(float(x.max()), 4),
            "skew": round(float(x.skew()), 3),
        }

    out["mixing_index_declared"] = describe(mi_decl)
    out["mixing_index_parsed"] = describe(mi_real)

    # -- tail handling -----------------------------------------------------
    q = float(mi_real.quantile(0.999))
    wins = mi_real.clip(upper=q)
    logt = np.log1p(mi_real)
    out["tail_handling"] = {
        "winsorise_at_p999": {"threshold": round(q, 4), **describe(wins)},
        "log1p": describe(logt),
        "n_above_p999": int((mi_real > q).sum()),
    }

    m = mi_real.notna()
    out["label_correlation"] = {
        "mi_declared": round(float(np.corrcoef(
            mi_decl.fillna(0), d["y"])[0, 1]), 4),
        "mi_parsed": round(float(np.corrcoef(mi_real[m], d["y"][m])[0, 1]), 4),
        "mi_parsed_log1p": round(float(np.corrcoef(
            logt[m], d["y"][m])[0, 1]), 4),
    }

    (RESULTS / "e12_address_columns.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
