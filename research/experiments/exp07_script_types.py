"""
E07 - Reconstructing script types, and the Taproot claim.

Round-1 reported that the author-curated corpus contains no confirmed P2TR
transactions and therefore used multi-input P2WPKH as a Taproot proxy. All
three reviewers asked for the Taproot claims to be moderated on that basis
(R1 comment 5, R2, R3 comment 3).

The premise turns out to be an artefact. The dataset's five script-type
booleans (`has_p2pk`, `has_p2pkh`, `has_p2sh`, `has_p2wpkh`, `has_p2wsh`,
`has_taproot`) are identically False for all 5,884,387 rows, so nothing can be
"confirmed" through them. The authoritative information sits in the
`input_script_types` / `output_script_types` columns, which carry the script
descriptors as reported by the node.

This experiment rebuilds the flags from those columns and quantifies actual
Taproot presence, so the manuscript can state the corpus composition correctly
instead of moderating a claim that rested on a broken column.

Outputs
-------
  results/e07_script_types.parquet    (per-transaction reconstructed flags)
  results/e07_script_type_summary.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR

# Bitcoin Core script descriptors -> conventional names.
SCRIPT_MAP = {
    "pubkey": "p2pk",
    "pubkeyhash": "p2pkh",
    "scripthash": "p2sh",
    "witness_v0_keyhash": "p2wpkh",
    "witness_v0_scripthash": "p2wsh",
    "witness_v1_taproot": "p2tr",
    "nulldata": "op_return",
    "multisig": "bare_multisig",
    "nonstandard": "nonstandard",
    "witness_unknown": "witness_unknown",
}
KINDS = ["p2pk", "p2pkh", "p2sh", "p2wpkh", "p2wsh", "p2tr",
         "op_return", "bare_multisig", "nonstandard", "witness_unknown"]

BATCH = 400_000


def _tokens(cell) -> list[str]:
    """Normalise one cell of the script-type column into a token list."""
    if cell is None:
        return []
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, list):
        cell = ";".join(str(c) for c in cell if c is not None)
    s = str(cell)
    if not s or s == "None":
        return []
    return [t for t in s.split(";") if t]


def main() -> None:
    pf = pq.ParquetFile(RAW_AUTHOR)
    cols = ["txid", "input_script_types", "output_script_types",
            "input_count", "output_count", "is_coinjoin_like", "timestamp"]

    frames = []
    out_counter: Counter = Counter()
    in_counter: Counter = Counter()
    n_rows = 0

    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        df = batch.to_pandas()
        n_rows += len(df)

        rec = {"txid": df["txid"].to_numpy()}
        out_tok = df["output_script_types"].map(_tokens)
        in_tok = df["input_script_types"].map(_tokens)

        for toks, counter in ((out_tok, out_counter), (in_tok, in_counter)):
            for lst in toks:
                counter.update(SCRIPT_MAP.get(t, "other") for t in lst)

        for kind in KINDS:
            rec[f"out_has_{kind}"] = out_tok.map(
                lambda ts, k=kind: any(SCRIPT_MAP.get(t) == k for t in ts)).to_numpy()
            rec[f"in_has_{kind}"] = in_tok.map(
                lambda ts, k=kind: any(SCRIPT_MAP.get(t) == k for t in ts)).to_numpy()

        # Output-side script uniformity: a real CoinJoin mixes uniform outputs.
        rec["out_script_uniform"] = out_tok.map(
            lambda ts: len({SCRIPT_MAP.get(t, "other") for t in ts
                            if SCRIPT_MAP.get(t) != "op_return"}) == 1).to_numpy()
        rec["out_n_script_kinds"] = out_tok.map(
            lambda ts: len({SCRIPT_MAP.get(t, "other") for t in ts})).to_numpy()
        rec["is_coinjoin_like"] = df["is_coinjoin_like"].to_numpy()
        rec["input_count"] = df["input_count"].to_numpy()
        rec["output_count"] = df["output_count"].to_numpy()
        rec["timestamp"] = df["timestamp"].to_numpy()
        frames.append(pd.DataFrame(rec))
        print(f"    processed {n_rows:,} rows", end="\r")

    res = pd.concat(frames, ignore_index=True)
    print(f"\n  reconstructed flags for {len(res):,} transactions")

    res.to_parquet(RESULTS / "e07_script_types.parquet", index=False)

    tot_out = sum(out_counter.values())
    tot_in = sum(in_counter.values())
    summary = {
        "n_transactions": int(len(res)),
        "time_range": [str(res["timestamp"].min()), str(res["timestamp"].max())],
        "declared_flags_all_false": True,
        "output_script_occurrences": {k: int(v) for k, v in out_counter.most_common()},
        "input_script_occurrences": {k: int(v) for k, v in in_counter.most_common()},
        "output_script_share_pct": {k: round(v / tot_out * 100, 4)
                                    for k, v in out_counter.most_common()},
        "input_script_share_pct": {k: round(v / tot_in * 100, 4)
                                   for k, v in in_counter.most_common()},
        "tx_with_any_taproot_output": int(res["out_has_p2tr"].sum()),
        "tx_with_any_taproot_input": int(res["in_has_p2tr"].sum()),
        "tx_taproot_either_side": int((res["out_has_p2tr"] | res["in_has_p2tr"]).sum()),
    }
    summary["tx_taproot_either_side_pct"] = round(
        summary["tx_taproot_either_side"] / len(res) * 100, 4)

    # Taproot presence among the label-positive ("CoinJoin-like") subset.
    cj = res[res["is_coinjoin_like"].astype(bool)]
    summary["coinjoin_like"] = {
        "n": int(len(cj)),
        "with_taproot_output": int(cj["out_has_p2tr"].sum()),
        "with_taproot_input": int(cj["in_has_p2tr"].sum()),
        "taproot_either_side": int((cj["out_has_p2tr"] | cj["in_has_p2tr"]).sum()),
        "all_taproot_outputs_and_uniform": int(
            (cj["out_has_p2tr"] & cj["out_script_uniform"]).sum()),
    }
    summary["coinjoin_like"]["taproot_either_side_pct"] = round(
        summary["coinjoin_like"]["taproot_either_side"] / max(len(cj), 1) * 100, 4)

    (RESULTS / "e07_script_type_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n  output script share (%):")
    for k, v in summary["output_script_share_pct"].items():
        print(f"    {k:18s} {v:8.4f}")
    print("\n  transactions touching Taproot: "
          f"{summary['tx_taproot_either_side']:,} "
          f"({summary['tx_taproot_either_side_pct']}%)")
    print("  among label-positive CoinJoin-like: "
          f"{summary['coinjoin_like']['taproot_either_side']:,} / "
          f"{summary['coinjoin_like']['n']:,} "
          f"({summary['coinjoin_like']['taproot_either_side_pct']}%)")


if __name__ == "__main__":
    main()
