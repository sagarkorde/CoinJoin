"""
Assemble the Zenodo deposit for the CoinJoin paper.

Follows the layout of the authors' existing Zenodo record for the FinTech
paper (10.5281/zenodo.23100775): raw API responses as numbered gzipped JSONL
parts, derived tables as Parquet, and a README, manifest and SHA256SUMS
alongside. Code is not deposited; it stays in the GitHub repository, which the
record links, exactly as in the earlier deposit.

What goes in, and why each item is here rather than regenerable:

  Raw explorer responses   6,269 transactions retrieved to confirm CoinJoin
                           status. Regenerating them costs about 6,000
                           requests against free public explorers and was
                           rate-limited when first collected, so these are the
                           part of the study a reader genuinely cannot
                           reproduce cheaply.
  Verification verdicts    The per-transaction outcome of applying the
                           protocol rules to those responses.
  Script-type evidence     The columns behind the Taproot correction, reduced
                           to the fields the claim rests on.
  Experiment outputs       Every table in the paper, in machine-readable form.

What stays out: the author corpus, already published with its own DOI on IEEE
DataPort, and the Elliptic Data Set, which is distributed under its own terms.
Re-hosting either would split citations or breach a licence.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(r"C:\Users\sagar\Desktop\CoinJoin")
RESULTS = REPO / "research" / "results"
CACHE = RESULTS / "e14_verified"
OUT = REPO / "MDPI Submission" / "Revised MDPI Submit" / "ZENODO_BUNDLE"

PART_TARGET_MB = 40          # uncompressed JSONL per part, as in the earlier record
DATAPORT_DOI = "10.21227/bxmt-mn56"
GITHUB = "https://github.com/sagarkorde/CoinJoin"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_responses(dst: Path) -> list[dict]:
    """Write the explorer responses as numbered gzipped JSONL parts."""
    files = sorted(CACHE.glob("*.json"))
    if not files:
        print("  no cached responses found"); return []
    print(f"  packing {len(files):,} explorer responses ...")

    # First pass: group files into parts by uncompressed size.
    groups, cur, cur_bytes = [], [], 0
    limit = PART_TARGET_MB * 1024 * 1024
    for f in files:
        size = f.stat().st_size
        if cur and cur_bytes + size > limit:
            groups.append(cur); cur, cur_bytes = [], 0
        cur.append(f); cur_bytes += size
    if cur:
        groups.append(cur)

    n = len(groups)
    written = []
    for i, grp in enumerate(groups, 1):
        name = f"d1_raw_api_responses_part{i:02d}of{n:02d}.jsonl.gz"
        path = dst / name
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as gz:
            for f in grp:
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                gz.write(json.dumps({"txid": f.stem, "response": rec},
                                    separators=(",", ":")) + "\n")
        written.append({"file": name, "transactions": len(grp),
                        "bytes": path.stat().st_size})
        print(f"    {name}  {len(grp):>5} tx  {path.stat().st_size/1e6:6.1f} MB")
    return written


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "experiment_outputs").mkdir()

    parts = pack_responses(OUT)

    print("  verification verdicts ...")
    v = pd.read_csv(RESULTS / "e14b_all_verified.csv")
    v.to_parquet(OUT / "d2_verification_verdicts.parquet", index=False)
    confirmed = int(v.get("confirmed_coinjoin", pd.Series(dtype=bool)).sum())

    # Script-type flags for the screening candidates only. The corpus-wide
    # table runs to 5.88 M rows and regenerates in about twenty minutes from
    # the DataPort corpus with exp07, so depositing it would contradict the
    # principle stated in the README: deposit what cannot be cheaply
    # reproduced. The candidate subset is kept because the Taproot share among
    # candidates is a headline claim and the subset is small.
    print("  script-type evidence (screening candidates) ...")
    keep = ["txid", "out_has_p2tr", "in_has_p2tr", "out_has_p2wpkh",
            "out_has_p2sh", "out_has_p2pkh", "out_has_p2wsh",
            "out_script_uniform", "is_coinjoin_like"]
    st = RESULTS / "e07_script_types.parquet"
    if st.exists():
        df = pd.read_parquet(st, columns=keep)
        df = df[df["is_coinjoin_like"].astype(bool)].copy()
        df.to_parquet(OUT / "d3_script_types_candidates.parquet", index=False,
                      compression="zstd")
        mb = (OUT / "d3_script_types_candidates.parquet").stat().st_size / 1e6
        print(f"    {len(df):,} screening candidates, {mb:.1f} MB")

    print("  experiment outputs ...")
    n_out = 0
    for pat in ("*.csv", "*.json", "*.npy"):
        for f in RESULTS.glob(pat):
            shutil.copy(f, OUT / "experiment_outputs" / f.name); n_out += 1
    for f in (REPO / "research" / "figures").glob("*.png"):
        shutil.copy(f, OUT / "experiment_outputs" / f.name); n_out += 1
    print(f"    {n_out} files")

    manifest = {
        "title": ("Blockchain verification records for CoinJoin confirmation, "
                  "Bitcoin 2022-2024, with raw explorer responses"),
        "version": "1.0.0",
        "date": date.today().isoformat(),
        "license": "CC-BY-4.0",
        "authors": [
            {"name": "Korde, Sagar D.",
             "affiliation": "K. J. Somaiya School of Engineering, "
                            "Somaiya Vidyavihar University, Mumbai, India"},
            {"name": "Shekokar, Narendra M.",
             "affiliation": "Dwarkadas J. Sanghvi College of Engineering, "
                            "Mumbai, India"},
        ],
        "contents": {
            "raw_api_response_parts": len(parts),
            "transactions_retrieved": sum(p["transactions"] for p in parts),
            "confirmed_coinjoin_transactions": confirmed,
            "experiment_output_files": n_out,
        },
        "parts": parts,
        "related": {
            "source_corpus_doi": DATAPORT_DOI,
            "code_repository": GITHUB,
            "elliptic_dataset": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        },
        "personal_data": "none; public blockchain records only",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    readme = f"""# Blockchain verification records for CoinJoin confirmation

Bitcoin transactions, July 2022 to September 2024, with the raw block-explorer
responses used to confirm CoinJoin status.

**Version 1.0.0 · CC BY 4.0 · {date.today().isoformat()}**

Sagar D. Korde, K. J. Somaiya School of Engineering, Somaiya Vidyavihar
University, Mumbai, India
Narendra M. Shekokar, Dwarkadas J. Sanghvi College of Engineering, Mumbai,
India

---

## What this is

Supporting records for a study of CoinJoin screening and illicit-transaction
detection. The study needed to know which transactions in a 5,884,387-row
corpus are genuinely CoinJoin outputs, and the corpus could not answer that:
it records aggregate and mean output values but not individual output amounts,
and every published detection rule turns on the individual amounts.

Candidates were therefore selected by the conditions the corpus columns can
express, and each candidate was retrieved from a public block explorer and
tested against the full rule using the per-output values the blockchain
records. This deposit contains those retrieved responses and the verdicts
derived from them.

**{sum(p['transactions'] for p in parts):,} transactions were retrieved and
{confirmed:,} were confirmed as CoinJoin outputs.**

## Why it is deposited

Regenerating these records costs roughly six thousand requests against free
public explorers, and collecting them the first time triggered rate limiting.
That makes them the part of the study a reader cannot cheaply reproduce.
Everything else either regenerates in minutes from released code or is already
published elsewhere.

## Files

| File | Contents |
|---|---|
| `d1_raw_api_responses_partNNofMM.jsonl.gz` | Raw explorer responses, one JSON object per line, `{{txid, response}}` |
| `d2_verification_verdicts.parquet` | Per-transaction verdict: candidate group, protocol match, confirmation |
| `d3_script_types_candidates.parquet` | Script-type flags for the 110,352 screening candidates, reconstructed from node descriptors |
| `experiment_outputs/` | Machine-readable results behind every table and figure in the paper |
| `manifest.json` | Structured metadata |
| `SHA256SUMS` | Checksums for every file |

## Detection rules

CoinJoin status follows the conditions implemented in Dumplings
(github.com/nopara73/Dumplings), the reference tool used to extract Wasabi,
Whirlpool and JoinMarket transactions from mainnet in the measurement
literature. They are applied here, not devised here.

*Whirlpool*: native SegWit throughout; equal input and output counts between
five and ten; every output exactly equal; that value one of
0.001, 0.01, 0.05 or 0.5 BTC; at least one input exactly pool-sized; every
other input within 0.0011 BTC above the pool size.

*WabiSabi (Wasabi 2.x)*: inputs exclusively P2WPKH or Taproot; at least fifty
inputs; input and output values each sorted descending; more than eighty
percent of outputs drawn from the WabiSabi denomination set.

## What is not here, and where it is

The transaction corpus these records annotate is published separately with its
own DOI: <https://doi.org/{DATAPORT_DOI}>. It is not duplicated here.

The corpus-wide script-type table is not deposited either. It covers all
5,884,387 transactions and regenerates in about twenty minutes by running
`exp07_script_types.py` against the corpus above, so it fails the test applied
to everything else here. Its aggregate findings are in
`experiment_outputs/e07_script_type_summary.json`.

The Elliptic Data Set, used for the illicit-classification experiments, is
distributed under its own terms at
<https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>.

Analysis code, the experiment log and per-experiment commit hashes are at
<{GITHUB}>.

## Reproducing the verification

```python
import gzip, json, pandas as pd

records = []
for part in sorted(Path('.').glob('d1_raw_api_responses_part*.jsonl.gz')):
    with gzip.open(part, 'rt', encoding='utf-8') as fh:
        records += [json.loads(line) for line in fh]

verdicts = pd.read_parquet('d2_verification_verdicts.parquet')
print(len(records), 'responses', verdicts.confirmed_coinjoin.sum(), 'confirmed')
```

The classification function is `classify()` in
`research/experiments/exp14_external_ground_truth.py` in the code repository.
Running it over these responses reproduces `d2` without contacting any
explorer.

## Personal data

None. Bitcoin transactions and addresses are public ledger records. No
identity, attribution or off-chain information is included, and none was
collected.

## Citation

Cite the paper for the analysis, this record for the verification data, and
the DataPort record for the source corpus.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print("  checksums ...")
    lines = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "SHA256SUMS":
            lines.append(f"{sha256(f)}  {f.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n  bundle: {OUT}")
    print(f"  {len(lines)+1} files, {total/1e6:.1f} MB total")


if __name__ == "__main__":
    main()
