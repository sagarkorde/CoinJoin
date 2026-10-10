"""Write a paste-ready metadata sheet into the Zenodo bundle."""
from __future__ import annotations

import json
from pathlib import Path

BUNDLE = Path(r"C:\Users\sagar\Desktop\CoinJoin\MDPI Submission\Revised MDPI Submit"
              r"\ZENODO_BUNDLE")
DATAPORT = "10.21227/bxmt-mn56"
GITHUB = "https://github.com/sagarkorde/CoinJoin"
FINTECH = "10.5281/zenodo.23100775"


def main() -> None:
    man = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    n_tx = man["contents"]["transactions_retrieved"]
    n_cj = man["contents"]["confirmed_coinjoin_transactions"]

    description = f"""Supporting records for a study of CoinJoin screening and illicit-transaction detection in Bitcoin.

The study needed to establish which transactions in a 5,884,387-row corpus are genuinely CoinJoin outputs. The corpus could not answer that question: it records aggregate and mean output values but not individual output amounts, and every published CoinJoin detection rule turns on the individual amounts. Candidates were therefore selected by the conditions the corpus columns can express, and each candidate was retrieved from a public block explorer and tested against the full rule using the per-output values recorded on the blockchain.

This deposit contains those retrieved responses and the verdicts derived from them. {n_tx:,} transactions were retrieved and {n_cj:,} were confirmed as CoinJoin outputs, comprising Whirlpool and WabiSabi rounds.

The records are deposited because regenerating them costs roughly six thousand requests against free public block explorers, and collecting them the first time triggered rate limiting. That makes them the part of the study a reader cannot cheaply reproduce. Everything else either regenerates in minutes from the released code or is published elsewhere.

CoinJoin status follows the conditions implemented in Dumplings, the reference tool used to extract Wasabi, Whirlpool and JoinMarket transactions from Bitcoin mainnet in the measurement literature. Those rules are applied here, not devised here.

Contents: raw block-explorer responses as gzipped JSONL parts; per-transaction verification verdicts; reconstructed script-type flags for the screening candidates; and the machine-readable outputs behind every table and figure in the article. A README, a structured manifest and SHA256 checksums accompany the data.

Not included: the transaction corpus these records annotate, which is published separately under DOI {DATAPORT}; and the Elliptic Data Set, which is distributed under its own terms. Analysis code is in the linked GitHub repository.

All transaction data are public ledger records. No personal, identifying or off-chain information was collected or is distributed."""

    sheet = f"""# Zenodo upload metadata

Paste-ready values for this deposit. Produced from `manifest.json`.

**Bundle:** {len(files)} files, {total/1e6:.1f} MB

---

## Upload type

`Dataset`

## Title

```
{man['title']}
```

## Authors

| Name | Affiliation | ORCID |
|---|---|---|
| Korde, Sagar D. | K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai, India | 0000-0002-6289-6741 |
| Shekokar, Narendra M. | Dwarkadas J. Sanghvi College of Engineering, Mumbai, India | 0000-0002-2507-4140 |

## Description

```
{description}
```

## License

`Creative Commons Attribution 4.0 International` (CC BY 4.0)

Matches the licence on the earlier FinTech deposit.

## Version

`1.0.0`

This is a **new record**, not a new version of {FINTECH}. That record holds a
held-out temporal transaction sample for a different article; this one holds
CoinJoin verification records. Starting a new version of the earlier record
would merge two unrelated datasets under one concept DOI.

## Keywords

```
Bitcoin
CoinJoin
blockchain forensics
Whirlpool
WabiSabi
transaction graph
reproducibility
ground truth
```

## Related identifiers

| Relation | Identifier | Resource type |
|---|---|---|
| `is supplement to` | *the article DOI, once assigned* | Publication / Article |
| `is derived from` | {DATAPORT} | Dataset |
| `is supplemented by` | {GITHUB} | Software |
| `is part of same study as` | {FINTECH} | Dataset (optional) |

Leave the article relation until the paper has a DOI, then edit the record and
add it. Zenodo allows metadata edits after publication without minting a new
version.

## Notes field

```
Verification rules follow Dumplings (https://github.com/nopara73/Dumplings).
Responses retrieved from mempool.space and blockstream.info between 27 September
and 10 October 2026. Checksums for every file are in SHA256SUMS.
```

## Files to upload

Upload the entire contents of `ZENODO_BUNDLE/`, preserving the
`experiment_outputs/` subfolder. Zenodo flattens directories on upload through
the browser, so either upload `experiment_outputs` as a single zip or accept
the flattened names; the manifest records the intended structure either way.
"""
    (BUNDLE / "ZENODO_METADATA.md").write_text(sheet, encoding="utf-8")
    print(f"wrote {BUNDLE / 'ZENODO_METADATA.md'}")
    print(f"bundle: {len(files)} files, {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
