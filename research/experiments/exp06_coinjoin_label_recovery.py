"""
E06 - Recovering the decision rule behind `is_coinjoin_like`.

Reviewers 1 (comment 3), 2 and 3 (comment 2) all ask the same thing: state the
exact rule that produced the author-curated CoinJoin label, and say whether the
variables the detector clusters on are the same variables that produced the
label. If they are, the round-1 evaluation measures agreement between two
heuristics rather than detection accuracy.

The rule was not documented, so this experiment recovers it from the data. A
decision tree is fitted to predict the label from the structural columns; if
the label is a deterministic function of a few of them, a shallow tree will
reproduce it exactly and the recovered rule can be read off and stated in the
manuscript.

Outputs
-------
  results/e06_label_rule.json
  results/e06_label_rule.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, RAW_AUTHOR

# Structural / behavioural columns that could plausibly enter a CoinJoin
# heuristic. Identifier, timestamp and raw address-list columns are excluded.
CANDIDATE_COLS = [
    "input_count", "output_count", "input_output_ratio",
    "input_address_count", "output_address_count", "total_addresses",
    "input_script_count", "output_script_count", "address_reuse",
    "total_input_value", "total_output_value", "fee", "value_difference",
    "avg_input_value", "avg_output_value", "value_concentration_ratio",
    "size", "vsize", "weight", "fee_rate_sat_per_vbyte",
    "is_self_transfer", "is_consolidation", "is_distribution",
    "is_peer_to_peer", "is_batch_payment",
    "has_p2pk", "has_p2pkh", "has_p2sh", "has_p2wpkh", "has_p2wsh",
    "has_taproot", "has_op_return", "rbf_enabled", "has_coinbase",
]


def main() -> None:
    print("  loading author-curated dataset ...")
    df = pd.read_parquet(RAW_AUTHOR, columns=CANDIDATE_COLS + ["is_coinjoin_like"])
    print(f"  {len(df):,} transactions")

    y = df["is_coinjoin_like"].astype(int).to_numpy()
    print(f"  positive rate: {y.mean() * 100:.4f}%  ({y.sum():,} positives)")

    X = df[CANDIDATE_COLS].astype(np.float32)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out: dict = {
        "n_transactions": int(len(df)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
    }

    # Fit increasingly deep trees until the label is reproduced exactly.
    for depth in (1, 2, 3, 4, 5):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
        tree.fit(X, y)
        pred = tree.predict(X)
        acc = float((pred == y).mean())
        print(f"    depth {depth}: training accuracy {acc:.8f}")
        if acc == 1.0:
            rule = export_text(tree, feature_names=CANDIDATE_COLS)
            out["recovered_exactly"] = True
            out["tree_depth"] = depth
            out["rule_text"] = rule
            imp = sorted(zip(CANDIDATE_COLS, tree.feature_importances_),
                         key=lambda kv: -kv[1])
            out["features_used"] = [f for f, v in imp if v > 0]
            print("\n  EXACT RULE RECOVERED at depth", depth)
            print(rule)
            break
    else:
        out["recovered_exactly"] = False
        tree = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
        out["tree_depth"] = 6
        out["rule_text"] = export_text(tree, feature_names=CANDIDATE_COLS)
        out["accuracy_depth6"] = float((tree.predict(X) == y).mean())

    # Direct test of the textbook CoinJoin heuristic.
    cand = {
        "input_count>=2 AND output_count>=3": (df["input_count"] >= 2) & (df["output_count"] >= 3),
        "input_count>=3 AND output_count>=3": (df["input_count"] >= 3) & (df["output_count"] >= 3),
        "input_count>=2 AND output_count>=2": (df["input_count"] >= 2) & (df["output_count"] >= 2),
        "input_count>=5 AND output_count>=5": (df["input_count"] >= 5) & (df["output_count"] >= 5),
    }
    checks = {}
    for name, m in cand.items():
        m = m.to_numpy()
        tp = int((m & (y == 1)).sum()); fp = int((m & (y == 0)).sum())
        fn = int((~m & (y == 1)).sum())
        checks[name] = {
            "exact_match": bool((m.astype(int) == y).all()),
            "precision": round(tp / max(tp + fp, 1), 6),
            "recall": round(tp / max(tp + fn, 1), 6),
        }
    out["textbook_heuristic_checks"] = checks
    print("\n  textbook heuristic checks:")
    print(json.dumps(checks, indent=2))

    (RESULTS / "e06_label_rule.json").write_text(json.dumps(out, indent=2),
                                                 encoding="utf-8")
    (RESULTS / "e06_label_rule.txt").write_text(out["rule_text"], encoding="utf-8")
    print("\n[OK] wrote results/e06_label_rule.json")


if __name__ == "__main__":
    main()
