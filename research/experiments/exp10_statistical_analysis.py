"""
E10 - Uncertainty quantification for the main GNN comparison.

Reviewer 1 (comment 11) observed that repeated seeds measure training
stochasticity on one fixed partition and do not constitute independent samples
of generalisation performance, and suggested a bootstrap over test-set metric
differences instead. Reviewer 3 (comment 4) made the same point about the
paired t-tests resting on df = 4.

This script reports the two sources of variation as distinct quantities.

  * **Seed variance** — spread across training runs on a fixed split. Reported
    as mean +/- sd and a paired t-test, described only as evidence of
    seed-level consistency.
  * **Test-set variance** — paired bootstrap resampling of test nodes, with a
    percentile confidence interval on the metric difference and a bootstrap
    p-value. This is the evidence for a claim about the population.

Outputs
-------
  results/e10_bootstrap.csv
  results/e10_seed_tests.csv
  results/e10_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS
from data import load_elliptic, labelled_masks
from metrics import (bootstrap_metric_ci, evaluate, paired_bootstrap_difference,
                     select_threshold)

N_BOOT = 2000
REFERENCE = "fusion"


def main() -> None:
    per_run = pd.read_csv(RESULTS / "e05_gnn_main.csv")

    # ----------------------------------------------------------- seed variance
    seed_rows = []
    ref = per_run[per_run["model"] == REFERENCE]
    # Prefer the inductive fusion arm as the reference system.
    if (ref["dgi_regime"] == "inductive").any():
        ref = ref[ref["dgi_regime"] == "inductive"]
    ref_by_seed = ref.set_index("seed")["f1_illicit"]

    for (model, regime), grp in per_run.groupby(["model", "dgi_regime"]):
        if model == REFERENCE and regime == ref["dgi_regime"].iloc[0]:
            continue
        other = grp.set_index("seed")["f1_illicit"]
        common = ref_by_seed.index.intersection(other.index)
        if len(common) < 3:
            continue
        a, b = ref_by_seed.loc[common], other.loc[common]
        t, p = stats.ttest_rel(a, b)
        seed_rows.append({
            "comparison": f"{REFERENCE} vs {model} ({regime})",
            "n_seeds": int(len(common)),
            "mean_reference": round(float(a.mean()), 4),
            "mean_other": round(float(b.mean()), 4),
            "mean_difference": round(float((a - b).mean()), 4),
            "sd_reference": round(float(a.std()), 4),
            "sd_other": round(float(b.std()), 4),
            "paired_t": round(float(t), 4),
            "p_value": round(float(p), 6),
        })
    seeds_df = pd.DataFrame(seed_rows)
    seeds_df.to_csv(RESULTS / "e10_seed_tests.csv", index=False)
    print("SEED-LEVEL CONSISTENCY (fixed split; not population evidence)")
    print(seeds_df.to_string(index=False))

    # ------------------------------------------------------- test-set variance
    nodes, _ = load_elliptic()
    lm = labelled_masks(nodes)
    y = nodes["label"].to_numpy()
    tem = lm["test"]
    vam = lm["val"]

    score_files = sorted(RESULTS.glob("e05_gnn_main_scores_*.npy"))
    scores, thresholds = {}, {}
    for f in score_files:
        name = f.stem.replace("e05_gnn_main_scores_", "")
        s = np.load(f)
        if len(s) != len(y):
            continue
        scores[name] = s
        thresholds[name] = select_threshold(y[vam], s[vam], criterion="f1")

    if not scores:
        print("\n  no score files found; run exp05_gnn_main.py first")
        return

    print(f"\n  bootstrapping {len(scores)} systems over the test split "
          f"({int(tem.sum()):,} nodes, {N_BOOT} resamples) ...")

    boot_rows = []
    for name, s in scores.items():
        for metric in ("f1_illicit", "pr_auc", "mcc"):
            ci = bootstrap_metric_ci(y[tem], s[tem], thresholds[name],
                                     metric=metric, n_boot=N_BOOT, seed=0)
            ci.update(system=name)
            boot_rows.append(ci)

    # Paired differences against the reference system.
    ref_key = next((k for k in scores if k.startswith(f"{REFERENCE}_inductive")),
                   next((k for k in scores if k.startswith(REFERENCE)), None))
    pair_rows = []
    if ref_key:
        for name, s in scores.items():
            if name == ref_key:
                continue
            for metric in ("f1_illicit", "pr_auc", "mcc"):
                d = paired_bootstrap_difference(
                    y[tem], scores[ref_key][tem], s[tem],
                    thresholds[ref_key], thresholds[name],
                    metric=metric, n_boot=N_BOOT, seed=0)
                d.update(reference=ref_key, other=name)
                pair_rows.append(d)

    boot_df = pd.DataFrame(boot_rows)
    pair_df = pd.DataFrame(pair_rows)
    boot_df.to_csv(RESULTS / "e10_bootstrap.csv", index=False)
    pair_df.to_csv(RESULTS / "e10_paired_bootstrap.csv", index=False)

    print("\nBOOTSTRAP CONFIDENCE INTERVALS (test-set resampling)")
    show = boot_df[boot_df["metric"] == "f1_illicit"].copy()
    show[["point", "ci_lo", "ci_hi"]] = show[["point", "ci_lo", "ci_hi"]].round(4)
    print(show[["system", "metric", "point", "ci_lo", "ci_hi"]].to_string(index=False))

    if not pair_df.empty:
        print(f"\nPAIRED BOOTSTRAP DIFFERENCES vs {ref_key}")
        sp = pair_df[pair_df["metric"] == "f1_illicit"].copy()
        sp[["difference", "ci_lo", "ci_hi"]] = sp[["difference", "ci_lo", "ci_hi"]].round(4)
        print(sp[["other", "difference", "ci_lo", "ci_hi",
                  "p_bootstrap"]].to_string(index=False))

    (RESULTS / "e10_summary.json").write_text(json.dumps({
        "reference_system": ref_key,
        "n_test_nodes": int(tem.sum()),
        "n_bootstrap": N_BOOT,
        "seed_tests": seed_rows,
        "bootstrap_ci": boot_rows,
        "paired_bootstrap": pair_rows,
    }, indent=2), encoding="utf-8")
    print("\n[OK] wrote results/e10_*")


if __name__ == "__main__":
    main()
