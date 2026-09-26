"""
E05 - Main GNN experiment, leakage-free.

Addresses round-1 comments 2, 8, 9, 10 and 11 in one consistent protocol.

  * Features: the 165 distributed Elliptic features only. The two withdrawn
    features (E01/E02) are gone; the topological set from E03 is available via
    --topo for the ablation but is not part of the headline model.
  * DGI regime: transductive and inductive are both run (comment 2).
  * DGI ablation: the no-DGI arm is trained from scratch with the fused slot
    zeroed, not ablated at inference (comment 9).
  * Threshold: selected on validation per run, then frozen (comment 8).
  * Seeds: 20 by default (comment 11 and Reviewer 2).
  * Naming: the model is a fusion of a frozen self-supervised representation
    with a supervised one, not "pre-training then fine-tuning" (comment 10).

Usage
-----
    python exp05_gnn_main.py --seeds 42 43        # smoke test
    python exp05_gnn_main.py                      # full 20-seed run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, SEEDS, DEVICE
from data import load_elliptic, base_feature_columns
from features import TOPO_FEATURES
from gnn import build_graph, train_dgi, train_classifier
from metrics import evaluate, select_threshold

MODELS = ["fusion", "gat_zero_dgi", "gat", "gcn", "sage"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--regimes", nargs="*", default=["transductive", "inductive"])
    ap.add_argument("--topo", action="store_true",
                    help="append the E03 topological features")
    ap.add_argument("--out", default="e05_gnn_main")
    args = ap.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    nodes, edges = load_elliptic()
    cols = base_feature_columns(nodes)
    if args.topo:
        topo = pd.read_parquet(RESULTS / "e03_topological_features.parquet")
        nodes = nodes.merge(topo, on="txId", how="left")
        cols = cols + TOPO_FEATURES

    data = build_graph(nodes, edges, cols)
    print(f"  graph: {data.num_nodes:,} nodes | {data.num_edges:,} directed edges "
          f"| {data.num_node_features} features")
    print(f"  labelled train/val/test: {int(data.train_mask.sum()):,} / "
          f"{int(data.val_mask.sum()):,} / {int(data.test_mask.sum()):,}")

    y = data.y.numpy()
    vam = data.val_mask.numpy()
    tem = data.test_mask.numpy()
    ts = data.time_step.numpy()

    rows, per_step_rows, dgi_info = [], [], []

    for regime in args.regimes:
        for seed in args.seeds:
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            emb, info = train_dgi(data, regime, device, seed)
            info["seed"] = seed
            dgi_info.append(info)
            print(f"\n  [{regime} seed {seed}] DGI: {info['epochs_run']} epochs, "
                  f"loss {info['best_loss']:.4f}, {info['seconds']:.1f}s")

            for kind in MODELS:
                # The plain GAT/GCN/SAGE arms do not involve DGI, so they are
                # only run once, under the first regime, to avoid duplicates.
                if kind in ("gat", "gcn", "sage") and regime != args.regimes[0]:
                    continue
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                r = train_classifier(data, kind, device, seed, dgi_emb=emb)
                s = r["scores"]

                thr = select_threshold(y[vam], s[vam], criterion="f1")
                m = evaluate(y[tem], s[tem], thr)
                m.update(model=kind,
                         # "n/a" round-trips through pandas as NaN and is then dropped by
                         # groupby, so a non-NA sentinel is used instead.
                         dgi_regime=regime if kind in ("fusion", "gat_zero_dgi")
                         else "none",
                         seed=seed, seconds=r["seconds"], n_params=r["n_params"],
                         peak_mem_mb=r["peak_mem_mb"], epochs_run=r["epochs_run"],
                         best_val_ap=r["best_val_ap"])
                rows.append(m)
                print(f"    {kind:14s} thr={thr:.4f} F1={m['f1_illicit']:.4f} "
                      f"P={m['precision']:.4f} R={m['recall']:.4f} "
                      f"PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f} "
                      f"({r['seconds']:.1f}s)")

                # Per-time-step breakdown at the frozen threshold.
                te_ts, te_y, te_s = ts[tem], y[tem], s[tem]
                for step in np.unique(te_ts):
                    sel = te_ts == step
                    ms = evaluate(te_y[sel], te_s[sel], thr)
                    ms.update(model=kind, dgi_regime=regime, seed=seed,
                              time_step=int(step), n_nodes=int(sel.sum()),
                              n_illicit=int((te_y[sel] == 1).sum()))
                    per_step_rows.append(ms)

                # Keep raw test scores from the primary seed for later analysis.
                if seed == args.seeds[0]:
                    np.save(RESULTS / f"{args.out}_scores_{kind}_{regime}.npy", s)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"{args.out}.csv", index=False)
    pd.DataFrame(per_step_rows).to_csv(RESULTS / f"{args.out}_per_timestep.csv",
                                       index=False)
    pd.DataFrame(dgi_info).to_csv(RESULTS / f"{args.out}_dgi_info.csv", index=False)

    agg = (res.groupby(["model", "dgi_regime"])
              .agg(n=("seed", "count"),
                   f1_mean=("f1_illicit", "mean"), f1_std=("f1_illicit", "std"),
                   precision=("precision", "mean"), recall=("recall", "mean"),
                   pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
                   mcc=("mcc", "mean"), fpr=("fpr", "mean"), fnr=("fnr", "mean"),
                   seconds=("seconds", "mean"), params=("n_params", "first"),
                   peak_mb=("peak_mem_mb", "mean"))
              .round(4).reset_index().sort_values("f1_mean", ascending=False))
    agg.to_csv(RESULTS / f"{args.out}_aggregate.csv", index=False)

    print("\n" + "=" * 120)
    print(agg.to_string(index=False))
    (RESULTS / f"{args.out}_aggregate.json").write_text(
        json.dumps(agg.to_dict(orient="records"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
