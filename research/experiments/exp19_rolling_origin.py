"""
E19 - Rolling-origin evaluation: can recalibration recover what drift takes?

Two questions meet in one experiment.

The first is a strictly prospective evaluation in which future graph nodes,
edges and neighbourhood statistics cannot influence earlier predictions, ideally
through an incremental graph-construction experiment that simulates realistic
monitoring. The second is whether threshold
recalibration remedies the collapse after time step 42, as the gentler fall
in ROC-AUC than in F1 suggests it might; frozen thresholds, recalibrated
decision rules and periodically retrained models are compared.

The design answers both. At each origin t the graph is rebuilt from the nodes
of time steps up to and including t, so nothing later exists in any form: not
as a node, not as an edge, not in the scaler, not in the self-supervised
pre-training. Standardisation uses training rows only, the DGI encoder is
fitted on the subgraph up to t-1, the classifier trains on labelled nodes up
to t-2 and selects on t-1, and step t is scored once.

  frozen         one model, trained once under the main protocol,
                 with the threshold selected on the validation block and held
                 fixed for every later step. This is the E05 setting.
  recalibrated   the same model and the same scores, with the threshold
                 reselected on the most recent labelled step before t. No
                 retraining, so the cost is an afternoon of labelling rather
                 than a training run.
  retrained      a model rebuilt from scratch at every origin on an
                 incrementally constructed graph, which is the strictly
                 prospective arm.

If recalibration recovers most of the loss, the operational recommendation is
relabelling and rethresholding. If only retraining recovers it, the
recommendation is a retraining schedule instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, SEEDS, DEVICE, TRAIN_END, VAL_END  # noqa: E402
from data import load_elliptic, base_feature_columns           # noqa: E402
from gnn import build_graph, train_dgi, train_classifier, set_seed  # noqa: E402
from models import DGI                                          # noqa: E402
from metrics import evaluate, select_threshold                  # noqa: E402

from torch_geometric.data import Data                           # noqa: E402

ORIGINS = list(range(VAL_END + 1, 50))   # 42 .. 49, the main test window


def build_graph_upto(nodes: pd.DataFrame, edges: pd.DataFrame,
                     cols: list[str], origin: int,
                     train_end: int, val_end: int) -> Data:
    """
    Build the graph as it would have existed at time step `origin`.

    Nodes after `origin` are absent entirely, so no later node, edge or
    neighbourhood statistic can reach an earlier prediction. Feature
    standardisation uses labelled training rows only, as in build_graph.
    """
    sub = nodes[nodes["time_step"] <= origin].copy()
    keep = set(sub["txId"].to_numpy().tolist())
    e = edges[edges["txId1"].isin(keep) & edges["txId2"].isin(keep)]

    idx = {int(t): i for i, t in enumerate(sub["txId"].to_numpy())}
    X = np.nan_to_num(sub[cols].to_numpy(dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = sub["label"].to_numpy()
    ts = sub["time_step"].to_numpy()
    has = y != -1

    train_mask = (ts <= train_end) & has
    val_mask = (ts > train_end) & (ts <= val_end) & has
    test_mask = (ts == origin) & has

    mu, sd = X[train_mask].mean(axis=0), X[train_mask].std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    src = e["txId1"].map(idx).to_numpy()
    dst = e["txId2"].map(idx).to_numpy()
    ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])

    return Data(x=torch.tensor(X, dtype=torch.float),
                y=torch.tensor(np.where(y == -1, 0, y), dtype=torch.long),
                edge_index=torch.tensor(ei, dtype=torch.long),
                train_mask=torch.tensor(train_mask),
                val_mask=torch.tensor(val_mask),
                test_mask=torch.tensor(test_mask),
                time_step=torch.tensor(ts, dtype=torch.long))


def dgi_upto(data: Data, cutoff: int, device: str, seed: int,
             hid_dim: int = 128, epochs: int = 200, lr: float = 1e-3,
             patience: int = 30) -> torch.Tensor:
    """Inductive DGI fitted on the subgraph up to `cutoff`, then applied to all."""
    set_seed(seed)
    keep = (data.time_step <= cutoff)
    sub_idx = keep.nonzero(as_tuple=True)[0]
    remap = -torch.ones(data.num_nodes, dtype=torch.long)
    remap[sub_idx] = torch.arange(sub_idx.numel())
    ei = data.edge_index
    emask = keep[ei[0]] & keep[ei[1]]
    x_fit, ei_fit = data.x[sub_idx].to(device), remap[ei[:, emask]].to(device)

    model = DGI(data.num_node_features, hid_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad, best_state = float("inf"), 0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = model.loss(x_fit, ei_fit)
        loss.backward()
        opt.step()
        lv = float(loss.item())
        if lv < best - 1e-5:
            best, bad = lv, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model.encoder(data.x.to(device),
                             data.edge_index.to(device)).detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS[:5])
    ap.add_argument("--origins", type=int, nargs="*", default=ORIGINS)
    ap.add_argument("--arms", nargs="*",
                    default=["frozen", "recalibrated", "retrained"])
    ap.add_argument("--out", default="e19_rolling_origin")
    args = ap.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print("  device: %s | seeds %s | origins %s"
          % (device, args.seeds, args.origins))

    nodes, edges = load_elliptic()
    cols = base_feature_columns(nodes)
    rows = []

    # ---- frozen and recalibrated: one model per seed, two threshold rules ----
    if {"frozen", "recalibrated"} & set(args.arms):
        full = build_graph(nodes, edges, cols)
        y, ts = full.y.numpy(), full.time_step.numpy()
        vam = full.val_mask.numpy()
        lab = nodes["label"].to_numpy() != -1

        for seed in args.seeds:
            emb, info = train_dgi(full, "inductive", device, seed)
            r = train_classifier(full, "fusion", device, seed, dgi_emb=emb)
            s = r["scores"]
            thr_frozen = select_threshold(y[vam], s[vam], criterion="f1")
            print("\n  [seed %d] base model trained, frozen threshold %.4f"
                  % (seed, thr_frozen))

            for t in args.origins:
                sel = (ts == t) & lab
                if sel.sum() == 0 or len(np.unique(y[sel])) < 2:
                    continue
                if "frozen" in args.arms:
                    m = evaluate(y[sel], s[sel], thr_frozen)
                    m.update(arm="frozen", seed=seed, time_step=t,
                             threshold=float(thr_frozen),
                             n_nodes=int(sel.sum()),
                             n_illicit=int((y[sel] == 1).sum()))
                    rows.append(m)
                if "recalibrated" in args.arms:
                    prev = (ts == t - 1) & lab
                    if prev.sum() == 0 or len(np.unique(y[prev])) < 2:
                        thr = thr_frozen
                    else:
                        thr = select_threshold(y[prev], s[prev], criterion="f1")
                    m = evaluate(y[sel], s[sel], thr)
                    m.update(arm="recalibrated", seed=seed, time_step=t,
                             threshold=float(thr), n_nodes=int(sel.sum()),
                             n_illicit=int((y[sel] == 1).sum()))
                    rows.append(m)
            print("    frozen and recalibrated scored for %d origins"
                  % len(args.origins))

    # ---- retrained: a fresh model on an incrementally built graph ----
    if "retrained" in args.arms:
        for seed in args.seeds:
            for t in args.origins:
                t0 = time.time()
                d = build_graph_upto(nodes, edges, cols, origin=t,
                                     train_end=t - 2, val_end=t - 1)
                if int(d.test_mask.sum()) == 0:
                    continue
                yy = d.y.numpy()
                if len(np.unique(yy[d.test_mask.numpy()])) < 2:
                    continue
                emb = dgi_upto(d, cutoff=t - 1, device=device, seed=seed)
                r = train_classifier(d, "fusion", device, seed, dgi_emb=emb)
                s = r["scores"]
                vm, tm = d.val_mask.numpy(), d.test_mask.numpy()
                thr = (select_threshold(yy[vm], s[vm], criterion="f1")
                       if len(np.unique(yy[vm])) > 1 else 0.5)
                m = evaluate(yy[tm], s[tm], thr)
                m.update(arm="retrained", seed=seed, time_step=t,
                         threshold=float(thr), n_nodes=int(tm.sum()),
                         n_illicit=int((yy[tm] == 1).sum()),
                         n_train=int(d.train_mask.sum()),
                         seconds=time.time() - t0)
                rows.append(m)
                print("    [seed %d] origin %d: nodes %d, train %d, "
                      "thr %.4f, F1 %.4f, MCC %.4f (%.0fs)"
                      % (seed, t, d.num_nodes, int(d.train_mask.sum()),
                         thr, m["f1_illicit"], m["mcc"], time.time() - t0))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / ("%s.csv" % args.out), index=False)

    per_step = (df.groupby(["arm", "time_step"])
                  .agg(f1=("f1_illicit", "mean"), f1_sd=("f1_illicit", "std"),
                       precision=("precision", "mean"),
                       recall=("recall", "mean"), mcc=("mcc", "mean"),
                       roc_auc=("roc_auc", "mean"))
                  .round(4).reset_index())
    per_step.to_csv(RESULTS / ("%s_per_step.csv" % args.out), index=False)

    overall = (df.groupby("arm")
                 .agg(f1=("f1_illicit", "mean"), mcc=("mcc", "mean"),
                      recall=("recall", "mean"), roc_auc=("roc_auc", "mean"))
                 .round(4).reset_index())
    overall.to_csv(RESULTS / ("%s_overall.csv" % args.out), index=False)

    print("\nPer time step:\n" + per_step.to_string(index=False))
    print("\nOverall across origins:\n" + overall.to_string(index=False))
    (RESULTS / ("%s_summary.json" % args.out)).write_text(
        json.dumps({"seeds": args.seeds, "origins": args.origins,
                    "arms": args.arms,
                    "per_step": per_step.to_dict("records"),
                    "overall": overall.to_dict("records")}, indent=1),
        encoding="utf-8")
    print("\n  written to %s_per_step.csv" % args.out)


if __name__ == "__main__":
    main()
