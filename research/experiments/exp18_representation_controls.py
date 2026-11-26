"""
E18 - Does the frozen representation contribute structure, or capacity?

E05 shows that the frozen DGI representation improves the GAT arm, but that
does not establish that self-supervised learning is the better modelling
strategy: the fused model
also has more parameters than the arm it is compared against, and no control
separates learned structural information from added capacity or the
regularising effect of a fixed extra input. This experiment adds a parameter-matched
comparison and a randomly initialised frozen representation.

Five arms, all under the inductive regime, which is the reference arm of
the study. Every arm uses the same GAT trunk, the same optimiser, the same
early-stopping rule on validation PR-AUC and the same frozen threshold.

  fusion            frozen DGI representation, trained self-supervised
  fusion_random     frozen representation from an encoder that is initialised
                    and never trained. An untrained two-layer GCN is still a
                    random projection of the node's own neighbourhood, so this
                    arm carries structure without any learning.
  fusion_noise      a frozen Gaussian matrix of the same shape, drawn without
                    reference to the graph. No structure of any kind.
  fusion_shuffled   the real DGI representation with its node rows permuted.
                    Marginal distribution preserved exactly, correspondence
                    between a node and its embedding destroyed.
  gat_zero_dgi      no frozen representation, as reported in E05
  gat_matched       no frozen representation, classifier head widened so the
                    trainable parameter count matches `fusion`

Each control isolates one explanation, and together they decompose the gain:

  fusion vs fusion_random    does self-supervised training add anything beyond
                             a random graph convolution?
  fusion_random vs _noise    does neighbourhood aggregation add anything beyond
                             an arbitrary fixed input?
  fusion vs fusion_shuffled  must the embedding belong to the node it is
                             attached to?
  fusion vs gat_matched      is the gain simply extra capacity?

A claim that self-supervised pre-training contributes learned structure
requires fusion to beat fusion_random. A claim that the frozen branch helps at
all requires it to beat gat_matched.
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

from config import RESULTS, SEEDS, DEVICE          # noqa: E402
from data import load_elliptic, base_feature_columns  # noqa: E402
from gnn import build_graph, train_dgi, set_seed   # noqa: E402
from models import DGIEncoder, FrozenDGIGATFusion  # noqa: E402
from metrics import evaluate, select_threshold     # noqa: E402

import torch.nn.functional as F                    # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

ARMS = ["fusion", "fusion_random", "fusion_noise", "fusion_shuffled",
        "gat_zero_dgi", "gat_matched"]

# Chosen so that the widened head matches `fusion`'s trainable parameter
# count: fusion's head is Linear(64+128, 64) + Linear(64, 2) = 12,482
# parameters, and Linear(64, h) + Linear(h, 2) = 67h + 2, so h = 186 lands
# within eighteen parameters of it. The run prints both counts to confirm.
MATCHED_HEAD_DIM = 186


def random_embedding(data, device: str, seed: int, dim: int) -> torch.Tensor:
    """A frozen representation from an encoder that is never trained."""
    set_seed(seed + 10_000)
    enc = DGIEncoder(data.num_node_features, dim).to(device)
    enc.eval()
    with torch.no_grad():
        return enc(data.x.to(device), data.edge_index.to(device)).detach()


def train_arm(data, arm: str, device: str, seed: int, emb: torch.Tensor,
              hid_dim: int = 64, heads: int = 8, dropout: float = 0.3,
              epochs: int = 300, lr: float = 5e-3, weight_decay: float = 5e-4,
              patience: int = 40) -> dict:
    """
    One arm, trained with E05's protocol.

    Written out here rather than routed through train_classifier because three
    of the five arms need a representation that train_classifier has no way to
    supply, and because gat_matched needs the widened head.
    """
    import time
    set_seed(seed)
    t0 = time.time()

    x, ei, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    trm, vam = data.train_mask.to(device), data.val_mask.to(device)

    zero = arm in ("gat_zero_dgi", "gat_matched")
    head = MATCHED_HEAD_DIM if arm == "gat_matched" else None
    model = FrozenDGIGATFusion(data.num_node_features,
                               0 if zero else emb.size(1),
                               hid_dim, heads, dropout,
                               zero_dgi=zero, head_dim=head).to(device)

    use = emb.to(device) if not zero else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    yt = data.y[data.train_mask]
    n_pos, n_neg = float((yt == 1).sum()), float((yt == 0).sum())
    tot = n_pos + n_neg
    w = torch.tensor([tot / (2 * max(n_neg, 1)), tot / (2 * max(n_pos, 1))],
                     dtype=torch.float, device=device)

    best_ap, bad, best_state, ep = -1.0, 0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, ei, use)[trm], y[trm], weight=w)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = F.softmax(model(x, ei, use), dim=1)[:, 1]
            ap = average_precision_score(y[vam].cpu().numpy(),
                                         p[vam].cpu().numpy())
        if ap > best_ap + 1e-5:
            best_ap, bad = ap, 0
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
        scores = F.softmax(model(x, ei, use), dim=1)[:, 1].cpu().numpy()

    return {"scores": scores, "best_val_ap": float(best_ap),
            "seconds": time.time() - t0, "epochs_run": ep + 1,
            "n_params": int(sum(p.numel() for p in model.parameters()
                                if p.requires_grad))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out", default="e18_representation_controls")
    args = ap.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print("  device: %s" % device)

    nodes, edges = load_elliptic()
    data = build_graph(nodes, edges, base_feature_columns(nodes))
    print("  graph: %d nodes, %d features"
          % (data.num_nodes, data.num_node_features))

    y, vam, tem = data.y.numpy(), data.val_mask.numpy(), data.test_mask.numpy()
    rows = []

    for seed in args.seeds:
        emb, info = train_dgi(data, "inductive", device, seed)
        rnd = random_embedding(data, device, seed, emb.size(1))
        set_seed(seed + 20_000)
        perm = torch.randperm(emb.size(0), device=emb.device)
        shuf = emb[perm].clone()
        # Matched to the real embedding's first two moments so that the only
        # thing withheld is the structure, not the scale.
        set_seed(seed + 30_000)
        noise = (torch.randn_like(emb) * emb.std() + emb.mean()).detach()
        print("\n  [seed %d] DGI inductive: %d epochs, loss %.4f, %.1fs"
              % (seed, info["epochs_run"], info["best_loss"], info["seconds"]))

        supply = {"fusion": emb, "fusion_random": rnd, "fusion_noise": noise,
                  "fusion_shuffled": shuf,
                  "gat_zero_dgi": emb, "gat_matched": emb}

        for arm in args.arms:
            r = train_arm(data, arm, device, seed, supply[arm])
            s = r["scores"]
            thr = select_threshold(y[vam], s[vam], criterion="f1")
            m = evaluate(y[tem], s[tem], thr)
            m.update(arm=arm, seed=seed, n_params=r["n_params"],
                     seconds=r["seconds"], epochs_run=r["epochs_run"],
                     best_val_ap=r["best_val_ap"])
            rows.append(m)
            print("    %-16s params=%7d F1=%.4f MCC=%.4f PR-AUC=%.4f (%.0fs)"
                  % (arm, r["n_params"], m["f1_illicit"], m["mcc"],
                     m["pr_auc"], r["seconds"]))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / ("%s.csv" % args.out), index=False)

    agg = (df.groupby("arm")
             .agg(n_params=("n_params", "first"),
                  f1=("f1_illicit", "mean"), f1_sd=("f1_illicit", "std"),
                  mcc=("mcc", "mean"), mcc_sd=("mcc", "std"),
                  pr_auc=("pr_auc", "mean"), seconds=("seconds", "mean"))
             .round(4).reset_index())

    # Paired tests against the fused arm, over seeds.
    from scipy import stats
    tests = []
    if "fusion" in set(df["arm"]):
        ref = df[df["arm"] == "fusion"].set_index("seed")["f1_illicit"]
        for arm in sorted(set(df["arm"]) - {"fusion"}):
            other = df[df["arm"] == arm].set_index("seed")["f1_illicit"]
            common = ref.index.intersection(other.index)
            if len(common) < 3:
                continue
            t, p = stats.ttest_rel(ref[common], other[common])
            tests.append({"comparison": "fusion vs %s" % arm,
                          "n_seeds": len(common),
                          "mean_fusion": round(float(ref[common].mean()), 4),
                          "mean_other": round(float(other[common].mean()), 4),
                          "difference": round(float((ref[common]
                                                     - other[common]).mean()), 4),
                          "t": round(float(t), 4),
                          "p_value": float(p)})
    pd.DataFrame(tests).to_csv(RESULTS / ("%s_tests.csv" % args.out), index=False)
    agg.to_csv(RESULTS / ("%s_aggregate.csv" % args.out), index=False)

    print("\n" + agg.to_string(index=False))
    if tests:
        print("\n" + pd.DataFrame(tests).to_string(index=False))
    (RESULTS / ("%s_summary.json" % args.out)).write_text(
        json.dumps({"arms": args.arms, "seeds": args.seeds,
                    "matched_head_dim": MATCHED_HEAD_DIM,
                    "aggregate": agg.to_dict("records"),
                    "paired_tests": tests}, indent=1), encoding="utf-8")
    print("\n  written to %s_aggregate.csv" % args.out)


if __name__ == "__main__":
    main()
