"""
E13 - Training diagnostics regenerated under the corrected protocol.

Three figures in the manuscript were produced by the superseded run that
included the withdrawn features: the self-supervised loss curve, the
supervised training curves, and a t-SNE projection of the learned embedding.
The last carried a substantive claim, that illicit nodes occupy distinct
regions of the embedding space before any supervised label is used. With a
feature that re-encodes the label present in the input, that separation could
be an artefact rather than a property of self-supervised learning, so the
figure is regenerated on the 165 distributed features alone and the claim is
re-examined against the result.

Produces
--------
  figures/fig_dgi_loss.png          self-supervised loss, both regimes
  figures/fig_training_curves.png   supervised curves, primary seed
  figures/fig_embedding_tsne.png    t-SNE of the frozen representation
  results/e13_training_curves.csv
  results/e13_tsne_separation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, FIGURES, DEVICE, PRIMARY_SEED, TRAIN_END
from data import load_elliptic, base_feature_columns
from gnn import build_graph, set_seed
from models import DGI, FrozenDGIGATFusion
from plots import C_PRIMARY, C_ACCENT, C_MUTED, C_OK

TSNE_SAMPLE = 6000


def dgi_loss_curve(data, regime, device, seed, epochs=200, lr=1e-3):
    """Train DGI, returning the per-epoch loss and the frozen embedding."""
    set_seed(seed)
    if regime == "inductive":
        keep = data.time_step <= TRAIN_END
        idx = keep.nonzero(as_tuple=True)[0]
        remap = -torch.ones(data.num_nodes, dtype=torch.long)
        remap[idx] = torch.arange(idx.numel())
        ei = data.edge_index
        em = keep[ei[0]] & keep[ei[1]]
        x_fit, ei_fit = data.x[idx].to(device), remap[ei[:, em]].to(device)
    else:
        x_fit, ei_fit = data.x.to(device), data.edge_index.to(device)

    model = DGI(data.num_node_features, 128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = model.loss(x_fit, ei_fit)
        loss.backward(); opt.step()
        losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        emb = model.encoder(data.x.to(device), data.edge_index.to(device)).detach()
    return losses, emb


def supervised_curve(data, emb, device, seed, epochs=300, lr=5e-3, wd=5e-4):
    """Train the fused classifier, logging per-epoch train and validation F1."""
    set_seed(seed)
    x, ei = data.x.to(device), data.edge_index.to(device)
    y = data.y.to(device)
    trm, vam = data.train_mask.to(device), data.val_mask.to(device)
    e = emb.to(device)

    model = FrozenDGIGATFusion(data.num_node_features, e.size(1), 64, 8, 0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    yt = data.y[data.train_mask]
    n_pos, n_neg = float((yt == 1).sum()), float((yt == 0).sum())
    tot = n_pos + n_neg
    w = torch.tensor([tot / (2 * max(n_neg, 1)), tot / (2 * max(n_pos, 1))],
                     dtype=torch.float, device=device)

    rows = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(x, ei, e)
        loss = F.cross_entropy(out[trm], y[trm], weight=w)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            p = F.softmax(model(x, ei, e), dim=1)[:, 1]
        ytr_np, ptr = y[trm].cpu().numpy(), p[trm].cpu().numpy()
        yva_np, pva = y[vam].cpu().numpy(), p[vam].cpu().numpy()
        rows.append({
            "epoch": ep,
            "loss": float(loss.item()),
            "train_f1": f1_score(ytr_np, (ptr >= 0.5).astype(int), zero_division=0),
            "val_f1": f1_score(yva_np, (pva >= 0.5).astype(int), zero_division=0),
            "val_ap": average_precision_score(yva_np, pva),
        })
    return pd.DataFrame(rows)


def main() -> None:
    device = DEVICE if torch.cuda.is_available() else "cpu"
    nodes, edges = load_elliptic()
    cols = base_feature_columns(nodes)
    data = build_graph(nodes, edges, cols)
    print(f"  graph: {data.num_nodes:,} nodes, {data.num_node_features} features "
          "(165 distributed only, no withdrawn features)")

    # -- self-supervised loss, both regimes --------------------------------
    curves, embs = {}, {}
    for regime in ("transductive", "inductive"):
        losses, emb = dgi_loss_curve(data, regime, device, PRIMARY_SEED)
        curves[regime], embs[regime] = losses, emb
        print(f"    DGI {regime:13s} final loss {losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for (regime, ls), col in zip(curves.items(), (C_PRIMARY, C_ACCENT)):
        ax.plot(ls, lw=1.4, color=col, label=regime)
    ax.set_xlabel("Pre-training epoch"); ax.set_ylabel("DGI loss")
    ax.set_title("Self-supervised pre-training loss, 165 distributed features")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES / "fig_dgi_loss.png"); plt.close(fig)

    # -- supervised curves --------------------------------------------------
    df = supervised_curve(data, embs["inductive"], device, PRIMARY_SEED)
    df.to_csv(RESULTS / "e13_training_curves.csv", index=False)
    print(f"    supervised: best validation AP {df['val_ap'].max():.4f} "
          f"at epoch {int(df['val_ap'].idxmax())}")

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(df["epoch"], df["train_f1"], lw=1.4, color=C_PRIMARY, label="Train $F_1$")
    ax.plot(df["epoch"], df["val_f1"], lw=1.4, color=C_ACCENT, label="Validation $F_1$")
    ax.plot(df["epoch"], df["val_ap"], lw=1.2, ls="--", color=C_OK,
            label="Validation PR-AUC")
    best = int(df["val_ap"].idxmax())
    ax.axvline(best, color=C_MUTED, ls=":", lw=1)
    ax.text(best + 4, 0.06, f"model selected\nepoch {best}", fontsize=7.5, color=C_MUTED)
    ax.set_xlabel("Supervised training epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Supervised training, fused model, primary seed")
    ax.legend(fontsize=8, loc="center right")
    fig.savefig(FIGURES / "fig_training_curves.png"); plt.close(fig)

    # -- t-SNE of the frozen representation ---------------------------------
    y = data.y.numpy()
    labelled = np.flatnonzero(nodes["label"].to_numpy() != -1)
    rng = np.random.default_rng(PRIMARY_SEED)
    sample = rng.choice(labelled, size=min(TSNE_SAMPLE, len(labelled)), replace=False)
    Z = embs["inductive"].cpu().numpy()[sample]
    ys = y[sample]

    proj = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=PRIMARY_SEED).fit_transform(Z)

    # Quantify separation rather than asserting it from the picture.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Two probes, because they answer different questions. The random split
    # measures how separable the representation is in distribution; the
    # chronological split measures whether that separability transfers to a
    # later period, which is the question the rest of the study asks.
    Zall = embs["inductive"].cpu().numpy()
    lab_mask = nodes["label"].to_numpy() != -1
    ts_all = data.time_step.numpy()
    tr_p = lab_mask & (ts_all <= TRAIN_END)
    te_p = lab_mask & (ts_all >= 42)

    auc_rand = cross_val_score(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        Z, ys, cv=5, scoring="roc_auc").mean()

    probe = LogisticRegression(max_iter=2000, class_weight="balanced")
    probe.fit(Zall[tr_p], y[tr_p])
    auc_chrono = roc_auc_score(y[te_p], probe.predict_proba(Zall[te_p])[:, 1])

    auc = auc_rand
    sep = {
        "n_sampled": int(len(sample)),
        "linear_probe_roc_auc_random_cv": round(float(auc_rand), 4),
        "linear_probe_roc_auc_chronological": round(float(auc_chrono), 4),
        "note": ("Two linear probes on the frozen representation. The random "
                 "five-fold value mixes time periods and measures in-"
                 "distribution separability. The chronological value fits on "
                 "the training period and scores the test period, and is the "
                 "figure comparable with the rest of the study."),
    }
    (RESULTS / "e13_tsne_separation.json").write_text(json.dumps(sep, indent=2),
                                                      encoding="utf-8")
    print(f"    linear probe on frozen embedding: ROC-AUC {auc:.4f}")

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(proj[ys == 0, 0], proj[ys == 0, 1], s=3, alpha=0.35,
               color=C_PRIMARY, label="Licit", linewidths=0)
    ax.scatter(proj[ys == 1, 0], proj[ys == 1, 1], s=5, alpha=0.75,
               color=C_ACCENT, label="Illicit", linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Frozen self-supervised representation\n"
                 f"linear probe ROC-AUC {auc_chrono:.3f} chronological, "
                 f"{auc_rand:.3f} in-distribution", fontsize=8.5)
    ax.legend(fontsize=8, markerscale=2.5)
    fig.savefig(FIGURES / "fig_embedding_tsne.png"); plt.close(fig)
    print("\n[OK] wrote three figures and two result files")


if __name__ == "__main__":
    main()
