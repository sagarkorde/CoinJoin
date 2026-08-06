"""
run_seed_experiment.py  |  Multi-seed validation + clean GAT-without-DGI ablation
====================================================================================
Reviewer-requested experiments, run per seed:

  1. Retrain the DGI encoder (300 epochs, same hyperparameters as Step 7).
  2. Retrain MixTrace GAT with the real DGI skip connection (500 epochs,
     same hyperparameters as Step 9).
  3. Retrain a CLEAN "GAT without DGI" ablation: a separate model trained
     from epoch 1 with dgi_embeds=None throughout (matching what the paper
     text claims), instead of the original pipeline's approach of zeroing
     the DGI input only at test time on a model that saw real DGI
     embeddings for all 500 training epochs.
  4. Retrain K-Means / KNN / Random Forest baselines on this seed.

Usage
-----
    python run_seed_experiment.py --seed 1 --out results_multiseed/seed_1.json

Each invocation is independent and safe to run in parallel across seeds.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATConv, GCNConv

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_PROC = HERE / "data" / "processed"

# ── Hyperparameters (identical to Steps 5, 7, 9) ──
DGI_HIDDEN, DGI_EMBED, DGI_EPOCHS, DGI_LR, DGI_WD, DGI_DROPOUT = 256, 128, 300, 1e-3, 5e-4, 0.3
GAT_HIDDEN, GAT_HEADS, GAT_EPOCHS, GAT_LR, GAT_WD, GAT_DROPOUT, GAT_PATIENCE = 128, 8, 500, 5e-4, 1e-4, 0.5, 20


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ── DGI ──
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, z, summary):
        Ws = torch.matmul(self.weight, summary.unsqueeze(-1))
        return torch.matmul(z, Ws).squeeze(-1)


class DGIModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, dropout):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, embed_dim, dropout)
        self.discriminator = Discriminator(embed_dim)

    def forward(self, x, edge_index, x_neg):
        z_pos = self.encoder(x, edge_index)
        z_neg = self.encoder(x_neg, edge_index)
        summary = torch.sigmoid(z_pos.mean(dim=0))
        return self.discriminator(z_pos, summary), self.discriminator(z_neg, summary)


def corrupt_features(x, rng):
    perm = torch.randperm(x.size(0), generator=rng)
    return x[perm]


def train_dgi(data, device, seed):
    rng = torch.Generator().manual_seed(seed)
    in_dim = data.num_node_features
    model = DGIModel(in_dim, DGI_HIDDEN, DGI_EMBED, DGI_DROPOUT).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=DGI_LR, weight_decay=DGI_WD)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    N = x.size(0)
    pos_labels, neg_labels = torch.ones(N, device=device), torch.zeros(N, device=device)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_state = float("inf"), None
    for epoch in range(1, DGI_EPOCHS + 1):
        model.train()
        optim.zero_grad()
        x_neg = corrupt_features(x, rng)
        pos_scores, neg_scores = model(x, edge_index, x_neg)
        loss = criterion(pos_scores, pos_labels) + criterion(neg_scores, neg_labels)
        loss.backward()
        optim.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    model.encoder.load_state_dict(best_state)
    return model.encoder


# ── GAT ──
class MixTraceGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_heads, n_classes, dgi_embed_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=n_heads, dropout=dropout, concat=True)
        self.bn1 = BatchNorm(hidden_dim * n_heads)
        self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.bn2 = BatchNorm(hidden_dim)
        self.skip_proj = nn.Linear(dgi_embed_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, edge_index, dgi_embeds=None):
        h = F.elu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn1(self.gat1(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.gat2(h, edge_index)))
        if dgi_embeds is not None:
            skip = F.elu(self.skip_proj(dgi_embeds))
            h = torch.cat([h, skip], dim=-1)
        else:
            h = torch.cat([h, torch.zeros_like(h)], dim=-1)
        return self.classifier(h)


class CleanGATNoDGI(nn.Module):
    """Genuinely DGI-free architecture: 128-dim GAT output feeding its own
    correctly-sized MLP head directly, with no zero-padded concatenation."""
    def __init__(self, in_dim, hidden_dim, n_heads, n_classes, dropout):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=n_heads, dropout=dropout, concat=True)
        self.bn1 = BatchNorm(hidden_dim * n_heads)
        self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.bn2 = BatchNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, edge_index):
        h = F.elu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn1(self.gat1(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.gat2(h, edge_index)))
        return self.classifier(h)


def compute_class_weights(y, mask):
    y_train = y[mask & (y >= 0)]
    n_ill = (y_train == 1).sum().item()
    n_lic = (y_train == 0).sum().item()
    n_total = n_ill + n_lic
    w_ill = n_total / (2.0 * n_ill) if n_ill > 0 else 1.0
    w_lic = n_total / (2.0 * n_lic) if n_lic > 0 else 1.0
    return torch.tensor([w_lic, w_ill], dtype=torch.float)


@torch.no_grad()
def evaluate_model_dev(model, x, edge_index, y_dev, mask_dev, dgi_embeds=None, has_dgi_arg=True):
    """All tensor args must already be on the target device."""
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, dgi_embeds) if has_dgi_arg else model(x, edge_index)
        probs = F.softmax(logits, dim=-1)[:, 1]
        labeled_mask = mask_dev & (y_dev >= 0)
        y_true = y_dev[labeled_mask].cpu().numpy()
        y_prob = probs[labeled_mask].cpu().numpy()
    y_pred = (y_prob > 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"f1_illicit": 0.0, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0, "roc_auc": 0.0}
    return {
        "f1_illicit": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def evaluate_model(model, data, mask, device, dgi_embeds=None, has_dgi_arg=True):
    """Convenience wrapper that transfers everything fresh -- only for one-off
    (non-hot-path) calls, e.g. the final test-set evaluation."""
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y_dev = data.y.to(device)
    mask_dev = mask.to(device)
    return evaluate_model_dev(model, x, edge_index, y_dev, mask_dev, dgi_embeds, has_dgi_arg)


def train_gat(model, data, device, class_weights, dgi_embeds=None, has_dgi_arg=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=GAT_LR, weight_decay=GAT_WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # transfer once -- reused for every training step and every validation check
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    labeled_train = train_mask & (y >= 0)
    best_val_f1, patience_cnt, best_state = 0.0, 0, None
    for epoch in range(1, GAT_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, dgi_embeds) if has_dgi_arg else model(x, edge_index)
        loss = criterion(logits[labeled_train], y[labeled_train])
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            val_metrics = evaluate_model_dev(model, x, edge_index, y, val_mask, dgi_embeds, has_dgi_arg)
            scheduler.step(val_metrics["f1_illicit"])
            if val_metrics["f1_illicit"] > best_val_f1:
                best_val_f1 = val_metrics["f1_illicit"]
                patience_cnt = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
            if patience_cnt >= GAT_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── Baselines ──
def load_elliptic_splits():
    X_all = np.load(DATA_PROC / "elliptic_X_all.npy")
    y_all = np.load(DATA_PROC / "elliptic_y_all.npy")
    mask_train = np.load(DATA_PROC / "elliptic_mask_train.npy")
    mask_val = np.load(DATA_PROC / "elliptic_mask_val.npy")
    mask_test = np.load(DATA_PROC / "elliptic_mask_test.npy")
    labeled = y_all >= 0
    train_idx = np.where(mask_train & labeled)[0]
    val_idx = np.where(mask_val & labeled)[0]
    test_idx = np.where(mask_test & labeled)[0]
    return (X_all[train_idx], y_all[train_idx], X_all[val_idx], y_all[val_idx],
            X_all[test_idx], y_all[test_idx])


def compute_metrics(y_true, y_pred, y_proba=None):
    m = {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_illicit": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        m["pr_auc"] = float(average_precision_score(y_true, y_proba))
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        m["pr_auc"] = 0.0
        m["roc_auc"] = 0.0
    return m


def run_baselines(seed):
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_elliptic_splits()
    out = {}

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2,
                                 class_weight="balanced", random_state=seed, n_jobs=4)
    X_tv, y_tv = np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val])
    rf.fit(X_tv, y_tv)
    y_pred, y_proba = rf.predict(X_te), rf.predict_proba(X_te)[:, 1]
    out["RandomForest"] = compute_metrics(y_te, y_pred, y_proba)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=4)
    knn.fit(X_tv, y_tv)
    y_pred, y_proba = knn.predict(X_te), knn.predict_proba(X_te)[:, 1]
    out["KNN"] = compute_metrics(y_te, y_pred, y_proba)

    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    X_all = np.vstack([X_tr, X_te])
    km.fit(X_all)
    train_clusters = km.predict(X_tr)
    illicit_cluster = int(pd.Series(train_clusters[y_tr == 1]).value_counts().idxmax())
    y_pred = (km.predict(X_te) == illicit_cluster).astype(int)
    out["KMeans"] = compute_metrics(y_te, y_pred, None)

    # ADASYN ablation on Random Forest: same features, with vs without ADASYN
    from imblearn.over_sampling import ADASYN
    try:
        ada = ADASYN(random_state=seed, n_neighbors=5)
        X_res, y_res = ada.fit_resample(X_tr, y_tr)
        rf_ada = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2,
                                         class_weight="balanced", random_state=seed, n_jobs=4)
        rf_ada.fit(np.vstack([X_res, X_val]), np.concatenate([y_res, y_val]))
        y_pred, y_proba = rf_ada.predict(X_te), rf_ada.predict_proba(X_te)[:, 1]
        out["RandomForest_ADASYN"] = compute_metrics(y_te, y_pred, y_proba)
    except Exception as e:
        out["RandomForest_ADASYN"] = {"error": str(e)}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dgi-epochs", type=int, default=None, help="override for smoke testing")
    ap.add_argument("--gat-epochs", type=int, default=None, help="override for smoke testing")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = ap.parse_args()

    global DGI_EPOCHS, GAT_EPOCHS
    if args.dgi_epochs:
        DGI_EPOCHS = args.dgi_epochs
    if args.gat_epochs:
        GAT_EPOCHS = args.gat_epochs

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"[seed {args.seed}] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
          flush=True)
    t_start = time.time()
    result = {"seed": args.seed}

    print(f"[seed {args.seed}] loading graph...", flush=True)
    data = torch.load(DATA_PROC / "elliptic_graph.pt", weights_only=False)
    in_dim = data.num_node_features
    class_weights = compute_class_weights(data.y, data.train_mask)

    # 1. DGI pretraining
    set_all_seeds(args.seed)
    t0 = time.time()
    print(f"[seed {args.seed}] training DGI encoder...", flush=True)
    encoder = train_dgi(data, device, args.seed)
    encoder.eval()
    with torch.no_grad():
        dgi_embeds = encoder(data.x.to(device), data.edge_index.to(device))
    print(f"[seed {args.seed}] DGI done in {time.time()-t0:.0f}s", flush=True)

    # 2. MixTrace GAT (with real DGI)
    set_all_seeds(args.seed)
    t0 = time.time()
    print(f"[seed {args.seed}] training MixTrace GAT (DGI+GAT)...", flush=True)
    model_full = MixTraceGAT(in_dim, GAT_HIDDEN, GAT_HEADS, 2, DGI_EMBED, GAT_DROPOUT).to(device)
    model_full = train_gat(model_full, data, device, class_weights, dgi_embeds, has_dgi_arg=True)
    result["mixtrace_dgi_gat"] = evaluate_model(model_full, data, data.test_mask.to(device), device, dgi_embeds, True)
    print(f"[seed {args.seed}] MixTrace GAT done in {time.time()-t0:.0f}s -> F1={result['mixtrace_dgi_gat']['f1_illicit']:.4f}", flush=True)

    # 3. Clean GAT-without-DGI ablation (trained from scratch, no DGI ever)
    set_all_seeds(args.seed)
    t0 = time.time()
    print(f"[seed {args.seed}] training clean GAT-without-DGI ablation...", flush=True)
    model_clean = CleanGATNoDGI(in_dim, GAT_HIDDEN, GAT_HEADS, 2, GAT_DROPOUT).to(device)
    model_clean = train_gat(model_clean, data, device, class_weights, dgi_embeds=None, has_dgi_arg=False)
    result["gat_no_dgi_clean"] = evaluate_model(model_clean, data, data.test_mask.to(device), device, None, False)
    print(f"[seed {args.seed}] Clean no-DGI GAT done in {time.time()-t0:.0f}s -> F1={result['gat_no_dgi_clean']['f1_illicit']:.4f}", flush=True)

    # 4. Baselines (+ ADASYN ablation)
    t0 = time.time()
    print(f"[seed {args.seed}] training baselines...", flush=True)
    result["baselines"] = run_baselines(args.seed)
    print(f"[seed {args.seed}] baselines done in {time.time()-t0:.0f}s", flush=True)

    result["total_time_s"] = time.time() - t_start
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[seed {args.seed}] ALL DONE in {result['total_time_s']:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
