"""
E11 - Computational cost on an idle device.

Reviewer 2 asked for training time, inference time, hardware specification,
peak memory and complexity, for practical deployment.

This is measured separately from E05 rather than reused from it. During part of
the E05 run a duplicate process was active on the same GPU, so the per-model
timings recorded there reflect contention and are not reported. The figures
below are collected with a single process on an otherwise idle device, and the
script refuses to run if another CUDA process is detected.

Complexity, for reference. Let $N$ be nodes, $E$ edges, $F$ input features, $H$
hidden width and $K$ attention heads. A GCN or DGI layer costs
$O(E F + N F H)$ per forward pass. A GAT layer adds per-edge attention,
$O(E K H + N F K H)$. Full-batch training multiplies by the epoch count. The
frozen fusion adds one concatenation and a two-layer head, $O(N (H + D) H)$ for
a DGI width $D$, and adds nothing to the message-passing cost because the
encoder is not re-run.

Outputs
-------
  results/e11_compute_cost.csv
  results/e11_compute_cost.json
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import RESULTS, DEVICE, PRIMARY_SEED
from data import load_elliptic, base_feature_columns
from gnn import build_graph, train_dgi, train_classifier

MODELS = ["fusion", "gat_zero_dgi", "gat", "gcn", "sage"]
N_INFERENCE_REPEATS = 20


def other_cuda_processes() -> int:
    """Count python processes holding the GPU, excluding this one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return 0
    pids = [int(x) for x in out.split() if x.strip().isdigit()]
    return len([p for p in pids if p != __import__("os").getpid()])


def hardware_profile() -> dict:
    prof = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        prof.update(gpu_name=p.name,
                    gpu_total_memory_gb=round(p.total_memory / 1024 ** 3, 2),
                    gpu_multiprocessors=p.multi_processor_count)
    return prof


def main() -> None:
    n_other = other_cuda_processes()
    if n_other:
        print(f"  refusing to benchmark: {n_other} other CUDA process(es) active")
        print("  wait for them to finish, then rerun")
        return

    device = DEVICE if torch.cuda.is_available() else "cpu"
    prof = hardware_profile()
    print("  hardware:", json.dumps(prof, indent=2))

    nodes, edges = load_elliptic()
    cols = base_feature_columns(nodes)

    t0 = time.time()
    data = build_graph(nodes, edges, cols)
    graph_seconds = time.time() - t0

    rows = []

    # Self-supervised pre-training, both regimes.
    for regime in ("transductive", "inductive"):
        torch.cuda.reset_peak_memory_stats(device) if device.startswith("cuda") else None
        emb, info = train_dgi(data, regime, device, PRIMARY_SEED)
        peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.startswith("cuda") else float("nan"))
        rows.append({"component": f"DGI pre-training ({regime})",
                     "train_seconds": round(info["seconds"], 2),
                     "epochs": info["epochs_run"],
                     "peak_mem_mb": round(peak, 1),
                     "n_params": sum(p.numel() for p in
                                     __import__("models").DGI(
                                         data.num_node_features, 128).parameters()),
                     "inference_ms": None})
        print(f"    DGI {regime:13s} {info['seconds']:7.2f}s  "
              f"{info['epochs_run']:3d} epochs  peak {peak:7.1f} MB")
        if regime == "inductive":
            emb_ind = emb

    # Classifiers.
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    for kind in MODELS:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        r = train_classifier(data, kind, device, PRIMARY_SEED,
                             dgi_emb=emb_ind, return_model=True)
        model = r["model"]
        model.eval()
        needs_emb = kind in ("fusion", "gat_zero_dgi")
        e = emb_ind.to(device) if needs_emb else None

        # Full-graph inference latency.
        with torch.no_grad():
            for _ in range(3):
                _ = model(x, ei, e) if needs_emb else model(x, ei)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.time()
            for _ in range(N_INFERENCE_REPEATS):
                _ = model(x, ei, e) if needs_emb else model(x, ei)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            infer_ms = (time.time() - t) / N_INFERENCE_REPEATS * 1000

        rows.append({"component": kind,
                     "train_seconds": round(r["seconds"], 2),
                     "epochs": r["epochs_run"],
                     "peak_mem_mb": round(r["peak_mem_mb"], 1),
                     "n_params": r["n_params"],
                     "inference_ms": round(infer_ms, 2)})
        print(f"    {kind:14s} train {r['seconds']:7.2f}s  "
              f"{r['epochs_run']:3d} ep  peak {r['peak_mem_mb']:7.1f} MB  "
              f"params {r['n_params']:>8,}  inference {infer_ms:6.2f} ms "
              f"({infer_ms / data.num_nodes * 1e3:.4f} us/node)")

    df = pd.DataFrame(rows)
    df["us_per_node"] = (df["inference_ms"] / data.num_nodes * 1e3).round(4)
    df.to_csv(RESULTS / "e11_compute_cost.csv", index=False)

    out = {
        "hardware": prof,
        "graph_build_seconds": round(graph_seconds, 2),
        "n_nodes": int(data.num_nodes),
        "n_edges_directed": int(data.num_edges),
        "n_features": int(data.num_node_features),
        "components": rows,
        "complexity": {
            "gcn_or_dgi_layer": "O(E F + N F H)",
            "gat_layer": "O(E K H + N F K H)",
            "fusion_head": "O(N (H + D) H), encoder not re-run",
            "full_batch_training": "per-epoch cost times epoch count",
        },
        "note": ("Measured with a single process on an idle device. E05 "
                 "timings are not reported because a duplicate process shared "
                 "the GPU during part of that run."),
    }
    (RESULTS / "e11_compute_cost.json").write_text(json.dumps(out, indent=2),
                                                   encoding="utf-8")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
