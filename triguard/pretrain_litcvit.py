"""Leakage-safe LitCVit NT-Xent pretraining on USTC training flows only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import UnlabeledETFlow
from model import LitCVit, NTXentLoss
from .runner import load_cache, set_seed


def pretrain(cache: str, output: str, seed: int = 42, epochs: int = 200, batch_size: int = 128,
            emb_dim: int = 192, rank: int = 16, num_workers: int = 8, log_every: int = 1) -> dict:
    """NT-Xent pretraining restricted to the manifest's USTC train partition.

    `emb_dim=192` matches the reference main-comparison LitCVit configuration;
    the resulting checkpoint must be loaded with the same `emb_dim` downstream.
    `num_workers>0` matters: the per-sample augmentation pipeline (random crop
    + bilinear resize via individual eager-mode ops) is expensive enough that
    a single-threaded `num_workers=0` loader made a 200-epoch, ~350k-flow run
    take upward of a day; multi-process loading parallelizes that cost.
    """
    data, metadata = load_cache(cache)
    set_seed(seed); device = "cuda" if torch.cuda.is_available() else "cpu"
    # Test and validation flows never enter self-supervised pretraining.
    train_indices = data["split"]["train"]
    dataset = UnlabeledETFlow(data["image"][train_indices])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) >= batch_size,
                        num_workers=num_workers, persistent_workers=num_workers > 0,
                        pin_memory=torch.cuda.is_available(), prefetch_factor=4 if num_workers > 0 else None)
    model = LitCVit(emb_dim=emb_dim, rank=rank, mode="pretrain").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = NTXentLoss(0.3); history = []
    started = time.time()
    for epoch in range(epochs):
        epoch_started = time.time()
        model.train(); losses = []
        for a, b in loader:
            opt.zero_grad(); loss = loss_fn(model(a.to(device)), model(b.to(device))); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else None
        history.append(mean_loss)
        if epoch % log_every == 0 or epoch == epochs - 1:
            elapsed = time.time() - started
            epoch_seconds = time.time() - epoch_started
            eta_seconds = epoch_seconds * (epochs - epoch - 1)
            print(f"[pretrain] epoch {epoch + 1}/{epochs} loss={mean_loss:.4f} "
                 f"epoch_time={epoch_seconds:.1f}s elapsed={elapsed:.0f}s eta={eta_seconds:.0f}s", flush=True)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "emb_dim": emb_dim, "rank": rank, "seed": seed, "epochs": epochs,
                "pretraining_partition": "USTC train only", "n_flows": len(train_indices)}, out)
    report = {"checkpoint": str(out.resolve()), "seed": seed, "epochs": epochs, "emb_dim": emb_dim, "rank": rank,
              "n_flows": len(train_indices), "loss": history, "train_time_seconds": time.time() - started,
              "validation_or_test_seen": False, "source_metadata": metadata.get("dataset", "USTC-TFC2016")}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", default="artifacts/ustc_full/ustc_multiview.npz"); parser.add_argument("--output", default="artifacts/ustc_full/litcvit_pretrain.pt"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=200); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--emb-dim", type=int, default=192); parser.add_argument("--rank", type=int, default=16); parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args(); print(json.dumps(pretrain(args.cache, args.output, args.seed, args.epochs, args.batch_size, args.emb_dim, args.rank, args.num_workers, args.log_every), indent=2))
