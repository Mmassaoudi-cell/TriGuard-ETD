"""
Self-supervised pre-training of LitCVit using NT-Xent contrastive loss.

Training protocol (Section VI-B / Algorithm 1):
  • Datasets used for pre-training : USTC-TFC2016  (pretrain_eligible=True)
  • Optimizer  : AdamW
  • LR schedule: cosine annealing with linear warmup
  • Batch size : 128
  • Temperature: τ = 0.3
  • Epochs     : PRETRAIN_EPOCHS (200 by default)

Usage:
    python training/pretrain.py  --out_dir checkpoints/pretrain
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEVICE, EMB_DIM, RANK, BYTE_WIN, PACKET_WIN, DROPOUT,
    PRETRAIN_EPOCHS, PRETRAIN_LR, PRETRAIN_BS, PRETRAIN_WARMUP,
    TEMPERATURE, PROCESSED_DIR, RANDOM_SEED
)
from model import LitCVit, NTXentLoss
from data  import UnlabeledETFlow, load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Cosine LR schedule with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr_with_warmup(optimizer, epoch: int, total_epochs: int,
                          warmup_epochs: int, base_lr: float):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(out_dir: str,
             processed_dir: str = PROCESSED_DIR,
             epochs: int        = PRETRAIN_EPOCHS,
             batch_size: int    = PRETRAIN_BS,
             lr: float          = PRETRAIN_LR,
             warmup: int        = PRETRAIN_WARMUP,
             temperature: float = TEMPERATURE,
             num_workers: int   = 0):

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Load unlabelled pre-training data (USTC-TFC2016) ─────────────────────
    print("=" * 60)
    print("LitCVit  |  Self-Supervised Pre-training")
    print("=" * 60)

    print("\nLoading pre-training datasets…")
    pretrain_datasets = []

    try:
        images_ustc, _, _ = load_dataset("ustc", processed_dir)
        pretrain_datasets.append(UnlabeledETFlow(images_ustc))
        print(f"  USTC-TFC2016: {len(images_ustc):,} unlabelled flows")
    except FileNotFoundError as e:
        print(f"  [WARNING] {e}")

    if not pretrain_datasets:
        print("[ERROR] No pre-training data found. "
              "Run 'python data/preprocess_pcap.py --dataset ustc' first.")
        return None

    combined = ConcatDataset(pretrain_datasets)
    loader   = DataLoader(combined, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(DEVICE=="cuda"),
                          drop_last=True)
    print(f"\n  Total unlabelled samples: {len(combined):,}")
    print(f"  Batch size              : {batch_size}")
    print(f"  Batches per epoch       : {len(loader)}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = LitCVit(
        emb_dim=EMB_DIM, rank=RANK, byte_win=BYTE_WIN, pkt_win=PACKET_WIN,
        dropout=DROPOUT, mode="pretrain"
    ).to(DEVICE)
    print(f"\n  Trainable parameters: {model.n_parameters:,}")

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = NTXentLoss(temperature=temperature)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    history   = {"epoch": [], "loss": [], "lr": []}

    print(f"\n  Starting pre-training for {epochs} epochs on {DEVICE.upper()}…\n")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        current_lr = cosine_lr_with_warmup(optimizer, epoch, epochs, warmup, lr)

        for v1, v2 in loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            optimizer.zero_grad()
            z1 = model(v1)      # (B, proj_dim)  – already L2-normalised in head
            z2 = model(v2)
            loss = criterion(z1, z2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:4d}/{epochs}] "
                  f"loss: {avg_loss:.4f}  lr: {current_lr:.2e}  "
                  f"time: {elapsed:.1f}s")

        # ── Save best checkpoint ───────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(out_dir, "litcvit_pretrained_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": {
                    "emb_dim": EMB_DIM, "rank": RANK,
                    "byte_win": BYTE_WIN, "pkt_win": PACKET_WIN,
                    "temperature": temperature,
                }
            }, ckpt_path)

    # ── Save final checkpoint ──────────────────────────────────────────────
    final_ckpt = os.path.join(out_dir, "litcvit_pretrained_final.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
    }, final_ckpt)

    # ── Save training history ──────────────────────────────────────────────
    import json
    with open(os.path.join(out_dir, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Pre-training complete. Best loss: {best_loss:.4f}")
    print(f"  Checkpoints saved to: {out_dir}")
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LitCVit Pre-training")
    parser.add_argument("--out_dir",      default="checkpoints/pretrain")
    parser.add_argument("--processed_dir",default=PROCESSED_DIR)
    parser.add_argument("--epochs",       type=int,   default=PRETRAIN_EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=PRETRAIN_BS)
    parser.add_argument("--lr",           type=float, default=PRETRAIN_LR)
    parser.add_argument("--warmup",       type=int,   default=PRETRAIN_WARMUP)
    parser.add_argument("--temperature",  type=float, default=TEMPERATURE)
    parser.add_argument("--num_workers",  type=int,   default=0)
    args = parser.parse_args()

    pretrain(
        out_dir       = args.out_dir,
        processed_dir = args.processed_dir,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        warmup        = args.warmup,
        temperature   = args.temperature,
        num_workers   = args.num_workers,
    )
