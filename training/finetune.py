"""
Fine-tuning LitCVit on each of the three datasets.

Fine-tuning protocol (Section IV-C / Algorithm 2):
  Phase 1 (epochs 0–99)  : encoder FROZEN  -> only classification head trained
  Phase 2 (epochs 100+)  : last two encoder blocks (pkt_enc + flow_enc) unfrozen
                           with a lower learning rate (1e-4 vs 1e-3 for head)
  Total epochs            : 500
  Loss                    : weighted cross-entropy (for class imbalance)
  Batch size              : 64
  Early stopping          : patience 30 epochs on validation F1

Usage:
    python training/finetune.py --dataset ustc   --ckpt checkpoints/pretrain/litcvit_pretrained_best.pt
    python training/finetune.py --dataset 5gad   --ckpt checkpoints/pretrain/litcvit_pretrained_best.pt
    python training/finetune.py --dataset tiissrc23 --ckpt checkpoints/pretrain/litcvit_pretrained_best.pt
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEVICE, EMB_DIM, RANK, BYTE_WIN, PACKET_WIN, DROPOUT,
    FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BS,
    UNFREEZE_EPOCH, FINETUNE_LR_HEAD, FINETUNE_LR_ENC,
    PROCESSED_DIR, RANDOM_SEED, DATASET_CONFIGS
)
from model import LitCVit
from data  import load_dataset, ETFlowSplit, get_class_weights


DATASET_KEY_MAP = {
    "ustc":      "USTC-TFC2016",
    "5gad":      "5GAD-2022",
    "tiissrc23": "TII-SSRC-23",
}


# ─────────────────────────────────────────────────────────────────────────────
# Single-epoch train / evaluate helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds   = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds   = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), correct / total, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main fine-tuning function
# ─────────────────────────────────────────────────────────────────────────────

def finetune(dataset_name: str,
             ckpt_path:    str  = None,
             out_dir:      str  = "checkpoints",
             processed_dir:str  = PROCESSED_DIR,
             epochs:       int  = FINETUNE_EPOCHS,
             batch_size:   int  = FINETUNE_BS,
             lr_head:      float = FINETUNE_LR_HEAD,
             lr_enc:       float = FINETUNE_LR_ENC,
             unfreeze_epoch:int  = UNFREEZE_EPOCH,
             num_workers:  int  = 0,
             patience:     int  = 50):

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ds_key = DATASET_KEY_MAP.get(dataset_name, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"LitCVit  |  Fine-tuning on {ds_key}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    images, labels, classes = load_dataset(dataset_name, processed_dir)
    n_classes = len(classes)
    print(f"\n  Classes: {n_classes}  ->  {classes}")

    splitter = ETFlowSplit(images, labels)
    tr_loader, va_loader, te_loader = splitter.get_loaders(
        batch_size=batch_size, num_workers=num_workers, class_balanced=True
    )

    # ── Build model ───────────────────────────────────────────────────────────
    if ckpt_path and os.path.exists(ckpt_path):
        model = LitCVit.from_pretrained(
            ckpt_path, n_classes=n_classes,
            emb_dim=EMB_DIM, rank=RANK, byte_win=BYTE_WIN, pkt_win=PACKET_WIN,
            dropout=DROPOUT
        )
    else:
        print("  [INFO] No pre-training checkpoint found -> training from scratch")
        model = LitCVit(emb_dim=EMB_DIM, rank=RANK, byte_win=BYTE_WIN,
                        pkt_win=PACKET_WIN, dropout=DROPOUT,
                        n_classes=n_classes, mode="finetune")
    model = model.to(DEVICE)

    # Phase 1: freeze encoder
    model.freeze_encoder()
    print(f"  Encoder frozen for first {unfreeze_epoch} epochs.")

    # ── Loss with class weights ───────────────────────────────────────────────
    class_wts = get_class_weights(splitter.train_labels).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_wts)

    # ── Optimiser (head only initially) ───────────────────────────────────────
    def make_optimizer(unfrozen: bool):
        if unfrozen:
            return torch.optim.AdamW([
                {"params": model.head.parameters(),    "lr": lr_head},
                {"params": model.encoder.pkt_enc.parameters(), "lr": lr_enc},
                {"params": model.encoder.flow_enc.parameters(), "lr": lr_enc},
                {"params": model.encoder.norm.parameters(),     "lr": lr_enc},
            ], weight_decay=1e-4)
        else:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_head, weight_decay=1e-4
            )

    optimizer  = make_optimizer(unfrozen=False)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1  = 0.0
    no_improve   = 0
    history      = {"epoch": [], "tr_loss": [], "tr_acc": [],
                     "va_loss": [], "va_acc": [], "va_f1": []}

    phase1_done = False
    for epoch in range(epochs):

        # Unfreeze last two blocks at epoch = unfreeze_epoch
        if epoch == unfreeze_epoch and not phase1_done:
            model.unfreeze_last_two_blocks()
            optimizer = make_optimizer(unfrozen=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - unfreeze_epoch, eta_min=1e-6
            )
            print(f"\n  [Epoch {epoch+1}] Unfreezing last two encoder blocks.")
            phase1_done = True

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        va_loss, va_acc, va_f1 = eval_epoch(model, va_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["epoch"].append(epoch + 1)
        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["va_loss"].append(va_loss)
        history["va_acc"].append(va_acc)
        history["va_f1"].append(va_f1)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:4d}/{epochs}] "
                  f"tr_loss: {tr_loss:.4f}  tr_acc: {tr_acc:.4f}  "
                  f"va_loss: {va_loss:.4f}  va_acc: {va_acc:.4f}  "
                  f"va_f1: {va_f1:.4f}  ({elapsed:.1f}s)")

        # ── Save best model ────────────────────────────────────────────────
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            no_improve  = 0
            best_path = os.path.join(out_dir, f"litcvit_{dataset_name}_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "n_classes": n_classes,
                "classes": classes,
            }, best_path)
        else:
            no_improve += 1

        # ── Early stopping ─────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs).")
            break

    # ── Save history ───────────────────────────────────────────────────────
    with open(os.path.join(out_dir,
              f"finetune_{dataset_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best validation Macro-F1: {best_val_f1:.4f}")
    print(f"  Model saved to: {best_path}")
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LitCVit Fine-tuning")
    parser.add_argument("--dataset",   required=True,
                        choices=["ustc", "5gad", "tiissrc23"])
    parser.add_argument("--ckpt",      default=None,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--out_dir",   default="checkpoints")
    parser.add_argument("--processed_dir", default=PROCESSED_DIR)
    parser.add_argument("--epochs",    type=int,   default=FINETUNE_EPOCHS)
    parser.add_argument("--batch_size",type=int,   default=FINETUNE_BS)
    parser.add_argument("--patience",  type=int,   default=50)
    parser.add_argument("--num_workers",type=int,  default=0)
    args = parser.parse_args()

    finetune(
        dataset_name  = args.dataset,
        ckpt_path     = args.ckpt,
        out_dir       = args.out_dir,
        processed_dir = args.processed_dir,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        patience      = args.patience,
        num_workers   = args.num_workers,
    )
