"""
Comprehensive evaluation of a fine-tuned LitCVit model.

Reported metrics (Table III of paper):
  • Accuracy   (overall)
  • Precision  (macro)
  • Recall     (macro)
  • F1-score   (macro)
  • TPR  (per-class and macro)
  • TNR  (per-class and macro)
  • FPR  (per-class and macro)
  • FNR  (per-class and macro)

Also generates:
  • Confusion matrix (Fig. 4 style)
  • Precision-Recall curves (Fig. 5 style)
  • Per-class breakdown table
  • UMAP embedding visualisation (optional)

Usage:
    python evaluation/evaluate.py  --dataset ustc
                                    --ckpt checkpoints/litcvit_ustc_best.pt
                                    --out_dir results/ustc
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEVICE, EMB_DIM, RANK, BYTE_WIN, PACKET_WIN, DROPOUT, PROCESSED_DIR
from model  import LitCVit
from data   import load_dataset, ETFlowSplit, ETFlowDataset


DATASET_KEY_MAP = {
    "ustc":      "USTC-TFC2016",
    "5gad":      "5GAD-2022",
    "tiissrc23": "TII-SSRC-23",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: LitCVit, loader: DataLoader, device: str = DEVICE):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_probs.append(probs)
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs))


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, classes):
    """
    Compute overall and per-class classification metrics.
    Returns a dict with keys: accuracy, precision, recall, f1,
                               tpr, tnr, fpr, fnr  (all macro-averaged)
    and per_class list of dicts.
    """
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(classes))))
    n_cls = len(classes)

    per_class = []
    for i, cls_name in enumerate(classes):
        tp  = cm[i, i]
        fn  = cm[i, :].sum() - tp
        fp  = cm[:, i].sum() - tp
        tn  = cm.sum() - tp - fn - fp

        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        n_support = cm[i, :].sum()

        per_class.append({
            "class":     cls_name,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "tpr":       tpr,
            "tnr":       tnr,
            "fpr":       fpr,
            "fnr":       fnr,
            "support":   int(n_support),
        })

    # Macro averages
    acc  = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    tpr_m  = np.mean([c["tpr"] for c in per_class])
    tnr_m  = np.mean([c["tnr"] for c in per_class])
    fpr_m  = np.mean([c["fpr"] for c in per_class])
    fnr_m  = np.mean([c["fnr"] for c in per_class])

    return {
        "accuracy":  acc,
        "precision": prec_m,
        "recall":    rec_m,
        "f1":        f1_m,
        "tpr":       tpr_m,
        "tnr":       tnr_m,
        "fpr":       fpr_m,
        "fnr":       fnr_m,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_arr, classes, title, save_path):
    cm = np.array(cm_arr)
    # Row-normalise to percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes))))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curves(y_true, y_probs, classes, title, save_path):
    n_cls = len(classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    # One-vs-rest PR curve for each class
    for i, cls_name in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, y_probs[:, i])
        ap = average_precision_score(y_bin, y_probs[:, i])
        ax.plot(rec, prec, linewidth=1.2, label=f"{cls_name} (AP={ap:.2f})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    if n_cls <= 12:
        ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tpr_fpr_bars(metrics_dict, dataset_names, save_path):
    """Replicate Fig. 6 style: TPR and FPR comparison across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tprs = [metrics_dict[d]["tpr"] for d in dataset_names]
    fprs = [metrics_dict[d]["fpr"] for d in dataset_names]

    x = np.arange(len(dataset_names))
    axes[0].bar(x, tprs, color="#4C72B0", width=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.replace("_", "\n") for d in dataset_names])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("TPR (Macro)")
    axes[0].set_title("True Positive Rate")

    axes[1].bar(x, fprs, color="#DD8452", width=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.replace("_", "\n") for d in dataset_names])
    axes[1].set_ylim(0, max(max(fprs) * 1.3, 0.05))
    axes[1].set_ylabel("FPR (Macro)")
    axes[1].set_title("False Positive Rate")

    plt.suptitle("LitCVit – TPR & FPR Across Datasets", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(dataset_name: str,
             ckpt_path:    str,
             out_dir:      str  = "results",
             processed_dir:str  = PROCESSED_DIR,
             batch_size:   int  = 64,
             num_workers:  int  = 0):

    ds_key = DATASET_KEY_MAP.get(dataset_name, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"LitCVit  |  Evaluation on {ds_key}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    images, labels, classes = load_dataset(dataset_name, processed_dir)
    splitter = ETFlowSplit(images, labels)
    _, _, te_loader = splitter.get_loaders(batch_size=batch_size,
                                            num_workers=num_workers,
                                            class_balanced=False)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_classes = ckpt.get("n_classes", len(classes))
    model = LitCVit(emb_dim=EMB_DIM, rank=RANK, byte_win=BYTE_WIN,
                    pkt_win=PACKET_WIN, dropout=DROPOUT,
                    n_classes=n_classes, mode="finetune")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("  Running inference on test set…")
    y_pred, y_true, y_probs = predict(model, te_loader)

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, classes)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  ── Overall Metrics ─────────────────────────────")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  (macro)")
    print(f"  Recall    : {metrics['recall']:.4f}  (macro)")
    print(f"  F1-score  : {metrics['f1']:.4f}  (macro)")
    print(f"  TPR       : {metrics['tpr']:.4f}  (macro)")
    print(f"  TNR       : {metrics['tnr']:.4f}  (macro)")
    print(f"  FPR       : {metrics['fpr']:.4f}  (macro)")
    print(f"  FNR       : {metrics['fnr']:.4f}  (macro)")

    # ── Per-class table ────────────────────────────────────────────────────────
    headers = ["Class", "Precision", "Recall", "F1", "TPR", "FPR", "Support"]
    rows = [[c["class"], f"{c['precision']:.4f}", f"{c['recall']:.4f}",
             f"{c['f1']:.4f}", f"{c['tpr']:.4f}", f"{c['fpr']:.4f}",
             c["support"]] for c in metrics["per_class"]]
    print("\n  ── Per-Class Breakdown ─────────────────────────")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ── Save metrics ──────────────────────────────────────────────────────────
    out_json = os.path.join(out_dir, f"{dataset_name}_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {out_json}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm_path = os.path.join(out_dir, f"{dataset_name}_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], classes,
                          f"LitCVit – {ds_key}", cm_path)
    print(f"  Confusion matrix -> {cm_path}")

    pr_path = os.path.join(out_dir, f"{dataset_name}_pr_curves.png")
    plot_pr_curves(y_true, y_probs, classes,
                   f"PR Curves – {ds_key}", pr_path)
    print(f"  PR curves -> {pr_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LitCVit Evaluation")
    parser.add_argument("--dataset",   required=True,
                        choices=["ustc", "5gad", "tiissrc23"])
    parser.add_argument("--ckpt",      required=True)
    parser.add_argument("--out_dir",   default="results")
    parser.add_argument("--processed_dir", default=PROCESSED_DIR)
    parser.add_argument("--batch_size",type=int, default=64)
    args = parser.parse_args()

    evaluate(
        dataset_name  = args.dataset,
        ckpt_path     = args.ckpt,
        out_dir       = args.out_dir,
        processed_dir = args.processed_dir,
        batch_size    = args.batch_size,
    )
