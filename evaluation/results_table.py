"""
Generate the final results table (analogous to Table III of the paper) and
summary figures after all three datasets have been evaluated.

Produces:
  results/
    summary_table.txt        – Table III style performance comparison
    summary_table.csv        – machine-readable version
    tpr_fpr_comparison.png   – Fig. 6 style bar chart (LitCVit only)
    accuracy_f1_radar.png    – radar chart across 3 datasets

Usage:
    python evaluation/results_table.py --results_dir results/
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DATASET_DISPLAY = {
    "ustc":      "USTC-TFC2016",
    "5gad":      "5GAD-2022",
    "tiissrc23": "TII-SSRC-23",
}


def load_metrics(results_dir: str, dataset_keys: list) -> dict:
    all_metrics = {}
    for key in dataset_keys:
        path = os.path.join(results_dir, f"{key}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[key] = json.load(f)
        else:
            print(f"[WARNING] {path} not found – skipping {key}")
    return all_metrics


def build_summary_table(all_metrics: dict) -> str:
    """
    Table analogous to Table III, reporting per-dataset:
    Accuracy, Precision, Recall, F1, TPR, TNR, FPR, FNR
    """
    rows = []
    for key, m in all_metrics.items():
        rows.append([
            DATASET_DISPLAY.get(key, key),
            f"{m['accuracy']  * 100:.2f}%",
            f"{m['precision'] * 100:.2f}%",
            f"{m['recall']    * 100:.2f}%",
            f"{m['f1']        * 100:.2f}%",
            f"{m['tpr']       * 100:.2f}%",
            f"{m['tnr']       * 100:.2f}%",
            f"{m['fpr']       * 100:.4f}",
            f"{m['fnr']       * 100:.2f}%",
        ])

    # Average row
    if len(all_metrics) > 1:
        keys = list(all_metrics)
        avg  = {
            "accuracy":  np.mean([all_metrics[k]["accuracy"]  for k in keys]),
            "precision": np.mean([all_metrics[k]["precision"] for k in keys]),
            "recall":    np.mean([all_metrics[k]["recall"]    for k in keys]),
            "f1":        np.mean([all_metrics[k]["f1"]        for k in keys]),
            "tpr":       np.mean([all_metrics[k]["tpr"]       for k in keys]),
            "tnr":       np.mean([all_metrics[k]["tnr"]       for k in keys]),
            "fpr":       np.mean([all_metrics[k]["fpr"]       for k in keys]),
            "fnr":       np.mean([all_metrics[k]["fnr"]       for k in keys]),
        }
        rows.append([
            "Average",
            f"{avg['accuracy']  * 100:.2f}%",
            f"{avg['precision'] * 100:.2f}%",
            f"{avg['recall']    * 100:.2f}%",
            f"{avg['f1']        * 100:.2f}%",
            f"{avg['tpr']       * 100:.2f}%",
            f"{avg['tnr']       * 100:.2f}%",
            f"{avg['fpr']       * 100:.4f}",
            f"{avg['fnr']       * 100:.2f}%",
        ])

    headers = ["Dataset", "Accuracy", "Precision", "Recall", "F1",
               "TPR", "TNR", "FPR", "FNR"]
    return tabulate(rows, headers=headers, tablefmt="grid")


def build_csv_table(all_metrics: dict, out_path: str):
    lines = ["Dataset,Accuracy,Precision,Recall,F1,TPR,TNR,FPR,FNR"]
    for key, m in all_metrics.items():
        lines.append(
            f"{DATASET_DISPLAY.get(key, key)},"
            f"{m['accuracy']:.4f},{m['precision']:.4f},"
            f"{m['recall']:.4f},{m['f1']:.4f},"
            f"{m['tpr']:.4f},{m['tnr']:.4f},"
            f"{m['fpr']:.6f},{m['fnr']:.4f}"
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def plot_tpr_fpr(all_metrics: dict, save_path: str):
    datasets  = [DATASET_DISPLAY.get(k, k) for k in all_metrics]
    tprs = [all_metrics[k]["tpr"] * 100 for k in all_metrics]
    fprs = [all_metrics[k]["fpr"] * 100 for k in all_metrics]

    x    = np.arange(len(datasets))
    w    = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(x, tprs, width=w, color="#4C72B0", label="LitCVit")
    ax1.set_xticks(x); ax1.set_xticklabels(datasets, rotation=15, ha="right")
    ax1.set_ylim(0, 105); ax1.set_ylabel("TPR (%)")
    ax1.set_title("True Positive Rate (TPR)")
    for i, v in enumerate(tprs):
        ax1.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=9)

    ax2.bar(x, fprs, width=w, color="#DD8452", label="LitCVit")
    ax2.set_xticks(x); ax2.set_xticklabels(datasets, rotation=15, ha="right")
    ax2.set_ylim(0, max(max(fprs) * 1.5, 2.0))
    ax2.set_ylabel("FPR (%)")
    ax2.set_title("False Positive Rate (FPR)")
    for i, v in enumerate(fprs):
        ax2.text(i, v + 0.02, f"{v:.3f}%", ha="center", fontsize=9)

    plt.suptitle("LitCVit — TPR & FPR on 3 Datasets", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar(all_metrics: dict, save_path: str):
    """Radar chart showing all 4 metrics per dataset."""
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_metrics = len(metrics_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for (key, m), color in zip(all_metrics.items(), colors):
        values = [m[k] * 100 for k in metrics_keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, color=color,
                label=DATASET_DISPLAY.get(key, key))
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("LitCVit Performance Across 3 Datasets", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(all_metrics: dict, save_path: str):
    """Bar chart of per-class F1 for each dataset."""
    n_datasets = len(all_metrics)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]

    for ax, (key, m) in zip(axes, all_metrics.items()):
        per_cls = m["per_class"]
        names   = [c["class"][:15] for c in per_cls]
        f1s     = [c["f1"] * 100 for c in per_cls]
        y_pos   = np.arange(len(names))
        ax.barh(y_pos, f1s, color="#4C72B0", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 110)
        ax.set_xlabel("F1-Score (%)")
        ax.set_title(DATASET_DISPLAY.get(key, key))
        for i, v in enumerate(f1s):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=7)

    plt.suptitle("LitCVit — Per-Class F1 Scores", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_results(results_dir: str):
    datasets = ["ustc", "5gad", "tiissrc23"]
    all_metrics = load_metrics(results_dir, datasets)

    if not all_metrics:
        print("[ERROR] No metrics files found. Run evaluation scripts first.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    table_str = build_summary_table(all_metrics)
    print("\n" + "=" * 70)
    print("  LitCVit — Performance Summary (3 Datasets)")
    print("=" * 70)
    print(table_str)

    out_txt = os.path.join(results_dir, "summary_table.txt")
    with open(out_txt, "w") as f:
        f.write("LitCVit — Performance on 3 Datasets\n")
        f.write("(Reproduction of Table III structure from the original paper)\n\n")
        f.write(table_str + "\n")
    print(f"\n  Table saved to: {out_txt}")

    out_csv = os.path.join(results_dir, "summary_table.csv")
    build_csv_table(all_metrics, out_csv)
    print(f"  CSV saved to:   {out_csv}")

    # ── Figures ────────────────────────────────────────────────────────────────
    plot_tpr_fpr(all_metrics, os.path.join(results_dir, "tpr_fpr_comparison.png"))
    print(f"  TPR/FPR plot  -> {results_dir}/tpr_fpr_comparison.png")

    plot_radar(all_metrics, os.path.join(results_dir, "accuracy_f1_radar.png"))
    print(f"  Radar chart   -> {results_dir}/accuracy_f1_radar.png")

    plot_per_class_f1(all_metrics, os.path.join(results_dir, "per_class_f1.png"))
    print(f"  Per-class F1  -> {results_dir}/per_class_f1.png")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results table & figures")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    generate_results(args.results_dir)
