"""Metrics, reliability summaries, and reproducible efficiency measurement."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray,
                           classes: list[str]) -> dict:
    labels = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = []
    ap_values, roc_values = [], []
    for i, name in enumerate(classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        binary = (y_true == i).astype(int)
        ap = float(average_precision_score(binary, probabilities[:, i])) if binary.min() != binary.max() else float("nan")
        auc = float(roc_auc_score(binary, probabilities[:, i])) if binary.min() != binary.max() else float("nan")
        ap_values.append(ap); roc_values.append(auc)
        per_class.append({"class": name, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i]), "tpr": float(tp / max(tp + fn, 1)), "tnr": float(tn / max(tn + fp, 1)),
                          "fpr": float(fp / max(fp + tn, 1)), "fnr": float(fn / max(fn + tp, 1)), "pr_auc": ap, "roc_auc": auc})
    absent_classes = [classes[i] for i, support in enumerate(s) if int(support) == 0]
    # Always pass the registered label list for macro averages.  Without this,
    # scikit-learn drops labels absent from y_true/y_pred, which can turn a
    # split with train-only classes into an over-optimistic macro score.
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "tpr": float(np.mean([x["tpr"] for x in per_class])), "tnr": float(np.mean([x["tnr"] for x in per_class])),
            "fpr": float(np.mean([x["fpr"] for x in per_class])), "fnr": float(np.mean([x["fnr"] for x in per_class])),
            "pr_auc": float(np.nanmean(ap_values)), "roc_auc": float(np.nanmean(roc_values)),
            "absent_test_classes": absent_classes,
            "evaluated_label_count": int(len(classes)),
            "present_test_label_count": int(np.count_nonzero(s)),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> dict:
    """Top-label calibration: ECE, Brier score, NLL, and a reliability diagram.

    Uses the predicted class's own confidence (max probability) against
    whether that prediction was correct, the standard top-label ECE
    definition; per-class calibration is not attempted separately here
    because several classes have too few test samples to bin reliably.
    """
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)
    n_classes = probabilities.shape[1]
    true_probs = np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    nll = float(-np.mean(np.log(true_probs)))
    one_hot = np.zeros_like(probabilities); one_hot[np.arange(len(y_true)), y_true] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)
    reliability = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            reliability.append({"bin_low": float(bin_edges[b]), "bin_high": float(bin_edges[b + 1]),
                               "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        ece += (count / len(y_true)) * abs(acc - mean_conf)
        reliability.append({"bin_low": float(bin_edges[b]), "bin_high": float(bin_edges[b + 1]),
                           "count": count, "mean_confidence": mean_conf, "accuracy": acc})
    return {"ece": float(ece), "brier_score": brier, "nll": nll, "n_bins": n_bins,
            "n_classes": int(n_classes), "reliability_diagram": reliability}


def operating_point_at_fpr(y_true: np.ndarray, probabilities: np.ndarray, malicious_class_ids: list[int],
                           target_fprs: tuple[float, ...] = (0.001, 0.01, 0.05)) -> dict:
    """Attack recall at fixed validation-selected FPR targets, collapsing the
    problem to benign-vs-malicious using the summed malicious-class probability
    as the decision score. Thresholds must be chosen on validation data only;
    the caller is responsible for passing validation (not test) arrays when
    selecting a threshold, and test arrays only to report the resulting recall.
    """
    is_malicious = np.isin(y_true, malicious_class_ids).astype(int)
    if is_malicious.min() == is_malicious.max():
        return {"status": "single-class split; benign-vs-malicious operating points undefined"}
    score = probabilities[:, malicious_class_ids].sum(axis=1)
    order = np.argsort(-score)
    sorted_scores = score[order]
    sorted_labels = is_malicious[order]
    n_benign = int((is_malicious == 0).sum())
    results = {}
    for target in target_fprs:
        max_fp = target * n_benign
        fp = 0; tp = 0; threshold = 1.0
        for s, lab in zip(sorted_scores, sorted_labels):
            if lab == 0:
                fp += 1
                if fp > max_fp:
                    break
            else:
                tp += 1
            threshold = s
        n_malicious = int(is_malicious.sum())
        results[f"fpr_{target}"] = {"threshold": float(threshold), "achieved_fpr": float(fp / max(n_benign, 1)),
                                    "recall": float(tp / max(n_malicious, 1)), "alerts": int(fp + tp)}
    return results


def summarize_seeds(rows: list[dict], key: str = "macro_f1") -> dict:
    values = np.asarray([x[key] for x in rows], dtype=float)
    n = len(values)
    if n < 2:
        return {"mean": float(values.mean()), "std": 0.0, "ci95_low": float(values.mean()), "ci95_high": float(values.mean()), "n": n}
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "ci95_low": float(values.mean() - half), "ci95_high": float(values.mean() + half), "n": n}


def benchmark_latency(forward: Callable[[], object], device: str, warmup: int = 50, repeats: int = 300) -> dict:
    """Batch-one latency with CUDA synchronization and an explicit CPU mode."""
    for _ in range(warmup):
        forward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        forward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    arr = np.asarray(timings)
    return {"latency_ms_mean": float(arr.mean()), "latency_ms_std": float(arr.std(ddof=1)),
            "latency_ms_p95": float(np.percentile(arr, 95)), "throughput_samples_s": float(1000 / arr.mean())}


def model_memory_mb(model: torch.nn.Module) -> float:
    return float(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2))


def model_macs(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict:
    """Best-effort MAC/FLOP accounting. Unsupported custom operations are noted."""
    try:
        from thop import profile
        macs, params = profile(model, inputs=inputs, verbose=False)
        return {"macs": float(macs), "flops_approx": float(2 * macs), "profile_parameters": int(params), "status": "estimated with thop"}
    except Exception as exc:  # profile tools should never prevent evaluation
        return {"macs": None, "flops_approx": None, "status": f"unavailable: {type(exc).__name__}"}
