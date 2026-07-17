"""Robustness/evasion diagnostic: perturb held-out data at inference time and
re-evaluate an already-trained model, without retraining.

Given the project's hard time budget this is a single-seed diagnostic, not a
multi-seed statistical claim, and is reported as such. Perturbations are
applied directly to the cached ET-flow image / timing / statistics arrays
(post-hoc feature-space perturbation), which is weaker evidence than
re-generating perturbed raw packets and reprocessing them through the full
PCAP pipeline; this distinction is stated explicitly rather than implied away.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from triguard.metrics import classification_metrics
from triguard.runner import load_cache, make_loader, predict_torch, train_litcvit, train_triguard


def _byte_padding(images: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Zero out a random contiguous tail fraction of each image's rows,
    approximating truncation/padding perturbation of the byte view."""
    out = images.copy()
    h = images.shape[1]
    cut = int(h * (1 - fraction))
    out[:, cut:, :] = 0
    return out


def _byte_insertion(images: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Random-byte insertion: overwrite a random fraction of pixels with
    uniform random byte noise."""
    out = images.copy()
    mask = rng.random(images.shape) < fraction
    out[mask] = rng.integers(0, 256, size=int(mask.sum()), dtype=np.uint8)
    return out


def _timing_jitter(timing: np.ndarray, std_fraction: float, rng: np.random.Generator) -> np.ndarray:
    out = timing.copy()
    scale = (np.abs(out).mean() + 1e-6) * std_fraction
    out = out + rng.normal(0, scale, size=out.shape).astype(out.dtype)
    return out


def _direction_corruption(timing: np.ndarray, fraction: float, rng: np.random.Generator, direction_col: int = 2) -> np.ndarray:
    out = timing.copy()
    mask = rng.random(out.shape[:-1]) < fraction
    out[..., direction_col] = np.where(mask, -out[..., direction_col], out[..., direction_col])
    return out


PERTURBATIONS = {
    "byte_padding_5pct": ("image", lambda x, rng: _byte_padding(x, 0.05, rng)),
    "byte_padding_10pct": ("image", lambda x, rng: _byte_padding(x, 0.10, rng)),
    "byte_padding_15pct": ("image", lambda x, rng: _byte_padding(x, 0.15, rng)),
    "byte_padding_20pct": ("image", lambda x, rng: _byte_padding(x, 0.20, rng)),
    "byte_insertion_10pct": ("image", lambda x, rng: _byte_insertion(x, 0.10, rng)),
    "timing_jitter_20pct": ("timing", lambda x, rng: _timing_jitter(x, 0.20, rng)),
    "timing_jitter_50pct": ("timing", lambda x, rng: _timing_jitter(x, 0.50, rng)),
    "direction_corruption_20pct": ("timing", lambda x, rng: _direction_corruption(x, 0.20, rng)),
    "missing_byte_view": ("mask", 0),
    "missing_timing_view": ("mask", 1),
    "missing_statistics_view": ("mask", 2),
}


def run_robustness(dataset: str, cache: str, output: str, seed: int = 1, epochs: int = 500,
                   batch_size: int = 128, patience: int = 15, log_every: int = 25) -> dict:
    data, metadata = load_cache(cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit, lit_meta = train_litcvit(data, seed, epochs, device, batch_size, None,
                                  log_every=log_every, log_prefix=f"{dataset}:robustness-litcvit:seed{seed}",
                                  patience=patience)
    tri, tri_meta = train_triguard(data, lit, seed, epochs, device, batch_size,
                                   log_every=log_every, log_prefix=f"{dataset}:robustness-triguard:seed{seed}",
                                   patience=patience)

    clean_test = make_loader(data, "test", batch_size, False)
    y0, p0, pr0, gate0 = predict_torch(tri, clean_test, device, "triguard")
    clean_metrics = classification_metrics(y0, p0, pr0, data["classes"])
    results = {"clean": {**clean_metrics, "average_gate_weights": gate0.mean(0).tolist()}}

    rng = np.random.default_rng(seed)
    for name, (kind, spec) in PERTURBATIONS.items():
        perturbed = dict(data)
        if kind == "image":
            perturbed["image"] = spec(data["image"], rng)
        elif kind == "timing":
            perturbed["timing"] = spec(data["timing"], rng)
        elif kind == "mask":
            mask = data["view_mask"].copy(); mask[:, spec] = False
            perturbed["view_mask"] = mask
        loader = make_loader(perturbed, "test", batch_size, False)
        y, p, pr, gate = predict_torch(tri, loader, device, "triguard")
        metrics = classification_metrics(y, p, pr, data["classes"])
        results[name] = {**metrics, "average_gate_weights": gate.mean(0).tolist(),
                         "macro_f1_delta_vs_clean": metrics["macro_f1"] - clean_metrics["macro_f1"]}
        print(f"[robustness {name}] macro_f1={metrics['macro_f1']:.4f} "
             f"(clean={clean_metrics['macro_f1']:.4f}, delta={results[name]['macro_f1_delta_vs_clean']:+.4f})", flush=True)

    record = {"status": "robustness-single-seed-diagnostic", "dataset": dataset, "seed": seed,
              "note": "Single-seed inference-time feature-space perturbation diagnostic, not a multi-seed "
                     "statistical claim and not a re-generated-raw-packet robustness test.",
              "litcvit_meta": {"validation_macro_f1": lit_meta["validation_macro_f1"]},
              "triguard_meta": {"validation_macro_f1": tri_meta["validation_macro_f1"]},
              "results": results}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    run_robustness(args.dataset, args.cache, args.output, args.seed, args.epochs, args.batch_size,
                   args.patience, args.log_every)
