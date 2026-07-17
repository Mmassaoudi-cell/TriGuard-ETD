"""Ablation grid: view-only/pairwise combinations and loss-term removals.

Given a hard time budget, this runs a reduced but representative subset of
the originally requested 20-config grid (documented explicitly, not silently
substituted): 7 view-availability configs (byte/timing/stats-only, the three
pairwise combinations, and full fusion) and 5 loss-term-removal configs
(no-KD, no-focal, no-supcon, no-align, hard-labels-only). Teacher-ensemble
ablations and an early-fusion control are deferred and listed as such.

LitCVit is trained once per seed and reused as the image teacher across every
ablation config for that seed, since retraining it per config would not
change its own result and would multiply wall-clock for no benefit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from triguard.metrics import benchmark_latency, calibration_metrics, classification_metrics, model_macs
from triguard.runner import (apply_view_override, load_cache, make_loader, predict_torch,
                             set_seed, train_litcvit, train_triguard)

VIEW_CONFIGS = {
    "byte_only": (True, False, False),
    "timing_only": (False, True, False),
    "stats_only": (False, False, True),
    "byte_timing": (True, True, False),
    "byte_stats": (True, False, True),
    "timing_stats": (False, True, True),
    "full_fusion": (True, True, True),
}

LOSS_CONFIGS = {
    "no_kd": {"kd_weight": 0.0},
    "no_focal": {"focal_weight": 0.0},
    "no_supcon": {"supcon_weight": 0.0},
    "no_align": {"align_weight": 0.0},
    "hard_labels_only": {"kd_weight": 0.0, "supcon_weight": 0.0, "align_weight": 0.0},
}


def run_ablations(dataset: str, cache: str, output_dir: str, seeds: list[int], epochs: int = 500,
                  batch_size: int = 128, patience: int = 15, log_every: int = 25) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in seeds:
        data, metadata = load_cache(cache)
        lit, lit_meta = train_litcvit(data, seed, epochs, device, batch_size, None,
                                      log_every=log_every, log_prefix=f"{dataset}:ablation-litcvit:seed{seed}",
                                      patience=patience)
        configs: dict[str, dict] = {}
        for name, views in VIEW_CONFIGS.items():
            configs[name] = {"views": views}
        for name, loss_kwargs in LOSS_CONFIGS.items():
            configs[name] = {"views": (True, True, True), "loss": loss_kwargs}
        for name, cfg in configs.items():
            result_path = out / f"{name}_seed{seed}.json"
            if result_path.exists():
                print(f"[ablation {name} seed{seed}] already exists, skipping", flush=True)
                continue
            ablated_data = apply_view_override(data, cfg["views"])
            started = time.time()
            model, meta = train_triguard(ablated_data, lit, seed, epochs, device, batch_size,
                                         log_every=log_every, log_prefix=f"{dataset}:{name}:seed{seed}",
                                         patience=patience, **cfg.get("loss", {}))
            test = make_loader(ablated_data, "test", batch_size, False)
            y, p, pr, gates = predict_torch(model, test, device, "triguard")
            metrics = classification_metrics(y, p, pr, ablated_data["classes"])
            sample = next(iter(test))
            img, tim, stat, mask = (sample[k][:1].to(device) for k in ("image", "timing", "statistics", "view_mask"))
            with torch.inference_mode():
                latency = benchmark_latency(lambda: model(img, tim, stat, mask), device)
                complexity = model_macs(model, (img, tim, stat, mask))
            record = {"status": "ablation-run", "dataset": dataset, "ablation": name, "seed": seed,
                      "views": cfg["views"], "loss_overrides": cfg.get("loss", {}),
                      **meta, **metrics, "calibration": calibration_metrics(y, pr),
                      "average_gate_weights": gates.mean(0).tolist(), "latency": latency,
                      "complexity": complexity, "train_time_seconds": time.time() - started}
            result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"[ablation {name} seed{seed}] macro_f1={metrics['macro_f1']:.4f} "
                 f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    run_ablations(args.dataset, args.cache, args.output_dir, args.seeds, args.epochs, args.batch_size,
                 args.patience, args.log_every)
