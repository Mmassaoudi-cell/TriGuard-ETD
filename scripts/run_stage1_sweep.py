"""Resumable multi-seed driver for the Stage 1 main comparison.

For one frozen dataset manifest, trains the reference LitCVit-192 baseline
and TriGuard-ETD across a fixed list of seeds, skipping any seed whose result
JSON already exists so a killed/restarted background job does not repeat
finished work. The manifest itself (cache + split) is never regenerated here
-- only the model-training seed varies across replications, matching the
"identical manifest for every paired comparison" requirement.

Usage (run from LitCVit_3datasets/):
  python -m scripts.run_stage1_sweep --dataset USTC-TFC2016 \
      --cache artifacts/ustc_full/ustc_multiview.npz \
      --output-dir artifacts/ustc_full/seeds \
      --seeds 1 2 3 4 5 6 7 8 9 10 \
      --pretrain-checkpoint artifacts/ustc_full/litcvit_pretrain.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from triguard.runner import run_experiment
from triguard.statistics import aggregate


def run_sweep(dataset: str, cache: str, output_dir: str, seeds: list[int],
             litcvit_epochs: int = 500, triguard_epochs: int = 500, batch_size: int = 128,
             pretrain_checkpoint: str | None = None, emb_dim: int = 192, rank: int = 16,
             freeze_epochs: int = 100, status: str = "full-protocol-run",
             patience: int | None = 25) -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    log = {"dataset": dataset, "cache": cache, "requested_seeds": seeds, "patience": patience,
          "completed": [], "skipped_existing": [], "failed": []}
    for seed in seeds:
        result_path = out / f"seed{seed}.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("status") == status:
                log["skipped_existing"].append(seed)
                print(f"[seed {seed}] already completed at {result_path}, skipping")
                continue
        started = time.time()
        try:
            run_experiment(dataset, cache, str(result_path), seed=seed, litcvit_epochs=litcvit_epochs,
                           triguard_epochs=triguard_epochs, batch_size=batch_size,
                           pretrain_checkpoint=pretrain_checkpoint, emb_dim=emb_dim, rank=rank,
                           freeze_epochs=freeze_epochs, status=status, patience=patience)
            elapsed = time.time() - started
            log["completed"].append({"seed": seed, "seconds": elapsed})
            print(f"[seed {seed}] done in {elapsed:.1f}s")
        except Exception as exc:  # a failed seed must not silently vanish from the record
            log["failed"].append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[seed {seed}] FAILED: {type(exc).__name__}: {exc}")
    (out / "sweep_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    if len(log["completed"]) + len(log["skipped_existing"]) >= 2:
        report = aggregate(str(out), str(out / "aggregate.json"), allowed_status=(status,))
        print(json.dumps({"aggregated_seeds": report["n_seeds"]}, indent=2))
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--litcvit-epochs", type=int, default=500)
    parser.add_argument("--triguard-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-checkpoint", default=None)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--freeze-epochs", type=int, default=100)
    parser.add_argument("--status", default="full-protocol-run")
    parser.add_argument("--patience", type=int, default=25, help="0 disables early stopping and restores the fixed schedule")
    args = parser.parse_args()
    run_sweep(args.dataset, args.cache, args.output_dir, args.seeds, args.litcvit_epochs, args.triguard_epochs,
             args.batch_size, args.pretrain_checkpoint, args.emb_dim, args.rank, args.freeze_epochs, args.status,
             args.patience or None)
