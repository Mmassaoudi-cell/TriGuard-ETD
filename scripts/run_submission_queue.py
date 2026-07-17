"""Long-running queue for submission-blocking TriGuard experiments.

This script intentionally runs experiments sequentially so the single local GPU
is not oversubscribed.  It assumes preprocessing artifacts already exist and
does not regenerate manifests.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def wait_for(path: Path, label: str, poll_seconds: int = 120) -> None:
    while not path.exists():
        print(f"[queue] waiting for {label}: {path}", flush=True)
        time.sleep(poll_seconds)
    print(f"[queue] found {label}: {path}", flush=True)


def run(cmd: list[str], cwd: Path) -> None:
    print("[queue] running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--pretrain-checkpoint", default="artifacts/ustc_full/litcvit_pretrain_192_bs64.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    py = sys.executable

    wait_for(root / args.pretrain_checkpoint, "USTC train-only LitCVit-192 pretraining checkpoint")

    # Shorter post-pretraining run first: it produces a complete, explicitly
    # sampled TII diagnostic before the much longer USTC repeated-seed sweep.
    run([py, "-m", "scripts.run_stage1_sweep",
         "--dataset", "TII-SSRC-23-stratified-sample",
         "--cache", "artifacts/tii_stratified_sample/tii_sample_multiview.npz",
         "--output-dir", "artifacts/tii_stratified_sample/seeds",
         "--seeds", *map(str, args.seeds),
         "--litcvit-epochs", "500", "--triguard-epochs", "500",
         "--batch-size", "128", "--emb-dim", "192", "--rank", "16",
         "--status", "stratified-sample-full-protocol-run",
         "--patience", "25"], root)

    run([py, "-m", "scripts.run_stage1_sweep",
         "--dataset", "USTC-TFC2016",
         "--cache", "artifacts/ustc_full/ustc_multiview.npz",
         "--output-dir", "artifacts/ustc_full/seeds_pretrained_bs64",
         "--seeds", *map(str, args.seeds),
         "--pretrain-checkpoint", args.pretrain_checkpoint,
         "--litcvit-epochs", "500", "--triguard-epochs", "500",
         "--batch-size", "128", "--emb-dim", "192", "--rank", "16",
         "--freeze-epochs", "100",
         "--status", "pretrained-full-protocol-run",
         "--patience", "25"], root)


if __name__ == "__main__":
    main()
