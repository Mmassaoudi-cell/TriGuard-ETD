"""
LitCVit — Full Pipeline Entry Point
====================================
Reproduces the LitCVit results on 3 datasets (USTC-TFC2016, 5GAD-2022,
TII-SSRC-23) following the original paper's experimental protocol.

Pipeline:
  Step 1 — Preprocess raw data (PCAP → ET-flow images; CSV → feature images)
  Step 2 — Self-supervised pre-training on USTC-TFC2016
  Step 3 — Fine-tune on each of the 3 datasets
  Step 4 — Evaluate and generate results tables + figures

Usage:
    # Full pipeline (all steps)
    python main.py --steps all

    # Individual steps
    python main.py --steps preprocess
    python main.py --steps pretrain
    python main.py --steps finetune  --datasets ustc 5gad tiissrc23
    python main.py --steps evaluate  --datasets ustc 5gad tiissrc23
    python main.py --steps report

Options:
    --processed_dir   Directory for preprocessed .npy files (default: processed_data/)
    --ckpt_dir        Directory for model checkpoints (default: checkpoints/)
    --results_dir     Directory for evaluation outputs (default: results/)
    --max_flows       Max flows per class during preprocessing (default: 3000)
    --pretrain_epochs Number of pre-training epochs (default: 200)
    --finetune_epochs Number of fine-tuning epochs (default: 500)
    --batch_size      Batch size (default: 64)
    --num_workers     DataLoader workers (default: 0)
    --quick           Quick mode: reduced epochs and flows for testing
"""

import os
import sys
import argparse
from pathlib import Path

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR, DEVICE


def step_preprocess(args):
    """Step 1: Convert raw data to ET-flow / feature images."""
    print("\n" + "=" * 60)
    print("STEP 1 — Data Preprocessing")
    print("=" * 60)

    # PCAP-based datasets
    try:
        from scapy.all import PcapReader   # noqa: F401
        scapy_ok = True
    except ImportError:
        scapy_ok = False
        print("[WARNING] scapy not installed. PCAP preprocessing will be skipped.")
        print("  Install with: pip install scapy")

    if scapy_ok:
        from data.preprocess_pcap import process_ustc, process_5gad
        process_ustc(args.processed_dir, max_flows_per_class=args.max_flows)
        process_5gad(args.processed_dir, max_flows_per_class=args.max_flows)
    else:
        print("  Skipping USTC-TFC2016 and 5GAD-2022 PCAP preprocessing.")

    # CSV-based dataset
    from data.preprocess_csv import process_tiissrc23
    from config import TIISSRC_CSV, DATASET_CONFIGS
    process_tiissrc23(
        csv_path  = TIISSRC_CSV,
        out_dir   = args.processed_dir,
        label_col = DATASET_CONFIGS["TII-SSRC-23"]["label_col"],
        max_rows  = DATASET_CONFIGS["TII-SSRC-23"]["max_rows"],
    )


def step_pretrain(args):
    """Step 2: Self-supervised pre-training on USTC-TFC2016."""
    print("\n" + "=" * 60)
    print("STEP 2 — Self-Supervised Pre-training")
    print("=" * 60)

    from training.pretrain import pretrain
    ckpt_path = pretrain(
        out_dir       = os.path.join(args.ckpt_dir, "pretrain"),
        processed_dir = args.processed_dir,
        epochs        = args.pretrain_epochs,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
    )
    return ckpt_path


def step_finetune(args, datasets, pretrain_ckpt=None):
    """Step 3: Fine-tune on each dataset."""
    print("\n" + "=" * 60)
    print("STEP 3 — Fine-tuning")
    print("=" * 60)

    from training.finetune import finetune
    ckpt_paths = {}

    # Auto-detect pretrain checkpoint if not passed
    if pretrain_ckpt is None:
        candidate = os.path.join(args.ckpt_dir, "pretrain",
                                 "litcvit_pretrained_best.pt")
        if os.path.exists(candidate):
            pretrain_ckpt = candidate
            print(f"  Using pre-training checkpoint: {pretrain_ckpt}")
        else:
            print("  [INFO] No pre-training checkpoint found; training from scratch.")

    for ds in datasets:
        print(f"\n  ── Dataset: {ds} ──")
        ckpt = finetune(
            dataset_name  = ds,
            ckpt_path     = pretrain_ckpt,
            out_dir       = args.ckpt_dir,
            processed_dir = args.processed_dir,
            epochs        = args.finetune_epochs,
            batch_size    = args.batch_size,
            num_workers   = args.num_workers,
        )
        ckpt_paths[ds] = ckpt

    return ckpt_paths


def step_evaluate(args, datasets, ckpt_paths=None):
    """Step 4: Evaluate each fine-tuned model."""
    print("\n" + "=" * 60)
    print("STEP 4 — Evaluation")
    print("=" * 60)

    from evaluation.evaluate import evaluate
    os.makedirs(args.results_dir, exist_ok=True)

    for ds in datasets:
        print(f"\n  ── Dataset: {ds} ──")
        # Find checkpoint
        if ckpt_paths and ds in ckpt_paths:
            ckpt = ckpt_paths[ds]
        else:
            ckpt = os.path.join(args.ckpt_dir, f"litcvit_{ds}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  [WARNING] Checkpoint not found: {ckpt}")
            continue
        evaluate(
            dataset_name  = ds,
            ckpt_path     = ckpt,
            out_dir       = args.results_dir,
            processed_dir = args.processed_dir,
            batch_size    = args.batch_size,
        )


def step_report(args):
    """Step 5: Generate summary table and figures."""
    print("\n" + "=" * 60)
    print("STEP 5 — Results Report")
    print("=" * 60)

    from evaluation.results_table import generate_results
    generate_results(args.results_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LitCVit – Reproduction Pipeline (3 Datasets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps", nargs="+",
        choices=["all", "preprocess", "pretrain", "finetune", "evaluate", "report"],
        default=["all"],
        help="Which steps to run (default: all)"
    )
    parser.add_argument("--datasets", nargs="+",
                        choices=["ustc", "5gad", "tiissrc23"],
                        default=["ustc", "5gad", "tiissrc23"])
    parser.add_argument("--processed_dir", default=PROCESSED_DIR)
    parser.add_argument("--ckpt_dir",      default=str(ROOT / "checkpoints"))
    parser.add_argument("--results_dir",   default=str(ROOT / "results"))
    parser.add_argument("--max_flows",     type=int, default=3000)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--finetune_epochs", type=int, default=500)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_workers",   type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 10 pretrain / 50 finetune epochs, 200 flows")
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.pretrain_epochs = 10
        args.finetune_epochs = 50
        args.max_flows       = 200
        print("[QUICK MODE] epochs: 10 pretrain / 50 finetune, max_flows: 200")

    run_all = "all" in args.steps
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LitCVit Reproduction — 3 Datasets")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"{'='*60}")

    pretrain_ckpt = None
    ckpt_paths    = None

    if run_all or "preprocess" in args.steps:
        step_preprocess(args)

    if run_all or "pretrain" in args.steps:
        pretrain_ckpt = step_pretrain(args)

    if run_all or "finetune" in args.steps:
        ckpt_paths = step_finetune(args, args.datasets, pretrain_ckpt)

    if run_all or "evaluate" in args.steps:
        step_evaluate(args, args.datasets, ckpt_paths)

    if run_all or "report" in args.steps:
        step_report(args)

    print("\n[Done] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
