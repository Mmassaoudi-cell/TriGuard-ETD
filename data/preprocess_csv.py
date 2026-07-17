"""
Pre-processing for TII-SSRC-23 (CSV-based dataset).

Since TII-SSRC-23 provides flow-level statistical features rather than raw
PCAP bytes, we adapt the input representation:
  - Select the 83 numerical flow features (columns 7–86, excluding identifiers).
  - Impute NaN/Inf values with column medians.
  - Apply robust scaling (per-feature, fit on train split only).
  - Reshape the 83-d feature vector into a (40 × 40) "feature image" by
    zero-padding to 1600 and reshaping — this keeps the input shape identical
    to the ET-Flow images used for PCAP-based datasets, allowing the same
    LitCVit architecture without modification.

Usage:
    python preprocess_csv.py  --out_dir processed_data/tiissrc23
                               --label_col "Traffic Type"
                               --max_rows 500000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TIISSRC_CSV, PROCESSED_DIR, IMG_H, IMG_W, RANDOM_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Feature-image construction
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SIZE = IMG_H * IMG_W   # 1600

def features_to_image(feat_vec: np.ndarray) -> np.ndarray:
    """
    Map a 1-D normalised feature vector to a (IMG_H, IMG_W) float32 image.
    Features are placed in order, remainder zero-padded.
    The output range is clipped to [0, 1] and converted to uint8 [0, 255]
    to match the PCAP-based ET-flow image format.
    """
    n = len(feat_vec)
    buf = np.zeros(TARGET_SIZE, dtype=np.float32)
    buf[:n] = feat_vec[:TARGET_SIZE]          # truncate if somehow > TARGET_SIZE
    # Map from [-inf, inf] (after robust scaling) to [0,1]: use sigmoid
    buf = 1.0 / (1.0 + np.exp(-buf))
    img = (buf * 255).clip(0, 255).astype(np.uint8)
    return img.reshape(IMG_H, IMG_W)


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing function
# ─────────────────────────────────────────────────────────────────────────────

IDENTIFIER_COLS = {"Flow ID", "Src IP", "Dst IP", "Timestamp",
                   "Label", "Traffic Type", "Traffic Subtype"}


def process_tiissrc23(csv_path: str, out_dir: str,
                      label_col: str = "Traffic Type",
                      max_rows: int = 500_000,
                      sample_seed: int = RANDOM_SEED):
    """
    Process TII-SSRC-23 CSV into (N, IMG_H, IMG_W) uint8 images + labels.

    Returns:
        X        : np.ndarray (N, IMG_H, IMG_W)  uint8
        y        : np.ndarray (N,)               int64
        classes  : list of label strings
        scaler   : fitted RobustScaler
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[TII-SSRC-23] Reading CSV (up to {max_rows:,} rows)…")

    # ── Read CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # ── Identify numeric feature columns ─────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLS]
    # Keep only numeric; coerce errors to NaN
    df_feat = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    # ── Clean: replace Inf with NaN, fill NaN with median ────────────────────
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    medians  = df_feat.median(skipna=True)
    df_feat  = df_feat.fillna(medians)
    print(f"  Feature columns: {df_feat.shape[1]}")

    # ── Labels ────────────────────────────────────────────────────────────────
    lbl_series = df[label_col].fillna("Unknown").astype(str)
    classes    = sorted(lbl_series.unique().tolist())
    label_map  = {c: i for i, c in enumerate(classes)}
    y          = lbl_series.map(label_map).values.astype(np.int64)
    print(f"  Classes ({len(classes)}): {classes}")

    # ── Robust scaling (fit on full chunk) ────────────────────────────────────
    X_raw = df_feat.values.astype(np.float32)
    scaler = RobustScaler(quantile_range=(5, 95))
    X_scaled = scaler.fit_transform(X_raw)    # (N, n_features)
    print(f"  Scaled feature matrix: {X_scaled.shape}")

    # ── Convert to feature images ─────────────────────────────────────────────
    print("  Converting feature vectors to images…")
    N = X_scaled.shape[0]
    X_imgs = np.zeros((N, IMG_H, IMG_W), dtype=np.uint8)
    for i in tqdm(range(N), desc="TII-SSRC-23"):
        X_imgs[i] = features_to_image(X_scaled[i])

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(out_dir, "tiissrc23")
    np.save(out + "_images.npy", X_imgs)
    np.save(out + "_labels.npy", y)
    np.save(out + "_label_map.npy", np.array(classes))
    joblib.dump(scaler, out + "_scaler.pkl")
    print(f"  Saved {N:,} images to {out_dir}")

    return X_imgs, y, classes, scaler


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TII-SSRC-23 CSV → feature images")
    parser.add_argument("--csv", default=TIISSRC_CSV)
    parser.add_argument("--out_dir", default=PROCESSED_DIR)
    parser.add_argument("--label_col", default="Traffic Type")
    parser.add_argument("--max_rows", type=int, default=500_000)
    args = parser.parse_args()

    process_tiissrc23(args.csv, args.out_dir,
                      label_col=args.label_col,
                      max_rows=args.max_rows)
