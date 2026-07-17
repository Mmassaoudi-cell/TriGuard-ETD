"""Create a leakage-safe TII-SSRC-23 multi-view cache from CSV features."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _labels_to_ids, create_split_bundle, fit_train_only_features

IDENTIFIERS = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Traffic Type", "Traffic Subtype"}


def _timing_proxy(features: np.ndarray, columns: list[str]) -> np.ndarray:
    """Build a 5-step sequence from *available aggregated CSV values*.

    This is explicitly a timing/statistics proxy, not recovered packet timing.
    It gives the temporal branch a consistent input shape while keeping the
    raw CSV limitation visible in metadata.
    """
    wanted = ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Flow Duration"]
    indices = [columns.index(c) for c in wanted if c in columns]
    values = features[:, indices] if indices else np.zeros((len(features), 1), np.float32)
    out = np.zeros((len(features), 5, 5), dtype=np.float32)
    for t in range(5):
        out[:, t, :values.shape[1]] = values
        out[:, t, -1] = t / 4.0
    return out


def _read_full_csv(csv_path: str, chunksize: int = 250_000, max_rows: int | None = None
                   ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream the complete TII-SSRC-23 CSV in bounded-memory chunks.

    Unlike a `nrows=` prefix read, this walks the entire file (default) so
    every row contributes to class coverage and the split, not just the first
    rows written to disk. `max_rows` is only for an explicit, labeled smoke
    run; it is never used for a final cache.
    """
    feature_columns: list[str] | None = None
    raw_parts: list[np.ndarray] = []
    flow_id_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    subtype_parts: list[np.ndarray] = []
    rows_read = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if "Traffic Type" not in chunk.columns:
            raise ValueError("TII CSV has no 'Traffic Type' label column")
        if feature_columns is None:
            feature_columns = [c for c in chunk.columns if c not in IDENTIFIERS]
        raw = chunk[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        raw[~np.isfinite(raw)] = np.nan
        raw_parts.append(raw)
        flow_id_parts.append(chunk["Flow ID"].fillna("").astype(str).to_numpy())
        timestamp_parts.append(chunk["Timestamp"].fillna("").astype(str).to_numpy() if "Timestamp" in chunk else np.full(len(chunk), "", dtype=object))
        type_parts.append(chunk["Traffic Type"].fillna("Unknown").astype(str).to_numpy())
        subtype_parts.append(chunk["Traffic Subtype"].fillna("Unknown").astype(str).to_numpy() if "Traffic Subtype" in chunk else np.full(len(chunk), "Unknown", dtype=object))
        rows_read += len(chunk)
        if max_rows and rows_read >= max_rows:
            break
    raw = np.concatenate(raw_parts, axis=0)
    flow_ids = np.concatenate(flow_id_parts, axis=0)
    timestamps = np.concatenate(timestamp_parts, axis=0)
    traffic_types = np.concatenate(type_parts, axis=0)
    traffic_subtypes = np.concatenate(subtype_parts, axis=0)
    if max_rows:
        raw, flow_ids, timestamps, traffic_types, traffic_subtypes = (
            a[:max_rows] for a in (raw, flow_ids, timestamps, traffic_types, traffic_subtypes))
    return raw, feature_columns, flow_ids, timestamps, traffic_types, traffic_subtypes


def _flow_id_label_conflicts(flow_ids: np.ndarray, labels: np.ndarray) -> dict:
    """Count Flow IDs that co-occur with more than one distinct Traffic Type."""
    by_flow: dict[str, set[str]] = defaultdict(set)
    for flow, label in zip(flow_ids, labels):
        if flow:
            by_flow[flow].add(label)
    conflicting = {flow: sorted(lbls) for flow, lbls in by_flow.items() if len(lbls) > 1}
    conflicting_rows = int(sum(1 for f in flow_ids if f in conflicting))
    return {
        "distinct_nonempty_flow_ids": len(by_flow),
        "flow_ids_with_conflicting_labels": len(conflicting),
        "rows_touched_by_conflicting_flow_ids": conflicting_rows,
        "example_conflicts": dict(list(conflicting.items())[:5]),
    }


def _whole_dataset_duplicate_audit(raw: np.ndarray) -> dict:
    """Exact-duplicate feature-vector count across the complete dataset (pre-split).

    Uses a vectorized structured-array `np.unique` instead of per-row hashing
    so it stays fast and memory-bounded on multi-million-row input.
    """
    clean = np.nan_to_num(raw.astype(np.float32), nan=0.0,
                          posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    clean = np.ascontiguousarray(clean)
    view = clean.view([(f"f{i}", clean.dtype) for i in range(clean.shape[1])])
    _, counts = np.unique(view, return_counts=True)
    duplicated_groups = counts[counts > 1]
    return {
        "unique_feature_vectors": int(len(counts)),
        "exact_duplicate_groups": int(len(duplicated_groups)),
        "rows_beyond_first_occurrence": int((duplicated_groups - 1).sum()) if len(duplicated_groups) else 0,
    }


def prepare_tii(csv_path: str, output_dir: str, max_rows: int | None = None, seed: int = 42,
                chunksize: int = 250_000) -> None:
    """Build a split-safe TII-SSRC-23 cache from the complete CSV.

    `max_rows` defaults to None, which reads the entire file rather than a
    prefix. Passing an explicit `max_rows` produces a cache whose metadata is
    labeled as a bounded smoke run, not a final dataset.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw, feature_columns, flow_ids, timestamps, traffic_types, traffic_subtypes = _read_full_csv(
        csv_path, chunksize=chunksize, max_rows=max_rows)
    labels, classes = _labels_to_ids(traffic_types)
    flow_id_audit = _flow_id_label_conflicts(flow_ids, traffic_types)
    duplicate_audit = _whole_dataset_duplicate_audit(raw)
    subtype_counts = dict(sorted(Counter(traffic_subtypes.tolist()).items(), key=lambda kv: -kv[1]))
    # The provided Flow ID is not globally label-consistent (see flow_id_audit
    # above). A composite key is the safest available session proxy; missing
    # pairs deliberately become a unique row key rather than being silently
    # pooled across a split boundary.
    groups = np.asarray([f"{flow}|{stamp}" if flow and stamp else f"row-{i}"
                         for i, (flow, stamp) in enumerate(zip(flow_ids, timestamps))])
    split = create_split_bundle(labels, groups, raw, classes, seed=seed)
    scaled, preprocessing = fit_train_only_features(raw, split)
    # Materializing a 40x40 zero-padded image for all 8,656,767 rows needs
    # ~51.6 GiB (confirmed by an out-of-memory crash on this exact call) and
    # is unnecessary: the runner computes the compatibility image lazily, once
    # per batch, from these same scaled statistics (see MultiViewDataset /
    # lazy_image_collate in triguard/data.py). Only a placeholder is stored
    # here so the cache format stays uniform with USTC/5GAD.
    images = np.zeros((1, 40, 40), dtype=np.uint8)
    timing = _timing_proxy(scaled, feature_columns)
    mask = np.zeros((len(labels), 3), dtype=np.bool_)
    mask[:, 0] = True  # compatibility feature image
    mask[:, 1] = True  # aggregate timing proxy
    mask[:, 2] = True  # direct statistics
    np.savez_compressed(out / "tiissrc23_multiview.npz", image=images, timing=timing,
                        statistics=scaled, labels=labels, groups=groups, view_mask=mask,
                        classes=np.asarray(classes, dtype=object), feature_columns=np.asarray(feature_columns, dtype=object))
    split.save(out / "tiissrc23_split", "TII-SSRC-23", seed)
    metadata = {
        "dataset": "TII-SSRC-23",
        "source": str(Path(csv_path).resolve()),
        "rows": int(len(labels)),
        "read_mode": "complete CSV file (chunked)" if max_rows is None else f"bounded smoke read, max_rows={max_rows}",
        "class_counts": {classes[i]: int((labels == i).sum()) for i in range(len(classes))},
        "traffic_subtype_counts": subtype_counts,
        "n_traffic_subtypes": len(subtype_counts),
        "flow_id_conflict_audit": flow_id_audit,
        "whole_dataset_duplicate_audit": duplicate_audit,
        "raw_pcap_used": False,
        "timing_view": "aggregate timing proxy constructed from CSV IAT/duration columns; not packet sequences",
        "image_lazy": True,
        "image_note": ("compatibility image is computed on-the-fly per batch from scaled statistics, not "
                       "precomputed for all rows (51.6 GiB would be required to materialize it up front)"),
        "preprocessing": preprocessing,
        "feature_columns": feature_columns,
        "grouping": ("Flow ID + Timestamp because Flow ID alone was observed with conflicting labels "
                    f"({flow_id_audit['flow_ids_with_conflicting_labels']} of "
                    f"{flow_id_audit['distinct_nonempty_flow_ids']} distinct Flow IDs); this is the safest "
                    "available session proxy, not an independently verified capture-session ID."),
    }
    if max_rows is not None:
        metadata["prefix_sample_warning"] = "This cache used a bounded max_rows read; do not call it a full-dataset evaluation."
    (out / "tiissrc23_preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(labels), "classes": classes, "n_traffic_subtypes": len(subtype_counts),
                      "flow_id_conflict_audit": flow_id_audit, "whole_dataset_duplicate_audit": duplicate_audit,
                      "split": {"train": len(split.train), "val": len(split.val), "test": len(split.test)}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="artifacts/tii_safe")
    parser.add_argument("--max-rows", type=int, default=0, help="0 (default) reads the complete CSV; a positive value produces an explicitly labeled bounded smoke cache")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_tii(args.csv, args.output_dir, args.max_rows or None, args.seed, args.chunksize)
