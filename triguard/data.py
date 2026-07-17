"""Multi-view storage, split manifests, and train-only preprocessing.

The module deliberately keeps sample provenance and makes the split before
fitting imputers or scalers.  It is the single data entry point for all
validated experiments in this repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset


def _labels_to_ids(labels: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    classes = sorted({str(x) for x in labels})
    mapping = {name: idx for idx, name in enumerate(classes)}
    return np.asarray([mapping[str(x)] for x in labels], dtype=np.int64), classes


def _canonical_hash(row: np.ndarray) -> str:
    """Stable duplicate key after replacing NaN/Inf, not a label-aware key."""
    clean = np.nan_to_num(np.asarray(row, dtype=np.float32), nan=0.0,
                          posinf=np.finfo(np.float32).max,
                          neginf=np.finfo(np.float32).min)
    return hashlib.sha256(clean.tobytes()).hexdigest()


@dataclass
class SplitBundle:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    classes: list[str]
    groups: np.ndarray
    duplicate_overlap: dict[str, int]
    train_only_classes: dict[str, dict] | None = None

    def save(self, path: Path, dataset: str, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), train=self.train, val=self.val,
                            test=self.test, groups=self.groups,
                            classes=np.asarray(self.classes, dtype=object))
        manifest = {
            "dataset": dataset,
            "seed": seed,
            "split": "stratified split at the saved provenance-group level; group IDs are preserved and checked",
            "sizes": {"train": int(len(self.train)), "val": int(len(self.val)), "test": int(len(self.test))},
            "duplicate_overlap": self.duplicate_overlap,
            "classes": self.classes,
            "train_only_classes": self.train_only_classes or {},
            "all_classes_have_validation_and_test_support": not bool(self.train_only_classes),
            "exact_feature_duplicates_cross_partitions": any(v > 0 for v in self.duplicate_overlap.values()),
            "warning": "If a dataset has one unique Flow ID per row, group-disjointness is equivalent to sample-disjointness; source-file separation must be handled by the PCAP extractor.",
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def create_split_bundle(
    labels: np.ndarray,
    groups: np.ndarray,
    raw_features: np.ndarray,
    class_names: list[str],
    seed: int = 42,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    min_groups_for_stratified_split: int = 3,
) -> SplitBundle:
    """Create a deterministic stratified train/validation/test split.

    Group overlap is rejected.  For sources with repeated groups the function
    requires every group to have a single class; otherwise a source can never
    be a valid supervised split unit and should be repaired upstream.

    A class whose flows fall inside fewer than `min_groups_for_stratified_split`
    provenance groups (e.g. a rare 5G attack captured in a single small pcap)
    cannot be split into train/val/test without either violating group
    disjointness or guessing which single group becomes the held-out set. Per
    the paper's own stated policy, such a class is *not* silently dropped and
    does *not* crash the pipeline: all of its groups go to train (so the model
    still sees it and it still counts toward class weighting), it has zero
    val/test support by construction, and this is recorded explicitly in the
    split manifest as `train_only_classes` rather than hidden.
    """
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    if len(labels) != len(groups) or len(labels) != len(raw_features):
        raise ValueError("labels, groups, and features must have identical length")
    # Verify group-label consistency without an O(n^2) boolean scan. This path
    # is used on large CSV datasets where one group per row is common.
    order = np.argsort(groups)
    sorted_groups, sorted_labels = groups[order], labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_groups)]
    for start, end in zip(starts, ends):
        if sorted_labels[start:end].min() != sorted_labels[start:end].max():
            raise ValueError(f"group {sorted_groups[start]!r} has multiple labels; fix provenance before splitting")

    # Split at the group level, then expand group assignments back to rows. This
    # preserves stratification while guaranteeing that no session group is split.
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_labels = np.empty(len(unique_groups), dtype=labels.dtype)
    group_labels[inverse[order[starts]]] = sorted_labels[starts]
    group_idx = np.arange(len(unique_groups))

    group_counts_per_class = np.bincount(group_labels, minlength=len(class_names))
    insufficient_class_ids = {i for i, c in enumerate(group_counts_per_class)
                              if 0 < c < min_groups_for_stratified_split}
    splittable_mask = ~np.isin(group_labels, list(insufficient_class_ids))
    splittable_idx, insufficient_idx = group_idx[splittable_mask], group_idx[~splittable_mask]

    if len(splittable_idx) == 0:
        raise RuntimeError("No class has enough provenance groups for a stratified split")
    first = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_rel, test_rel = next(first.split(splittable_idx, group_labels[splittable_idx]))
    trainval_local, test_groups = splittable_idx[trainval_rel], splittable_idx[test_rel]
    val_rel = val_ratio / (1.0 - test_ratio)
    second = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    train_rel, val_rel_idx = next(second.split(trainval_local, group_labels[trainval_local]))
    train_groups = np.concatenate([trainval_local[train_rel], insufficient_idx])
    val_groups = trainval_local[val_rel_idx]
    train = np.flatnonzero(np.isin(inverse, train_groups))
    val = np.flatnonzero(np.isin(inverse, val_groups))
    test = np.flatnonzero(np.isin(inverse, test_groups))

    partitions = {"train": train, "val": val, "test": test}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(groups[partitions[left]]) & set(groups[partitions[right]])
        if overlap:
            raise RuntimeError(f"group leakage between {left} and {right}: {len(overlap)} groups")
    hashes = {name: {_canonical_hash(x) for x in raw_features[part]}
              for name, part in partitions.items()}
    duplicate_overlap = {
        "train_val": len(hashes["train"] & hashes["val"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "val_test": len(hashes["val"] & hashes["test"]),
    }
    train_only_classes = {
        class_names[i]: {"groups_available": int(group_counts_per_class[i]),
                         "reason": f"fewer than {min_groups_for_stratified_split} provenance groups; "
                                   "all flows routed to train, zero val/test support by construction"}
        for i in sorted(insufficient_class_ids)
    }
    return SplitBundle(train, val, test, class_names, groups, duplicate_overlap, train_only_classes)


def fit_train_only_features(raw: np.ndarray, split: SplitBundle) -> tuple[np.ndarray, dict]:
    """Median imputation and robust scaling fitted exclusively on train rows."""
    raw = np.asarray(raw, dtype=np.float32)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    medians = np.nanmedian(raw[split.train], axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    clean = np.where(np.isnan(raw), medians[None, :], raw)
    scaler = RobustScaler(quantile_range=(5, 95))
    scaled = scaler.fit(clean[split.train]).transform(clean).astype(np.float32)
    metadata = {
        "imputation": "column median fitted on train partition only",
        "scaler": "RobustScaler(quantile_range=(5,95)) fitted on train partition only",
        "medians": medians.astype(float).tolist(),
        "center": scaler.center_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
    }
    return scaled, metadata


def feature_images(scaled_features: np.ndarray, h: int = 40, w: int = 40) -> np.ndarray:
    """Compatibility image only; direct statistics remain the primary TII view."""
    n = len(scaled_features)
    canvas = np.zeros((n, h * w), dtype=np.float32)
    canvas[:, :min(scaled_features.shape[1], h * w)] = scaled_features[:, :h * w]
    canvas = 1.0 / (1.0 + np.exp(-np.clip(canvas, -30, 30)))
    return (canvas.reshape(n, h, w) * 255).round().astype(np.uint8)


class MultiViewDataset(Dataset):
    """Returns byte image, timing sequence, statistics, availability mask, label.

    When `image_lazy=True` (TII-SSRC-23 at full scale), `image` is only a tiny
    placeholder: materializing a real 40x40 image for all ~8.65M rows up front
    needs ~51.6 GiB. Instead `__getitem__` returns a zero placeholder and
    `lazy_image_collate` reconstructs the real compatibility image once per
    batch from the already-loaded scaled statistics, bounding peak memory to
    one batch instead of the whole dataset.
    """
    def __init__(self, image: np.ndarray, timing: np.ndarray, statistics: np.ndarray,
                 labels: np.ndarray, indices: np.ndarray, view_mask: np.ndarray | None = None,
                 image_lazy: bool = False):
        self.image = image
        self.image_lazy = image_lazy
        self.timing = timing
        self.statistics = statistics
        self.labels = labels
        self.indices = np.asarray(indices)
        if view_mask is None:
            view_mask = np.ones((len(labels), 3), dtype=np.bool_)
        self.view_mask = view_mask

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = self.indices[i]
        if self.image_lazy:
            img = torch.zeros(1, 40, 40, dtype=torch.float32)
        else:
            img = torch.from_numpy(self.image[j:j + 1].astype(np.float32) / 255.0)
        return {
            "image": img,
            "timing": torch.from_numpy(self.timing[j].astype(np.float32)),
            "statistics": torch.from_numpy(self.statistics[j].astype(np.float32)),
            "view_mask": torch.from_numpy(self.view_mask[j].astype(np.bool_)),
            "label": torch.tensor(int(self.labels[j]), dtype=torch.long),
            "sample_index": torch.tensor(int(j), dtype=torch.long),
        }


def lazy_image_collate(batch: list[dict]) -> dict:
    """Batch collate that reconstructs the TII compatibility image once per
    batch from scaled statistics, instead of once per dataset row."""
    from torch.utils.data import default_collate
    collated = default_collate(batch)
    stats = collated["statistics"].numpy()
    imgs = feature_images(stats)  # (B, 40, 40) uint8
    collated["image"] = torch.from_numpy(imgs[:, None, :, :].astype(np.float32) / 255.0)
    return collated
