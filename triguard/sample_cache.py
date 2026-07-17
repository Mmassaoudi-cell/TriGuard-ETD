"""Create leakage-preserving stratified samples from an existing multiview cache.

This is used when a complete cache is available but full repeated training is
computationally impractical.  The sampler draws only from the already frozen
train/validation/test partitions, so no row crosses a split boundary.  It does
not refit preprocessing: the source cache must already have fitted imputation
and scaling on its training partition only.  The output metadata records this
limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _choose(labels: np.ndarray, indices: np.ndarray, n_classes: int, per_class: int, rng: np.random.Generator) -> np.ndarray:
    picked: list[np.ndarray] = []
    part_labels = labels[indices]
    for cls in range(n_classes):
        cls_idx = indices[part_labels == cls]
        if len(cls_idx) <= per_class:
            picked.append(cls_idx)
        else:
            picked.append(rng.choice(cls_idx, size=per_class, replace=False))
    out = np.concatenate(picked) if picked else np.asarray([], dtype=np.int64)
    rng.shuffle(out)
    return out


def sample_cache(source_cache: str, output_dir: str, output_prefix: str,
                 train_per_class: int = 10_000, val_per_class: int = 2_000,
                 test_per_class: int = 2_000, seed: int = 42) -> dict:
    source = Path(source_cache)
    root = source.parent
    source_prefix = source.stem.removesuffix("_multiview")
    source_split = np.load(root / f"{source_prefix}_split.npz", allow_pickle=True)
    source_meta_path = root / f"{source_prefix}_preprocessing.json"
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    payload = np.load(source, allow_pickle=True)
    labels = payload["labels"]
    classes = payload["classes"].tolist()
    n_classes = len(classes)
    rng = np.random.default_rng(seed)

    train = _choose(labels, source_split["train"], n_classes, train_per_class, rng)
    val = _choose(labels, source_split["val"], n_classes, val_per_class, rng)
    test = _choose(labels, source_split["test"], n_classes, test_per_class, rng)
    selected = np.concatenate([train, val, test])
    remap = {int(old): i for i, old in enumerate(selected)}
    new_split = {
        "train": np.asarray([remap[int(i)] for i in train], dtype=np.int64),
        "val": np.asarray([remap[int(i)] for i in val], dtype=np.int64),
        "test": np.asarray([remap[int(i)] for i in test], dtype=np.int64),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_cache = out / f"{output_prefix}_multiview.npz"
    image_lazy = bool(source_meta.get("image_lazy", False))
    image = payload["image"] if image_lazy else payload["image"][selected]
    arrays = {
        "image": image,
        "timing": payload["timing"][selected],
        "statistics": payload["statistics"][selected],
        "labels": labels[selected],
        "groups": payload["groups"][selected],
        "view_mask": payload["view_mask"][selected],
        "classes": payload["classes"],
    }
    if "feature_columns" in payload:
        arrays["feature_columns"] = payload["feature_columns"]
    np.savez_compressed(out_cache, **arrays)
    np.savez_compressed(out / f"{output_prefix}_split.npz", **new_split,
                        groups=arrays["groups"], classes=payload["classes"])

    def counts(indices: np.ndarray) -> dict[str, int]:
        c = np.bincount(arrays["labels"][indices], minlength=n_classes)
        return {classes[i]: int(c[i]) for i in range(n_classes)}

    duplicate = source_meta.get("whole_dataset_duplicate_audit", {})
    split_class_counts = {k: counts(v) for k, v in new_split.items()}
    split_manifest = {
        "dataset": f"{source_meta.get('dataset', source_prefix)} stratified sample",
        "seed": seed,
        "source_cache": str(source.resolve()),
        "source_split": str((root / f"{source_prefix}_split.npz").resolve()),
        "split": "sampled inside the already frozen source train/validation/test partitions",
        "sizes": {k: int(len(v)) for k, v in new_split.items()},
        "classes": classes,
        "class_counts": split_class_counts,
        "all_classes_have_validation_and_test_support": all(split_class_counts["test"][name] > 0 and split_class_counts["val"][name] > 0 for name in classes),
        "duplicate_note": "duplicate audit inherited from source cache unless recomputed downstream",
    }
    (out / f"{output_prefix}_split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    metadata = dict(source_meta)
    metadata.update({
        "dataset": split_manifest["dataset"],
        "source_cache": str(source.resolve()),
        "rows": int(len(selected)),
        "sampled_from_complete_cache": True,
        "sample_seed": seed,
        "sample_caps_per_partition": {
            "train_per_class": train_per_class,
            "val_per_class": val_per_class,
            "test_per_class": test_per_class,
        },
        "sampling_note": ("Rows were sampled within the source cache's frozen partitions. "
                          "Preprocessing was inherited from the complete source cache and was fitted on source-train rows only, "
                          "not refitted on this smaller sample."),
        "source_duplicate_audit": duplicate,
    })
    (out / f"{output_prefix}_preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_cache": str(out_cache), "rows": int(len(selected)), "split": split_manifest["sizes"],
            "class_counts": split_manifest["class_counts"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="sample")
    parser.add_argument("--train-per-class", type=int, default=10_000)
    parser.add_argument("--val-per-class", type=int, default=2_000)
    parser.add_argument("--test-per-class", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(sample_cache(args.source_cache, args.output_dir, args.output_prefix,
                                  args.train_per_class, args.val_per_class, args.test_per_class, args.seed), indent=2))
