"""Create a fast, class-preserving subsample of an existing full multiview cache.

Unlike proportional subsampling, this caps each class at `max_per_class` rows
*within each already-correct partition* rather than sampling a fixed fraction
of the whole file. That matters for severely imbalanced datasets (e.g. TII's
Background class has 32 rows total out of 8.65M) -- a proportional 1% sample
would erase such classes entirely; a per-class cap keeps every available class
represented (using all its rows if there are fewer than the cap).

Because rows are only ever removed from within an existing train/val/test
partition (never moved across the boundary), the original group-disjoint split
guarantee is preserved automatically -- no group/label consistency re-check is
needed. This is an explicit, documented tractability decision for a hard time
budget, not a return to the old unstratified prefix-read bug.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _capped_indices(labels: np.ndarray, indices: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kept = []
    for c in np.unique(labels[indices]):
        class_idx = indices[labels[indices] == c]
        if len(class_idx) > max_per_class:
            class_idx = rng.choice(class_idx, size=max_per_class, replace=False)
        kept.append(class_idx)
    return np.concatenate(kept) if kept else np.asarray([], dtype=indices.dtype)


def subsample(cache: str, output_dir: str, max_train_per_class: int, max_val_per_class: int,
             max_test_per_class: int, seed: int = 42) -> dict:
    root = Path(cache).parent
    prefix = Path(cache).stem.removesuffix("_multiview")
    payload = np.load(cache, allow_pickle=True)
    split = np.load(root / f"{prefix}_split.npz", allow_pickle=True)
    preprocessing = json.loads((root / f"{prefix}_preprocessing.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((root / f"{prefix}_split.json").read_text(encoding="utf-8"))

    labels = payload["labels"]
    new_split = {
        "train": _capped_indices(labels, split["train"], max_train_per_class, seed),
        "val": _capped_indices(labels, split["val"], max_val_per_class, seed + 1),
        "test": _capped_indices(labels, split["test"], max_test_per_class, seed + 2),
    }

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    image = payload["image"]  # left as-is; lazy-image caches keep their (1,H,W) placeholder untouched
    np.savez_compressed(out / f"{prefix}_multiview.npz", image=image, timing=payload["timing"],
                        statistics=payload["statistics"], labels=labels, groups=payload["groups"],
                        view_mask=payload["view_mask"], classes=payload["classes"],
                        **({"feature_columns": payload["feature_columns"]} if "feature_columns" in payload else {}))
    np.savez_compressed(out / f"{prefix}_split.npz", train=new_split["train"], val=new_split["val"],
                        test=new_split["test"], groups=payload["groups"], classes=payload["classes"])

    class_names = payload["classes"].tolist()
    sizes = {part: {"before": int(len(split[part])), "after": int(len(idx))} for part, idx in new_split.items()}
    class_counts_after = {
        part: {class_names[c]: int((labels[idx] == c).sum()) for c in range(len(class_names))}
        for part, idx in new_split.items()
    }
    preprocessing = dict(preprocessing)
    preprocessing["subsampled_for_time_budget"] = {
        "method": "per-class cap within each already-correct partition (group-disjointness preserved; no rows moved across partitions)",
        "max_per_class": {"train": max_train_per_class, "val": max_val_per_class, "test": max_test_per_class},
        "sizes": sizes,
        "class_counts_after": class_counts_after,
        "reason": "Full-scale schedule was computationally infeasible within the session's hard time budget; "
                 "this is an explicit, documented tractability reduction, not a return to the unstratified prefix-read bug.",
    }
    (out / f"{prefix}_preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    split_manifest = dict(split_manifest)
    split_manifest["sizes"] = {k: v["after"] for k, v in sizes.items()}
    split_manifest["subsampled_for_time_budget"] = True
    (out / f"{prefix}_split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    print(json.dumps({"sizes": sizes}, indent=2))
    return {"sizes": sizes, "class_counts_after": class_counts_after}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-per-class", type=int, required=True)
    parser.add_argument("--max-val-per-class", type=int, required=True)
    parser.add_argument("--max-test-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    subsample(args.cache, args.output_dir, args.max_train_per_class, args.max_val_per_class,
              args.max_test_per_class, args.seed)
