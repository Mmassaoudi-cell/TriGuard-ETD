"""Validated experiment runner for a cached multi-view dataset.

It is intentionally conservative: validation chooses checkpoints and teacher
weights; the test split is only read after model selection. The current CLI is
useful for smoke runs and repeated-seed studies, not for silently filling paper
tables with unexecuted values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from model import LitCVit
from .data import MultiViewDataset, lazy_image_collate
from .losses import class_balanced_focal, distillation_kl, supervised_contrastive, view_alignment
from .metrics import benchmark_latency, calibration_metrics, classification_metrics, model_macs, model_memory_mb
from .models import TriGuardETD


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(cache: str) -> tuple[dict, dict]:
    payload = np.load(cache, allow_pickle=True)
    root = Path(cache).parent
    prefix = Path(cache).stem.removesuffix("_multiview")
    split_npz_path = root / f"{prefix}_split.npz"
    split_json_path = root / f"{prefix}_split.json"
    preprocessing_path = root / f"{prefix}_preprocessing.json"
    split = np.load(split_npz_path, allow_pickle=True)
    data = {k: payload[k] for k in ("image", "timing", "statistics", "labels", "view_mask")}
    data["classes"] = payload["classes"].tolist()
    data["split"] = {k: split[k] for k in ("train", "val", "test")}
    metadata = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if split_json_path.exists():
        metadata["split_manifest"] = json.loads(split_json_path.read_text(encoding="utf-8"))
    metadata["artifact_checksums"] = {
        "cache_sha256": sha256_file(Path(cache)),
        "split_npz_sha256": sha256_file(split_npz_path),
        "split_manifest_sha256": sha256_file(split_json_path) if split_json_path.exists() else None,
        "preprocessing_sha256": sha256_file(preprocessing_path),
    }
    data["image_lazy"] = bool(metadata.get("image_lazy", False))
    return data, metadata


def make_loader(data: dict, partition: str, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    """`num_workers=0` by default: `MultiViewDataset.__getitem__` is cheap
    (numpy indexing, no augmentation), so multi-process loading buys no real
    throughput here. It does cost real memory on Windows, where multiprocessing
    uses `spawn` rather than `fork` -- each worker pickles and reloads its own
    full copy of the dataset's arrays (no copy-on-write), which multiplied
    memory badly on the multi-GB TII cache and contributed to an OOM. Only
    the (now-unused) NT-Xent pretraining path had per-sample work expensive
    enough to justify workers."""
    image_lazy = data.get("image_lazy", False)
    ds = MultiViewDataset(data["image"], data["timing"], data["statistics"], data["labels"],
                          data["split"][partition], data["view_mask"], image_lazy=image_lazy)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available(),
                      collate_fn=lazy_image_collate if image_lazy else None)


@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader, device: str, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval(); labels = []; preds = []; probs = []; gates = []
    for b in loader:
        y = b["label"].numpy(); labels.append(y)
        if kind == "litcvit":
            out = model(b["image"].to(device)); gate = None
        else:
            result = model(b["image"].to(device), b["timing"].to(device), b["statistics"].to(device), b["view_mask"].to(device))
            out, gate = result["logits"], result["gates"]
        pr = torch.softmax(out, dim=1).cpu().numpy(); probs.append(pr); preds.append(pr.argmax(1))
        if gate is not None: gates.append(gate.cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds), np.concatenate(probs), (np.concatenate(gates) if gates else None)


def train_litcvit(data: dict, seed: int, epochs: int, device: str, batch_size: int,
                  pretrain_checkpoint: str | None = None, freeze_epochs: int = 100,
                  emb_dim: int = 192, rank: int = 16, log_every: int = 10,
                  log_prefix: str = "litcvit", patience: int | None = 25) -> tuple[nn.Module, dict]:
    """Image-only LitCVit control trained with validation-only early stopping.

    `emb_dim=192` is the reference main-comparison configuration; pass 128 only
    for a separate, explicitly labeled efficiency-control run.

    `epochs=500` remains the reference schedule's hard cap, but a fixed
    500-epoch run is computationally infeasible across all three datasets and
    multiple seeds. Per the registered validation-only early-stopping
    allowance, training stops once validation macro-F1 has not improved for
    `patience` epochs (`patience=None` restores the original fixed schedule).
    When a pretrained encoder is transferred, early stopping never triggers
    before at least one epoch of the phase-2 unfreeze has run, so the
    two-phase schedule is never truncated to phase 1 only.
    """
    set_seed(seed)
    n_classes = len(data["classes"])
    transferred = bool(pretrain_checkpoint and Path(pretrain_checkpoint).exists())
    if transferred:
        model = LitCVit.from_pretrained(pretrain_checkpoint, n_classes=n_classes, emb_dim=emb_dim, rank=rank, dropout=0.2)
        model.freeze_encoder()
    else:
        model = LitCVit(emb_dim=emb_dim, rank=rank, n_classes=n_classes, dropout=0.2, mode="finetune")
    model = model.to(device)
    counts = np.bincount(data["labels"][data["split"]["train"]], minlength=n_classes)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes, dtype=torch.float32, device=device)
    def make_optimizer(unfrozen: bool):
        if not unfrozen:
            return torch.optim.AdamW(model.head.parameters(), lr=8e-4, weight_decay=1e-4)
        return torch.optim.AdamW([{"params": model.head.parameters(), "lr": 8e-4}, {"params": model.encoder.pkt_enc.parameters(), "lr": 1e-4},
                                  {"params": model.encoder.flow_enc.parameters(), "lr": 1e-4}, {"params": model.encoder.norm.parameters(), "lr": 1e-4}], weight_decay=1e-4)
    opt = (make_optimizer(False) if transferred else torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4))
    train, val = make_loader(data, "train", batch_size, True), make_loader(data, "val", batch_size, False)
    best_state, best = None, -1.0
    unfrozen = not transferred
    min_epochs = (freeze_epochs + 1) if transferred else 1
    no_improve = 0
    epochs_run = epochs
    started = time.time()
    for epoch in range(epochs):
        epoch_started = time.time()
        if transferred and not unfrozen and epoch == freeze_epochs:
            model.unfreeze_last_two_blocks(); opt = make_optimizer(True); unfrozen = True
        model.train()
        for b in train:
            opt.zero_grad(); logits = model(b["image"].to(device)); loss = nn.functional.cross_entropy(logits, b["label"].to(device), weight=weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        y, p, _, _ = predict_torch(model, val, device, "litcvit")
        score = f1_score(y, p, average="macro", zero_division=0)
        if score > best: best, best_state, no_improve = score, copy.deepcopy(model.state_dict()), 0
        else: no_improve += 1
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            epoch_seconds = time.time() - epoch_started
            eta = epoch_seconds * (epochs - epoch - 1)
            print(f"[{log_prefix}] epoch {epoch + 1}/{epochs} val_macro_f1={score:.4f} best={best:.4f} "
                 f"epoch_time={epoch_seconds:.1f}s elapsed={time.time() - started:.0f}s eta={eta:.0f}s", flush=True)
        if patience and (epoch + 1) >= min_epochs and no_improve >= patience:
            epochs_run = epoch + 1
            print(f"[{log_prefix}] early stop at epoch {epochs_run}/{epochs} "
                 f"(no val improvement for {patience} epochs)", flush=True)
            break
    model.load_state_dict(best_state)
    return model, {"validation_macro_f1": best, "parameters": sum(p.numel() for p in model.parameters()), "memory_mb": model_memory_mb(model),
                   "emb_dim": emb_dim, "rank": rank, "epochs_requested": epochs, "epochs_run": epochs_run,
                   "early_stopped": epochs_run < epochs, "patience": patience,
                   "pretrained": transferred, "pretrain_checkpoint": pretrain_checkpoint if transferred else None,
                   "freeze_epochs_requested": freeze_epochs, "last_two_blocks_unfrozen": unfrozen}


def _teacher_ensemble(data: dict, lit_model: nn.Module, device: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validation-F1 weighted LitCVit, LightGBM, and timing logistic teachers."""
    tr, va = data["split"]["train"], data["split"]["val"]
    n_classes = len(data["classes"])
    tree = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.06, num_leaves=24, random_state=seed, verbosity=-1)
    tree.fit(data["statistics"][tr], data["labels"][tr])
    time_teacher = LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    time_teacher.fit(data["timing"][tr].reshape(len(tr), -1), data["labels"][tr])
    loaders = {x: make_loader(data, x, 256, False) for x in ("train", "val")}
    _, _, ltr, _ = predict_torch(lit_model, loaders["train"], device, "litcvit")
    yv, _, lva, _ = predict_torch(lit_model, loaders["val"], device, "litcvit")
    def probabilities(model, x):
        raw = model.predict_proba(x)
        out = np.zeros((len(x), n_classes), dtype=np.float32)
        out[:, model.classes_.astype(int)] = raw
        return out
    ttr, tva = (probabilities(tree, data["statistics"][x]) for x in (tr, va))
    qtr, qva = (probabilities(time_teacher, data["timing"][x].reshape(len(x), -1)) for x in (tr, va))
    validation = [f1_score(yv, p.argmax(1), average="macro", zero_division=0) for p in (lva, tva, qva)]
    weights = np.asarray(validation, dtype=np.float64); weights /= weights.sum()
    return (sum(w * p for w, p in zip(weights, (ltr, ttr, qtr))), sum(w * p for w, p in zip(weights, (lva, tva, qva))),
            {"teacher_validation_macro_f1": dict(zip(["litcvit", "lightgbm", "timing_logistic"], validation)), "weights": dict(zip(["litcvit", "lightgbm", "timing_logistic"], weights.tolist()))})


def apply_view_override(data: dict, views: tuple[bool, bool, bool]) -> dict:
    """Return a copy of `data` with view_mask ANDed against a forced
    (byte, timing, statistics) availability pattern, for view-ablation runs.
    The original per-sample mask is preserved where it already excludes a
    view; this only ever removes availability, never adds it back."""
    forced = np.asarray(views, dtype=np.bool_)
    out = dict(data)
    out["view_mask"] = data["view_mask"] & forced[None, :]
    return out


def train_triguard(data: dict, lit_model: nn.Module, seed: int, epochs: int, device: str, batch_size: int,
                   kd_weight: float = 0.25, log_every: int = 10, log_prefix: str = "triguard",
                   patience: int | None = 25, ce_weight: float = 0.6, focal_weight: float = 0.4,
                   supcon_weight: float = 0.05, align_weight: float = 0.03) -> tuple[TriGuardETD, dict]:
    """`patience`: validation-macro-F1-based early stopping (see `train_litcvit`
    docstring for the rationale); `patience=None` restores the fixed schedule.
    The five `*_weight` arguments default to the reference composite loss;
    setting one to 0 is how the loss-ablation grid removes a term without
    touching any other code path."""
    set_seed(seed)
    n_classes = len(data["classes"])
    model = TriGuardETD(data["statistics"].shape[1], n_classes).to(device)
    teacher_train, _, teacher_meta = _teacher_ensemble(data, lit_model, device, seed)
    train, val = make_loader(data, "train", batch_size, True), make_loader(data, "val", batch_size, False)
    counts = np.bincount(data["labels"][data["split"]["train"]], minlength=n_classes)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    no_improve = 0
    epochs_run = epochs
    # Stable lookup preserves teacher order even though train loader is shuffled.
    row_to_teacher = {int(row): teacher_train[i] for i, row in enumerate(data["split"]["train"])}
    started = time.time()
    for epoch in range(epochs):
        epoch_started = time.time()
        model.train()
        # Use deterministic in-memory sample index recovery via labels/features
        # is unsafe; train without KD if batch lookup isn't explicit. The next
        # runner revision adds index tensors. KD remains available for ordered
        # full-batch evaluation but is not silently misaligned here.
        for b in train:
            opt.zero_grad()
            out = model(b["image"].to(device), b["timing"].to(device), b["statistics"].to(device), b["view_mask"].to(device))
            labels = b["label"].to(device)
            focal = class_balanced_focal(out["logits"], labels, weights)
            supcon = supervised_contrastive(out["projection"], labels)
            align = view_alignment([out["byte_embedding"], out["timing_embedding"], out["statistics_embedding"]], b["view_mask"].to(device))
            teacher = torch.as_tensor(np.stack([row_to_teacher[int(i)] for i in b["sample_index"].numpy()]), dtype=torch.float32, device=device)
            kd = distillation_kl(out["logits"], teacher, temperature=2.0)
            # CE + focal protect calibration/minority recall; SupCon, KD, and
            # alignment shape complementary discriminative views.
            loss = (ce_weight * nn.functional.cross_entropy(out["logits"], labels, weight=weights)
                   + focal_weight * focal + supcon_weight * supcon + kd_weight * kd + align_weight * align)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        y, p, _, _ = predict_torch(model, val, device, "triguard")
        score = f1_score(y, p, average="macro", zero_division=0)
        if score > best: best, best_state, no_improve = score, copy.deepcopy(model.state_dict()), 0
        else: no_improve += 1
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            epoch_seconds = time.time() - epoch_started
            eta = epoch_seconds * (epochs - epoch - 1)
            print(f"[{log_prefix}] epoch {epoch + 1}/{epochs} val_macro_f1={score:.4f} best={best:.4f} "
                 f"epoch_time={epoch_seconds:.1f}s elapsed={time.time() - started:.0f}s eta={eta:.0f}s", flush=True)
        if patience and no_improve >= patience:
            epochs_run = epoch + 1
            print(f"[{log_prefix}] early stop at epoch {epochs_run}/{epochs} "
                 f"(no val improvement for {patience} epochs)", flush=True)
            break
    model.load_state_dict(best_state)
    return model, {"validation_macro_f1": best, "parameters": model.parameter_count, "memory_mb": model_memory_mb(model),
                   "epochs_requested": epochs, "epochs_run": epochs_run, "early_stopped": epochs_run < epochs,
                   "patience": patience,
                   "distillation": {"enabled": kd_weight > 0, "temperature": 2.0, "kd_weight": kd_weight},
                   "loss_weights": {"ce": ce_weight, "focal": focal_weight, "supcon": supcon_weight,
                                    "kd": kd_weight, "align": align_weight},
                   "view_availability": {"byte": bool(data["view_mask"][:, 0].any()),
                                        "timing": bool(data["view_mask"][:, 1].any()),
                                        "statistics": bool(data["view_mask"][:, 2].any())},
                   **teacher_meta}


def _residual_limitations(metadata: dict) -> list[str]:
    """Pull only the true, dataset-specific caveats out of a cache's own
    preprocessing metadata, instead of hardcoding one dataset's wording."""
    limitations = []
    for key in ("prefix_sample_warning", "timing_view", "grouping"):
        if metadata.get(key):
            limitations.append(metadata[key])
    manifest = metadata.get("split_manifest", {})
    train_only = manifest.get("train_only_classes", {})
    if train_only:
        limitations.append("The split is not an all-class held-out evaluation: "
                           + ", ".join(sorted(train_only))
                           + " have zero validation/test support under the saved provenance grouping.")
    duplicate_overlap = manifest.get("duplicate_overlap", {})
    if any(int(v) > 0 for v in duplicate_overlap.values()):
        limitations.append("Exact feature-vector duplicates occur across partitions "
                           f"(train-val={duplicate_overlap.get('train_val', 0)}, "
                           f"train-test={duplicate_overlap.get('train_test', 0)}, "
                           f"val-test={duplicate_overlap.get('val_test', 0)}); these are reported as duplicate risk, not hidden.")
    return limitations


def run_experiment(dataset: str, cache: str, output: str, seed: int = 42,
                   litcvit_epochs: int = 500, triguard_epochs: int = 500, batch_size: int = 128,
                   pretrain_checkpoint: str | None = None, emb_dim: int = 192, rank: int = 16,
                   freeze_epochs: int = 100, status: str = "full-protocol-run", log_every: int = 10,
                   patience: int | None = 25) -> dict:
    """Train the reference LitCVit baseline and TriGuard-ETD on one frozen manifest.

    Dataset-agnostic: `dataset`/`output`/`status` are supplied by the caller
    (e.g. the multi-seed sweep driver) rather than inferred, so a pilot run and
    a final-protocol run are never confused with each other.

    `patience`: validation-macro-F1 early-stopping window applied to both
    models in place of always running the full `*_epochs` budget (see
    `train_litcvit` for the exact rule); this is the registered
    validation-only early-stopping variant of the reference schedule, adopted
    because the fixed 500-epoch schedule is computationally infeasible across
    all three datasets and multiple seeds (measured: ~55h/seed for LitCVit
    alone on USTC, ~956h/seed on TII, at 500 fixed epochs).
    """
    data, metadata = load_cache(cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    lit, lit_meta = train_litcvit(data, seed, litcvit_epochs, device, batch_size, pretrain_checkpoint,
                                  freeze_epochs=freeze_epochs, emb_dim=emb_dim, rank=rank,
                                  log_every=log_every, log_prefix=f"{dataset}:litcvit:seed{seed}", patience=patience)
    tri, tri_meta = train_triguard(data, lit, seed, triguard_epochs, device, batch_size,
                                   log_every=log_every, log_prefix=f"{dataset}:triguard:seed{seed}", patience=patience)
    test = make_loader(data, "test", batch_size, False)
    yl, pl, prl, _ = predict_torch(lit, test, device, "litcvit")
    yt, pt, prt, gates = predict_torch(tri, test, device, "triguard")
    result = {"status": status, "dataset": dataset, "seed": seed, "rows": len(data["labels"]),
              "classes": data["classes"], "split_sizes": {k: len(v) for k, v in data["split"].items()},
              "test_used_after_validation_selection": True, "limitations": _residual_limitations(metadata),
              "split_manifest": metadata.get("split_manifest", {}),
              "artifact_checksums": metadata.get("artifact_checksums", {}),
              "litcvit": {**lit_meta, **classification_metrics(yl, pl, prl, data["classes"]),
                         "calibration": calibration_metrics(yl, prl)},
              "triguard": {**tri_meta, **classification_metrics(yt, pt, prt, data["classes"]),
                          "average_gate_weights": gates.mean(0).tolist(), "calibration": calibration_metrics(yt, prt)},
              "train_time_seconds": time.time() - started}
    # Benchmark follows the same batch-one protocol for each model.
    sample = next(iter(test)); img, tim, stat, mask = (sample[k][:1].to(device) for k in ("image", "timing", "statistics", "view_mask"))
    with torch.inference_mode():
        result["litcvit"]["latency"] = benchmark_latency(lambda: lit(img), device)
        result["triguard"]["latency"] = benchmark_latency(lambda: tri(img, tim, stat, mask), device)
        result["litcvit"]["complexity"] = model_macs(lit, (img,))
        result["triguard"]["complexity"] = model_macs(tri, (img, tim, stat, mask))
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset label recorded in the result JSON, e.g. USTC-TFC2016, 5GAD-2022, TII-SSRC-23")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--litcvit-epochs", type=int, default=500)
    parser.add_argument("--triguard-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-checkpoint", default=None)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--freeze-epochs", type=int, default=100)
    parser.add_argument("--status", default="full-protocol-run")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=25, help="0 disables early stopping and restores the fixed schedule")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.dataset, args.cache, args.output, args.seed, args.litcvit_epochs,
                                    args.triguard_epochs, args.batch_size, args.pretrain_checkpoint,
                                    args.emb_dim, args.rank, args.freeze_epochs, args.status, args.log_every,
                                    args.patience or None), indent=2))
