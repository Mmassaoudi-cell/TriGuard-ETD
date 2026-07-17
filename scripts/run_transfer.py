"""Cross-dataset transfer diagnostic (binary benign-vs-malicious common ground).

USTC-TFC2016, 5GAD-2022, and TII-SSRC-23 have incompatible multiclass label
taxonomies, so no direct multiclass transfer is defined. Binary
benign-vs-malicious is the only label space common to all three; the mapping
for each dataset is declared explicitly below rather than inferred, and is a
real reduction in task difficulty compared to the multiclass results
elsewhere in this paper -- it is reported as such, not confused with them.

The byte (image) and timing branches take fixed-shape input (40x40 image,
5x5 timing) regardless of dataset, so their weights are transferable; the
statistics branch is not (input dimensionality differs per dataset), so a
transferred model keeps a freshly-initialized statistics branch and gate.
This is declared as the transferred encoder rather than "the whole model."
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from triguard.metrics import calibration_metrics, classification_metrics
from triguard.models import TriGuardETD
from triguard.runner import load_cache, make_loader, predict_torch, set_seed

BINARY_MAPS = {
    "ustc": {"benign": ["BitTorrent", "Facetime", "FTP", "Gmail", "MySQL", "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft"],
            "malicious": ["Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"]},
    "5gad": {"benign": ["Normal"],
            "malicious": ["AMFLookingForUDM", "automatedDropWithTimer", "automatedRedirectWithTimer", "CrashNRF",
                         "FakeAMFDelete", "FakeAMFInsert", "GetAllNFs", "GetUserData", "randomAMFInsert", "randomDataDump"]},
    "tii": {"benign": ["Audio", "Background", "Text", "Video"],
           "malicious": ["Bruteforce", "DoS", "Information Gathering", "Mirai"]},
}


def binarize(data: dict, dataset_key: str) -> dict:
    mapping = BINARY_MAPS[dataset_key]
    classes = data["classes"]
    benign_ids = {classes.index(c) for c in mapping["benign"] if c in classes}
    new_labels = np.asarray([0 if lbl in benign_ids else 1 for lbl in data["labels"]], dtype=np.int64)
    out = dict(data); out["labels"] = new_labels; out["classes"] = ["benign", "malicious"]
    return out


def train_binary_triguard(data: dict, seed: int, epochs: int, device: str, batch_size: int,
                          patience: int, transfer_state: dict | None = None,
                          log_prefix: str = "transfer") -> tuple[TriGuardETD, dict]:
    set_seed(seed)
    model = TriGuardETD(data["statistics"].shape[1], 2).to(device)
    transferred_keys = []
    if transfer_state is not None:
        own = model.state_dict()
        compatible = {k: v for k, v in transfer_state.items()
                     if k in own and own[k].shape == v.shape and (k.startswith("byte.") or k.startswith("timing."))}
        own.update(compatible); model.load_state_dict(own)
        transferred_keys = sorted(compatible.keys())
    train, val = make_loader(data, "train", batch_size, True), make_loader(data, "val", batch_size, False)
    counts = np.bincount(data["labels"][data["split"]["train"]], minlength=2)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / 2, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, no_improve = -1.0, None, 0
    epochs_run = epochs
    for epoch in range(epochs):
        model.train()
        for b in train:
            opt.zero_grad()
            out = model(b["image"].to(device), b["timing"].to(device), b["statistics"].to(device), b["view_mask"].to(device))
            loss = nn.functional.cross_entropy(out["logits"], b["label"].to(device), weight=weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        y, p, _, _ = predict_torch(model, val, device, "triguard")
        score = f1_score(y, p, average="macro", zero_division=0)
        if score > best: best, best_state, no_improve = score, copy.deepcopy(model.state_dict()), 0
        else: no_improve += 1
        if epoch % 25 == 0:
            print(f"[{log_prefix}] epoch {epoch + 1}/{epochs} val_macro_f1={score:.4f} best={best:.4f}", flush=True)
        if patience and no_improve >= patience:
            epochs_run = epoch + 1
            print(f"[{log_prefix}] early stop at epoch {epochs_run}/{epochs}", flush=True)
            break
    model.load_state_dict(best_state)
    return model, {"validation_macro_f1": best, "epochs_run": epochs_run, "transferred_parameter_tensors": transferred_keys}


def run_transfer(source_key: str, source_cache: str, target_key: str, target_cache: str, output: str,
                 seed: int = 1, epochs: int = 500, batch_size: int = 128, patience: int = 15) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_data, _ = load_cache(source_cache); src_data = binarize(src_data, source_key)
    tgt_data, _ = load_cache(target_cache); tgt_data = binarize(tgt_data, target_key)

    started = time.time()
    src_model, src_meta = train_binary_triguard(src_data, seed, epochs, device, batch_size, patience,
                                                log_prefix=f"source-{source_key}")
    source_time = time.time() - started

    started = time.time()
    scratch_model, scratch_meta = train_binary_triguard(tgt_data, seed, epochs, device, batch_size, patience,
                                                        log_prefix=f"target-{target_key}-scratch")
    y, p, pr, _ = predict_torch(scratch_model, make_loader(tgt_data, "test", batch_size, False), device, "triguard")
    scratch_metrics = classification_metrics(y, p, pr, tgt_data["classes"])

    started = time.time()
    transferred_model, transfer_meta = train_binary_triguard(tgt_data, seed, epochs, device, batch_size, patience,
                                                             transfer_state=src_model.state_dict(),
                                                             log_prefix=f"target-{target_key}-transferred")
    y, p, pr, _ = predict_torch(transferred_model, make_loader(tgt_data, "test", batch_size, False), device, "triguard")
    transfer_metrics = classification_metrics(y, p, pr, tgt_data["classes"])

    y, p, pr, _ = predict_torch(src_model, make_loader(src_data, "test", batch_size, False), device, "triguard")
    source_self_metrics = classification_metrics(y, p, pr, src_data["classes"])

    zero_shot_status = "not attempted: statistics branch dimensionality differs across datasets, so the " \
                       "source model cannot run forward on target inputs without the freshly-initialized " \
                       "statistics branch and gate seen during target training; a true zero-shot forward " \
                       "pass is architecturally undefined here and is not claimed."

    record = {"status": "cross-dataset-transfer-single-seed-diagnostic", "source": source_key, "target": target_key,
              "seed": seed, "binary_maps": BINARY_MAPS,
              "note": "Single-seed diagnostic under a binary benign-vs-malicious reduction, the only label "
                     "space common to all three datasets; not comparable to the multiclass main results. "
                     "Only byte/timing branch weights transfer (statistics branch dims differ per dataset).",
              "zero_shot": zero_shot_status,
              "source_self_evaluation": {**src_meta, **source_self_metrics, "train_time_seconds": source_time},
              "target_from_scratch": {**scratch_meta, **scratch_metrics},
              "target_transferred_encoder": {**transfer_meta, **transfer_metrics},
              "macro_f1_delta_transfer_minus_scratch": transfer_metrics["macro_f1"] - scratch_metrics["macro_f1"]}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[transfer {source_key}->{target_key}] scratch={scratch_metrics['macro_f1']:.4f} "
         f"transferred={transfer_metrics['macro_f1']:.4f} delta={record['macro_f1_delta_transfer_minus_scratch']:+.4f}", flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-key", required=True, choices=list(BINARY_MAPS))
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--target-key", required=True, choices=list(BINARY_MAPS))
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    run_transfer(args.source_key, args.source_cache, args.target_key, args.target_cache, args.output,
                args.seed, args.epochs, args.batch_size, args.patience)
