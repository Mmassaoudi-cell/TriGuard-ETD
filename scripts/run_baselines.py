"""Standalone conventional baselines, trained and evaluated on the same
frozen manifest and train-only-fitted features as TriGuard-ETD and LitCVit.

Covers: LightGBM (statistics), XGBoost (statistics), logistic regression
(statistics, balanced), and an early-concatenation baseline (byte image
flattened + timing flattened + statistics, single feature vector) trained
with LightGBM. These are reported as directly deployable baselines, distinct
from the *teacher* roles LightGBM/logistic already play inside TriGuard-ETD's
distillation ensemble -- the same model class, but evaluated here as a
standalone classifier rather than as a distillation input.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from triguard.metrics import calibration_metrics, classification_metrics
from triguard.runner import load_cache


def _predict_proba_full(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), n_classes), dtype=np.float32)
    out[:, model.classes_.astype(int)] = raw
    return out


def run_baselines(dataset: str, cache: str, output: str, seed: int = 1) -> dict:
    data, metadata = load_cache(cache)
    tr, te = data["split"]["train"], data["split"]["test"]
    n_classes = len(data["classes"])
    y_tr, y_te = data["labels"][tr], data["labels"][te]
    x_stats_tr, x_stats_te = data["statistics"][tr], data["statistics"][te]
    # Lazy-image caches (TII at full/subsample scale) only carry a tiny
    # placeholder in data["image"]; early fusion there concatenates the
    # genuinely available timing and statistics views, not a materialized
    # zero-padded compatibility image.
    image_lazy = data.get("image_lazy", False)
    image_parts_tr = [] if image_lazy else [data["image"][tr].reshape(len(tr), -1).astype(np.float32) / 255.0]
    image_parts_te = [] if image_lazy else [data["image"][te].reshape(len(te), -1).astype(np.float32) / 255.0]
    x_early_tr = np.concatenate(image_parts_tr + [data["timing"][tr].reshape(len(tr), -1), x_stats_tr], axis=1)
    x_early_te = np.concatenate(image_parts_te + [data["timing"][te].reshape(len(te), -1), x_stats_te], axis=1)

    results = {}

    started = time.time()
    lgbm = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.06, num_leaves=24, random_state=seed, verbosity=-1)
    lgbm.fit(x_stats_tr, y_tr)
    pr = _predict_proba_full(lgbm, x_stats_te, n_classes)
    results["lightgbm_statistics"] = {**classification_metrics(y_te, pr.argmax(1), pr, data["classes"]),
                                      "calibration": calibration_metrics(y_te, pr),
                                      "train_time_seconds": time.time() - started,
                                      "parameters": int(sum(t.num_leaves() for t in [lgbm.booster_])) if hasattr(lgbm.booster_, "num_leaves") else None}

    started = time.time()
    xgbm = xgb.XGBClassifier(n_estimators=180, learning_rate=0.06, max_depth=6, random_state=seed,
                             eval_metric="mlogloss", verbosity=0)
    xgbm.fit(x_stats_tr, y_tr)
    pr = xgbm.predict_proba(x_stats_te)
    results["xgboost_statistics"] = {**classification_metrics(y_te, pr.argmax(1), pr, data["classes"]),
                                     "calibration": calibration_metrics(y_te, pr),
                                     "train_time_seconds": time.time() - started}

    started = time.time()
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    logreg.fit(x_stats_tr, y_tr)
    pr = _predict_proba_full(logreg, x_stats_te, n_classes)
    results["logistic_regression_statistics"] = {**classification_metrics(y_te, pr.argmax(1), pr, data["classes"]),
                                                  "calibration": calibration_metrics(y_te, pr),
                                                  "train_time_seconds": time.time() - started}

    started = time.time()
    lgbm_early = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.06, num_leaves=24, random_state=seed, verbosity=-1)
    lgbm_early.fit(x_early_tr, y_tr)
    pr = _predict_proba_full(lgbm_early, x_early_te, n_classes)
    results["early_fusion_lightgbm"] = {**classification_metrics(y_te, pr.argmax(1), pr, data["classes"]),
                                        "calibration": calibration_metrics(y_te, pr),
                                        "train_time_seconds": time.time() - started,
                                        "input_dim": int(x_early_tr.shape[1])}

    record = {"status": "standalone-baseline-run", "dataset": dataset, "seed": seed,
              "note": "Trained on the same frozen train partition and train-only-fitted statistics as "
                     "TriGuard-ETD/LitCVit; not the same object as the LightGBM/logistic *teachers* used "
                     "inside TriGuard-ETD's distillation ensemble, though the same model class.",
              "results": results}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    for name, r in results.items():
        print(f"[baseline {name}] macro_f1={r['macro_f1']:.4f} accuracy={r['accuracy']:.4f}", flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run_baselines(args.dataset, args.cache, args.output, args.seed)
