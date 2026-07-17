"""Seed aggregation with paired tests and Holm correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .metrics import summarize_seeds


def _metric_value(row: dict, model: str, metric: str) -> float:
    """Return a metric with conservative repairs for older result JSONs.

    Early runner versions called sklearn's macro-F1 without passing the full
    registered label set.  If a split had a train-only class, the scalar could
    omit that absent class.  The per-class table was already label-complete, so
    aggregates recompute macro quantities from it when possible.
    """
    block = row[model]
    if metric == "macro_f1" and block.get("per_class"):
        return float(np.mean([float(x.get("f1", 0.0)) for x in block["per_class"]]))
    if metric == "precision" and block.get("per_class"):
        return float(np.mean([float(x.get("precision", 0.0)) for x in block["per_class"]]))
    if metric == "recall" and block.get("per_class"):
        return float(np.mean([float(x.get("recall", 0.0)) for x in block["per_class"]]))
    return float(block[metric])


def holm(p_values: dict[str, float]) -> dict[str, dict]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered); adjusted = {}; running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (m - rank)))
        adjusted[name] = {"raw_p": float(value), "holm_p": float(running), "reject_0.05": bool(running < 0.05)}
    return adjusted


def aggregate(directory: str, output: str, allowed_status: tuple[str, ...] = ("full-protocol-run",),
             metrics: tuple[str, ...] = ("accuracy", "macro_f1", "weighted_f1", "pr_auc", "roc_auc", "fpr", "fnr")) -> dict:
    """Aggregate a directory of matched-seed `run_experiment` result JSONs.

    `allowed_status` gates which run type this aggregation is allowed to
    summarize (default: only genuine full-protocol runs on frozen, full-data
    manifests). Pass e.g. `("pilot-run",)` explicitly for a calibration run;
    the resulting report is labeled accordingly and must not be silently
    relabeled as a main result.
    """
    files = sorted(Path(directory).glob("seed*.json"))
    if len(files) < 2:
        raise ValueError("Need at least two seed result files")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    bad = {r.get("status") for r in rows if r.get("status") not in allowed_status}
    if bad:
        raise ValueError(f"Refusing to aggregate: found result status(es) {sorted(bad)} not in allowed_status={allowed_status}")
    datasets = {r.get("dataset") for r in rows}
    if len(datasets) > 1:
        raise ValueError(f"Refusing to aggregate across different datasets/manifests: {sorted(datasets)}")
    summary = {model: {metric: summarize_seeds([{metric: _metric_value(r, model, metric)} for r in rows], metric)
                       for metric in metrics} for model in ("litcvit", "triguard")}
    pvals = {}
    tests = {}
    for metric in metrics:
        base = np.asarray([_metric_value(r, "litcvit", metric) for r in rows])
        proposed = np.asarray([_metric_value(r, "triguard", metric) for r in rows])
        try: t = stats.ttest_rel(proposed, base); w = stats.wilcoxon(proposed, base, alternative="two-sided")
        except ValueError: t = type("R", (), {"statistic": np.nan, "pvalue": np.nan})(); w = t
        tests[metric] = {"mean_delta_triguard_minus_litcvit": float((proposed - base).mean()), "paired_t_statistic": float(t.statistic),
                         "paired_t_p": float(t.pvalue), "wilcoxon_statistic": float(w.statistic), "wilcoxon_p": float(w.pvalue)}
        pvals[metric] = float(w.pvalue) if np.isfinite(w.pvalue) else 1.0
    # Residual, run-specific caveats (e.g. a CSV timing proxy) are collected
    # from the individual seed results rather than a hardcoded dataset note.
    limitations = sorted({item for r in rows for item in r.get("limitations", [])})
    absent = {
        model: sorted({pc["class"] for r in rows for pc in r[model].get("per_class", []) if int(pc.get("support", 0)) == 0})
        for model in ("litcvit", "triguard")
    }
    warnings = []
    for model, classes in absent.items():
        if classes:
            warnings.append(f"{model} test split has zero support for: {', '.join(classes)}; macro metrics include these classes as zero-F1.")
    report = {"status": rows[0]["status"], "dataset": rows[0].get("dataset"), "n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
              "limitations": limitations, "evaluation_warnings": warnings,
              "summary": summary, "paired_tests": tests, "holm_wilcoxon": holm(pvals)}
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-status", nargs="+", default=["full-protocol-run"])
    args = parser.parse_args()
    print(json.dumps(aggregate(args.directory, args.output, tuple(args.allowed_status)), indent=2))
