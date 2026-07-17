# TriGuard-ETD

TriGuard-ETD is a compact, multi-view detector for encrypted malicious traffic. It fuses three representations of a network flow — a byte-level ET-flow image, a packet timing/length sequence, and direct flow-statistical features — through depthwise convolutional, temporal, and MLP branches, combined with a masked adaptive gate. The student model is trained with a composite objective (cross-entropy, class-balanced focal loss, supervised contrastive learning, knowledge distillation from a validation-weighted teacher ensemble, and cross-view alignment) and is compared against a from-scratch reproduction of LitCVit, a lightweight self-supervised contrastive vision transformer baseline for the same task.

This repository contains the full implementation used to produce the accompanying paper's results: dataset preprocessing, the LitCVit baseline, the TriGuard-ETD model and losses, the leakage-safe training and evaluation pipeline, and the ablation, robustness, baseline-comparison, and cross-dataset-transfer scripts. It does not include raw datasets, precomputed caches, trained checkpoints, or the paper's result tables and figures — only the source code needed to regenerate them.

## Design principles this codebase enforces

The pipeline is built around a leakage-safe evaluation protocol rather than a conventional random train/test split, because encrypted-traffic datasets are unusually easy to leak across a split boundary (duplicated flows, shared source captures, or scalers fitted before splitting). Concretely:

Every dataset is assigned a provenance group before any preprocessing happens — for packet-capture data this is the source capture file plus a canonical five-tuple, and for CSV-only data it is a Flow ID and timestamp composite, since Flow ID alone was found to co-occur with conflicting labels in at least one of the target datasets. Splitting happens at the group level, so no single flow's packets or duplicate feature vectors can appear in more than one partition. Median imputation and feature scaling are fit exclusively on the training partition and then applied to validation and test rows. Self-supervised pretraining, where used, reads only the training partition. Early stopping, teacher-ensemble weighting, and any hyperparameter search use validation performance only; the test partition is loaded once, after every modeling choice has already been locked in.

A class whose flows fall inside too few provenance groups to support a three-way split is not silently dropped and does not crash the pipeline — its groups are routed entirely into the training partition, it is recorded explicitly as train-only in the split manifest, and it correctly receives zero held-out support in the reported metrics rather than being hidden from them.

## Repository layout

`config.py` and `main.py` are the entry points for the original single-pass LitCVit reproduction pipeline (preprocess, pretrain, fine-tune, evaluate, report).

`data/` contains the original PCAP and CSV preprocessing utilities and PyTorch dataset classes used by that pipeline.

`model/` contains the LitCVit architecture (convolutional patchify stem, hierarchical windowed factorized attention, NT-Xent contrastive pretraining head, and classification head) and its loss functions.

`training/` contains the self-supervised pretraining loop and the two-phase fine-tuning loop (frozen encoder, then partial unfreezing) for LitCVit.

`evaluation/` contains classification metrics, confusion matrices, PR curves, and a results-table generator for the LitCVit pipeline. A legacy module that stored hardcoded literature values instead of measured results was deliberately excluded from this repository; nothing in this codebase should be treated as an experimental result unless it was produced by actually running the code against real data.

`triguard/` is the leakage-safe multi-view pipeline that supersedes the original single-baseline reproduction. It owns provenance-preserving PCAP and CSV extraction, group-disjoint split construction with train-only preprocessing fits, the TriGuard-ETD model and its loss terms, the adaptive gate, the validation-weighted teacher ensemble (LitCVit image teacher, LightGBM statistics teacher, logistic-regression timing teacher), classification and calibration metrics, and multi-seed statistical aggregation with paired significance testing and Holm correction.

`scripts/` contains the higher-level experiment drivers built on top of `triguard/`: a resumable multi-seed sweep runner, a class-preserving stratified-subsampling utility for datasets too large to train on repeatedly within a limited time budget, a view- and loss-ablation grid runner, a single-model robustness/evasion perturbation runner, a standalone conventional-baseline runner (LightGBM, XGBoost, logistic regression, early fusion), a cross-dataset transfer diagnostic, and a sequential queue for running several long experiments one at a time on a single GPU.

## Requirements

Python 3.10 or later and the packages listed in `requirements.txt`, which cover PyTorch, scikit-learn, LightGBM, XGBoost, pandas, and scapy. PCAP parsing depends on scapy and, on some platforms, a working libpcap installation; consult scapy's own installation notes if capture reading fails. A CUDA-capable GPU is strongly recommended for anything beyond the smallest datasets, though every training script also runs on CPU.

Install dependencies with pip against the provided requirements file from the repository root.

## Data

No raw datasets are distributed with this repository. The paper uses USTC-TFC2016, 5GAD-2022, and TII-SSRC-23, each of which must be obtained separately from its original source and placed on disk before running any preprocessing script. USTC-TFC2016 and 5GAD-2022 are supplied as packet captures; TII-SSRC-23 is supplied as a flow-statistics CSV with an accompanying packet-capture corpus that this codebase intentionally does not claim to have verified a row-level join against, so TII is treated as a statistics-and-timing-proxy dataset rather than a genuine tri-view one unless that join is independently established.

## Reproducing the pipeline

The general workflow is: extract provenance-preserving multi-view caches from raw captures or CSVs, optionally build a class-preserving stratified subsample of a cache if the full dataset is too large to train on repeatedly, run the multi-seed sweep to train the LitCVit baseline and TriGuard-ETD under an identical frozen manifest for each seed, and aggregate the resulting per-seed result files into summary statistics with paired significance tests.

Preprocessing entry points live in `triguard/preprocess_pcap.py` (USTC-TFC2016 and 5GAD-2022) and `triguard/preprocess_tii.py` (TII-SSRC-23); each is a standalone script invoked with `python -m` and a `--root` or `--csv` path to the raw data, plus an output directory for the resulting cache. Both refuse to proceed if a supplied 5GAD capture turns out to be an unresolved Git LFS pointer rather than real packet data, since silently parsing a pointer file as a capture would otherwise produce a misleadingly successful but empty run.

`scripts/subsample_cache.py` (and the equivalent `triguard/sample_cache.py`) draw a class-preserving sample from an already-preprocessed cache by capping the number of rows per class within each existing train/validation/test partition, rather than sampling proportionally, so that a dataset's rarest classes are retained rather than erased by a size reduction.

`scripts/run_stage1_sweep.py` is the main experiment driver: given a cache, a list of seeds, and an epoch budget, it trains the LitCVit baseline and TriGuard-ETD once per seed on the identical frozen manifest, skips any seed whose result file already exists so an interrupted run can be resumed without repeating finished work, and aggregates the completed seeds into a summary report once enough of them are done. Training uses validation-macro-F1-based early stopping rather than always running a fixed epoch budget, which is a registered variation of the reference protocol adopted because the fixed-epoch schedule is not computationally tractable across every dataset and seed count.

`triguard/statistics.py` implements the aggregation step on its own if you need to re-run it independently of the sweep driver: it computes per-metric means, standard deviations, 95% confidence intervals, paired t-tests, Wilcoxon signed-rank tests, and Holm-corrected significance across a directory of matched-seed result files, and refuses to mix result files that used different manifests or different run statuses.

The remaining scripts in `scripts/` are independent diagnostics rather than part of the core training loop: `run_ablations.py` runs the view- and loss-term ablation grid, `run_robustness.py` applies inference-time perturbations to an already-trained model's held-out data, `run_baselines.py` trains standalone conventional classifiers on the same frozen features TriGuard-ETD uses, and `run_transfer.py` runs a binary cross-dataset transfer diagnostic between two datasets' byte and timing branches.

## Notes on what this code does not claim

This is research code for a paper that documents its own limitations in detail; the code itself reflects that stance rather than smoothing it over. Model checkpoints are not persisted by the main sweep driver, so metrics computed inline during a run (including calibration statistics) are not retroactively recoverable without re-running that seed. Ablation and robustness scripts default to small seed counts because they are diagnostic in nature, and their result files are tagged with a status string that distinguishes them from the main multi-seed statistical comparisons. Readers looking for the paper's actual measured results, tables, and figures should consult the paper itself rather than this repository, which intentionally contains only the code that produced them.

## Citation

If you use this code, please cite the accompanying paper. A citation entry will be added here once the paper has a final, stable reference.
