# TriGuard-ETD

TriGuard-ETD is a compact multi-view detector for encrypted malicious traffic. It encodes three views of a network flow (a 40x40 byte image, a 5x5 packet-timing sequence, and a flow-statistics vector) with lightweight depthwise-convolutional, temporal, and gated-MLP branches, and fuses them with an availability-masked adaptive gate that gives zero weight to views a sample does not have. Training combines class-weighted cross-entropy, a class-balanced focal loss, supervised contrastive learning, cross-view alignment, per-view auxiliary supervision, and knowledge distillation from a teacher ensemble of temperature-calibrated specialists (LightGBM, XGBoost, ExtraTrees, RandomForest, and a logistic regression on timing). The ensemble weighting and the stopping epoch are chosen on validation macro-F1 only.

The whole model (fitting, calibration, and scoring) is in one file, `triguard_etd.py`.

## Installation

Python 3.10 or later. Install the dependencies directly:

    pip install torch numpy scipy scikit-learn lightgbm xgboost

Then copy `triguard_etd.py` next to your code, or clone this repository.

## Usage

Each partition is a dictionary with `image` (N,40,40 uint8), `timing` (N,5,5), `statistics` (N,d), and optionally `view_mask` (N,3 bool, ordered byte, timing, statistics). Statistics must be imputed and scaled with parameters fitted on the training partition only, and the splits should be group-disjoint.

    from triguard_etd import TriGuardETD

    model = TriGuardETD(profile="kd_balanced", seed=1, epochs=100, patience=12)
    model.fit(train_views, y_train, val_views, y_val)

    probabilities = model.predict_proba(test_views)
    predictions = model.predict(test_views)
    report = model.score(test_views, y_test)   # accuracy, macro/weighted F1, ECE, Brier, NLL

Two loss/optimizer profiles are provided: `kd_balanced` (used for USTC-TFC2016 and TII-SSRC-23) and `focal_weighted` (used for 5GAD-2022 with `n_thresholds=0`). A custom profile can be passed as a dictionary with the same keys. `predict_proba(views, return_gates=True)` also returns the per-sample view weights.

## Result

On 5GAD-2022, TriGuard-ETD reaches a macro-F1 of 0.815 under strict five-tuple grouping and 0.993 under temporal-session grouping, with all comparisons made on frozen, leakage-controlled partitions and matched seeds. On USTC-TFC2016 and TII-SSRC-23 it stays competitive with strong tabular learners, which obtain higher point estimates in feature-dominant settings.

## Citation

    @misc{massaoudi2026triguardetd,
      title  = {TriGuard-ETD: A Leakage-Controlled Multi-View Learning Framework for Malicious Encrypted-Traffic Detection},
      author = {Massaoudi, Mohamed and Ez Eddin, Maymouna and Refaat, Shady S. and Davis, Katherine R.},
      year   = {2026},
      note   = {Code: https://github.com/Mmassaoudi-cell/TriGuard-ETD}
    }

## License

MIT. See the `LICENSE` file.

## Contact

Mohamed Massaoudi, mmassaoudi@tarleton.edu
