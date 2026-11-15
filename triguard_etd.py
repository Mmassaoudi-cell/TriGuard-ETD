"""TriGuard-ETD: a compact tri-view detector for encrypted malicious traffic.

The model fuses three views of a network flow -- a 40x40 byte image, a 5x5
packet-timing sequence and a flow-statistics vector -- with an availability-
masked adaptive gate. It is trained with a composite objective (class-weighted
cross-entropy, class-balanced focal loss, supervised contrastive loss,
cross-view alignment, per-view auxiliary supervision) plus knowledge
distillation from a teacher ensemble of calibrated specialists
(LightGBM, XGBoost, ExtraTrees, RandomForest on the statistics view and a
logistic regression on the timing view).

Public interface::

    model = TriGuardETD(profile="kd_balanced", seed=1)
    model.fit(train_views, train_labels, val_views, val_labels)
    probabilities = model.predict_proba(test_views)
    report = model.score(test_views, test_labels)

``*_views`` is a dict with ``image`` (N,40,40) uint8, ``timing`` (N,5,5),
``statistics`` (N,d) and, optionally, ``view_mask`` (N,3) bool ordered
(byte, timing, statistics). Statistics must already be imputed and scaled with
parameters fitted on the training partition only.

Dependencies: numpy, scipy, scikit-learn, torch, lightgbm, xgboost.
"""

from __future__ import annotations

import copy
import random

import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

__all__ = ["TriGuardETD", "TriGuardNet", "PROFILES", "calibration_metrics"]

# Loss weights and optimiser settings. ``focal_weighted`` is the profile used
# for 5GAD-2022; ``kd_balanced`` is the profile used for USTC-TFC2016 and
# TII-SSRC-23. ``branch_kd`` weights an optional distillation term on the
# statistics-branch head (off in both profiles).
PROFILES = {
    "kd_balanced": {"class_power": 0.50, "ce": 1.0, "focal": 0.10, "supcon": 0.02,
                    "kd": 0.60, "branch_kd": 0.0, "align": 0.01, "aux": 0.25, "kd_temperature": 2.0,
                    "learning_rate": 8e-4, "weight_decay": 2e-4},
    "focal_weighted": {"class_power": 1.0, "ce": 0.60, "focal": 0.40, "supcon": 0.05,
                       "kd": 0.25, "branch_kd": 0.0, "align": 0.03, "aux": 0.15, "kd_temperature": 2.0,
                       "learning_rate": 1e-3, "weight_decay": 1e-4},
}


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class _SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                  nn.GELU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class _DSResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = _SqueezeExcite(channels)

    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.dw(x)))
        x = self.se(self.bn2(self.pw(x)))
        return F.gelu(x + residual)


class _ByteBranch(nn.Module):
    """Depthwise-separable residual CNN over the byte image."""

    def __init__(self, embedding_dim: int, channels: int = 40, depth: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.GELU())
        self.blocks = nn.Sequential(*[_DSResidualBlock(channels) for _ in range(depth)])
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(channels),
                                 nn.Linear(channels, embedding_dim))

    def forward(self, x):
        return self.out(self.blocks(self.stem(x)))


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.dw = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.pw(F.gelu(self.dw(x))).transpose(1, 2)
        return residual + self.dropout(x)


class _TimingBranch(nn.Module):
    """Depthwise temporal convolutions over the 5-step timing sequence."""

    def __init__(self, embedding_dim: int, input_dim: int = 5, channels: int = 48, depth: int = 2):
        super().__init__()
        self.project = nn.Linear(input_dim, channels)
        self.blocks = nn.Sequential(*[_TemporalBlock(channels) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, embedding_dim))

    def forward(self, x):
        return self.out(self.blocks(self.project(x)).mean(dim=1))


class _GatedResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.expand = nn.Linear(hidden, hidden * 2)
        self.contract = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        value, gate = self.expand(self.norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.contract(value * F.gelu(gate)))


class _StatisticsBranch(nn.Module):
    """Gated residual MLP over the statistics vector.

    Each feature is augmented with indicators ``x > c_k`` against per-feature
    cut points ``c_k`` (training-set quantiles); with zero cut points the
    branch consumes the raw features only.
    """

    def __init__(self, input_dim: int, embedding_dim: int, cut_points: torch.Tensor,
                 hidden: int = 192, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        if cut_points.ndim != 2 or cut_points.shape[0] != input_dim:
            raise ValueError("cut_points must have shape [n_features, n_thresholds]")
        self.register_buffer("cut_points", cut_points.float())
        expanded = input_dim * (1 + cut_points.shape[1])
        self.input = nn.Sequential(nn.LayerNorm(expanded), nn.Linear(expanded, hidden), nn.GELU())
        self.blocks = nn.Sequential(*[_GatedResidualBlock(hidden, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))

    def forward(self, x):
        thresholds = (x.unsqueeze(-1) > self.cut_points.unsqueeze(0)).to(x.dtype)
        expanded = torch.cat((x, thresholds.flatten(1)), dim=-1)
        return self.out(self.blocks(self.input(expanded)))


class _AdaptiveGate(nn.Module):
    """Sample-specific convex combination of the view embeddings; views whose
    mask entry is False receive zero weight."""

    def __init__(self, embedding_dim: int, hidden: int = 96):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(embedding_dim * 3, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, embeddings, mask):
        stacked = torch.stack(embeddings, dim=1)
        logits = self.network(torch.cat(embeddings, dim=-1))
        logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1), weights


class TriGuardNet(nn.Module):
    """Three-branch network with a masked adaptive gate and per-view heads."""

    def __init__(self, stats_dim: int, n_classes: int, stats_cut_points: torch.Tensor,
                 embedding_dim: int = 64, byte_channels: int = 40, byte_depth: int = 3,
                 temporal_depth: int = 2, stats_hidden: int = 192, stats_depth: int = 2,
                 dropout: float = 0.12):
        super().__init__()
        self.byte = _ByteBranch(embedding_dim, byte_channels, byte_depth)
        self.timing = _TimingBranch(embedding_dim, 5, 48, temporal_depth)
        self.statistics = _StatisticsBranch(stats_dim, embedding_dim, stats_cut_points,
                                            stats_hidden, stats_depth, dropout)
        self.gate = _AdaptiveGate(embedding_dim, hidden=96)
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Dropout(dropout),
                                        nn.Linear(embedding_dim, n_classes))
        self.branch_heads = nn.ModuleList([nn.Linear(embedding_dim, n_classes) for _ in range(3)])
        self.projection = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
                                        nn.Linear(embedding_dim, embedding_dim))

    def forward(self, image, timing, statistics, view_mask=None):
        if view_mask is None:
            view_mask = torch.ones(image.shape[0], 3, device=image.device, dtype=torch.bool)
        embeddings = (self.byte(image), self.timing(timing), self.statistics(statistics))
        fused, gates = self.gate(embeddings, view_mask)
        branch_logits = torch.stack([head(z) for head, z in zip(self.branch_heads, embeddings)], dim=1)
        logits = self.classifier(fused) + (gates.unsqueeze(-1) * branch_logits).sum(dim=1)
        return {"logits": logits, "embedding": fused,
                "projection": F.normalize(self.projection(fused), dim=-1),
                "byte_embedding": embeddings[0], "timing_embedding": embeddings[1],
                "statistics_embedding": embeddings[2], "gates": gates,
                "branch_logits": branch_logits}

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# Loss terms
# --------------------------------------------------------------------------

def _class_balanced_focal(logits, labels, class_weights, gamma: float = 2.0):
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt).pow(gamma) * ce).mean()


def _supervised_contrastive(features, labels, temperature: float = 0.15):
    if len(features) < 2:
        return features.new_zeros(())
    features = F.normalize(features, dim=-1)
    logits = features @ features.T / temperature
    diagonal = torch.eye(len(features), device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal, -1e9)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positives.any(dim=1)
    if not valid.any():
        return features.new_zeros(())
    return -((F.log_softmax(logits, dim=1) * positives.float()).sum(1) /
             positives.sum(1).clamp_min(1))[valid].mean()


def _distillation_kl(student_logits, teacher_probs, temperature: float = 2.0):
    return (temperature ** 2) * F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                                         teacher_probs.clamp_min(1e-8).log(),
                                         reduction="batchmean", log_target=True)


def _view_alignment(views, mask):
    losses = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            available = mask[:, i] & mask[:, j]
            if available.any():
                losses.append((1 - F.cosine_similarity(views[i][available], views[j][available], dim=-1)).mean())
    return torch.stack(losses).mean() if losses else views[0].new_zeros(())


# --------------------------------------------------------------------------
# Teacher ensemble with temperature calibration
# --------------------------------------------------------------------------

def _full_probabilities(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), n_classes), dtype=np.float64)
    out[:, model.classes_.astype(int)] = raw
    return np.clip(out, 1e-9, 1.0)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _fit_temperature(probabilities: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Temperature minimising validation NLL, searched on [0.25, 8]."""
    objective = lambda log_t: log_loss(y, _temperature_scale(probabilities, float(np.exp(log_t))),
                                       labels=np.arange(n_classes))
    result = minimize_scalar(objective, bounds=(np.log(0.25), np.log(8.0)), method="bounded")
    return float(np.exp(result.x))


def _macro_f1(y, pred, n_classes: int) -> float:
    return float(f1_score(y, pred, labels=np.arange(n_classes), average="macro", zero_division=0))


def _fit_teacher(xs_train, xs_val, xt_train, xt_val, y_train, y_val, n_classes: int, seed: int):
    """Fit the specialists on the training partition, calibrate each one's
    temperature on validation NLL, choose the ensemble weighting on validation,
    and return the ensemble posterior on the training rows."""
    models = {
        "lightgbm": lgb.LGBMClassifier(n_estimators=240, learning_rate=0.05, num_leaves=31,
                                       random_state=seed, verbosity=-1),
        "xgboost": xgb.XGBClassifier(n_estimators=240, learning_rate=0.05, max_depth=6,
                                     random_state=seed, eval_metric="mlogloss", verbosity=0,
                                     tree_method="hist", n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                                            class_weight="balanced", random_state=seed, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                class_weight="balanced_subsample",
                                                random_state=seed, n_jobs=-1),
    }
    train_probs, val_probs, temperatures, val_f1 = {}, {}, {}, {}
    for name, model in models.items():
        model.fit(xs_train, y_train)
        p_train = _full_probabilities(model, xs_train, n_classes)
        p_val = _full_probabilities(model, xs_val, n_classes)
        temperature = _fit_temperature(p_val, y_val, n_classes)
        train_probs[name] = _temperature_scale(p_train, temperature)
        val_probs[name] = _temperature_scale(p_val, temperature)
        temperatures[name] = temperature
        val_f1[name] = _macro_f1(y_val, p_val.argmax(1), n_classes)

    timing = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    timing.fit(xt_train, y_train)
    p_train = _full_probabilities(timing, xt_train, n_classes)
    p_val = _full_probabilities(timing, xt_val, n_classes)
    temperature = _fit_temperature(p_val, y_val, n_classes)
    train_probs["timing_logistic"] = _temperature_scale(p_train, temperature)
    val_probs["timing_logistic"] = _temperature_scale(p_val, temperature)
    temperatures["timing_logistic"] = temperature
    val_f1["timing_logistic"] = _macro_f1(y_val, p_val.argmax(1), n_classes)

    names = list(train_probs)
    val_stack = np.stack([val_probs[n] for n in names], axis=1)
    train_stack = np.stack([train_probs[n] for n in names], axis=1)
    uniform = np.full(len(names), 1.0 / len(names))
    nll = lambda w: log_loss(y_val, np.einsum("m,nmc->nc", w, val_stack), labels=np.arange(n_classes))
    stacked = minimize(nll, uniform, method="SLSQP", bounds=[(0.0, 1.0)] * len(names),
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                       options={"maxiter": 300, "ftol": 1e-10})
    candidates = {name: np.eye(len(names))[i] for i, name in enumerate(names)}
    candidates["uniform"] = uniform
    candidates["nll_stacking"] = stacked.x if stacked.success else uniform
    scores = {}
    for name, weights in candidates.items():
        p = np.einsum("m,nmc->nc", weights, val_stack)
        scores[name] = {"macro_f1": _macro_f1(y_val, p.argmax(1), n_classes),
                        "nll": float(log_loss(y_val, p, labels=np.arange(n_classes)))}
    selected = max(candidates, key=lambda n: (scores[n]["macro_f1"], -scores[n]["nll"]))
    teacher_train = np.einsum("m,nmc->nc", candidates[selected], train_stack).astype(np.float32)
    info = {"specialist_validation_macro_f1": val_f1, "calibration_temperature": temperatures,
            "candidate_validation_scores": scores, "selected_ensemble": selected,
            "selected_weights": dict(zip(names, candidates[selected].tolist()))}
    return teacher_train, info


# --------------------------------------------------------------------------
# Data handling and metrics
# --------------------------------------------------------------------------

class _Views(Dataset):
    def __init__(self, views: dict, labels: np.ndarray | None):
        image = np.asarray(views["image"])
        self.image = image[:, None] if image.ndim == 3 else image
        self.timing = np.asarray(views["timing"])
        self.statistics = np.asarray(views["statistics"])
        n = len(self.statistics)
        mask = views.get("view_mask")
        self.view_mask = np.ones((n, 3), dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_)
        self.labels = np.zeros(n, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.statistics)

    def __getitem__(self, i):
        return {"image": torch.from_numpy(self.image[i].astype(np.float32) / 255.0),
                "timing": torch.from_numpy(self.timing[i].astype(np.float32)),
                "statistics": torch.from_numpy(self.statistics[i].astype(np.float32)),
                "view_mask": torch.from_numpy(self.view_mask[i]),
                "label": torch.tensor(int(self.labels[i]), dtype=torch.long),
                "index": torch.tensor(i, dtype=torch.long)}


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> dict:
    """Top-label expected calibration error, Brier score and NLL."""
    confidences = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(np.float64)
    rows = np.arange(len(y_true))
    nll = float(-np.mean(np.log(np.clip(probabilities[rows, y_true], 1e-12, 1.0))))
    one_hot = np.zeros_like(probabilities)
    one_hot[rows, y_true] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidences, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return {"ece": float(ece), "brier_score": brier, "nll": nll}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _quantile_cut_points(x_train: np.ndarray, n_thresholds: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_thresholds + 2, dtype=np.float64)[1:-1]
    cuts = np.quantile(x_train.astype(np.float64), quantiles, axis=0).T
    return np.nan_to_num(cuts, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------

class TriGuardETD:
    """Fit, calibrate and score the TriGuard-ETD detector.

    Parameters
    ----------
    profile : "kd_balanced" or "focal_weighted", or a dict with the keys of
        ``PROFILES`` entries.
    epochs, batch_size, patience : maximum epochs, mini-batch size, and the
        number of epochs without validation macro-F1 improvement after which
        training stops; the best-validation weights are restored.
    n_thresholds : quantile cut points per statistics feature (0 disables).
    n_classes : number of classes; inferred from the labels when None.
    device : torch device string; CUDA is used when available.
    """

    def __init__(self, profile="kd_balanced", seed: int = 1, epochs: int = 100,
                 batch_size: int = 256, patience: int = 12, n_thresholds: int = 12,
                 n_classes: int | None = None, device: str | None = None):
        self.profile = dict(PROFILES[profile]) if isinstance(profile, str) else dict(profile)
        self.seed, self.epochs, self.batch_size = seed, epochs, batch_size
        self.patience, self.n_thresholds, self.n_classes = patience, n_thresholds, n_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.network: TriGuardNet | None = None
        self.teacher_info_: dict = {}
        self.history_: list[dict] = []

    # -- training ---------------------------------------------------------
    def fit(self, train_views: dict, y_train, val_views: dict, y_val) -> "TriGuardETD":
        y_train = np.asarray(y_train, dtype=np.int64)
        y_val = np.asarray(y_val, dtype=np.int64)
        n_classes = self.n_classes or int(max(y_train.max(), y_val.max())) + 1
        self.n_classes = n_classes
        profile, device = self.profile, self.device

        _set_seed(self.seed)
        stats_train = np.asarray(train_views["statistics"])
        cut_points = _quantile_cut_points(stats_train, self.n_thresholds)
        net = TriGuardNet(stats_train.shape[1], n_classes, torch.from_numpy(cut_points)).to(device)

        timing_train = np.asarray(train_views["timing"]).reshape(len(y_train), -1)
        timing_val = np.asarray(val_views["timing"]).reshape(len(y_val), -1)
        teacher_train, self.teacher_info_ = _fit_teacher(
            stats_train, np.asarray(val_views["statistics"]), timing_train, timing_val,
            y_train, y_val, n_classes, self.seed)

        train_loader = DataLoader(_Views(train_views, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(_Views(val_views, y_val), batch_size=self.batch_size * 2, shuffle=False)
        counts = np.bincount(y_train, minlength=n_classes)
        weights = np.power(counts.sum() / np.maximum(counts, 1), profile["class_power"])
        weights /= weights.mean()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=profile["learning_rate"],
                                      weight_decay=profile["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=profile["learning_rate"] * 0.1)

        best, best_state, no_improve = -1.0, None, 0
        self.history_ = []
        for epoch in range(self.epochs):
            net.train()
            for batch in train_loader:
                image = batch["image"].to(device)
                timing = batch["timing"].to(device)
                statistics = batch["statistics"].to(device)
                labels = batch["label"].to(device)
                mask = batch["view_mask"].to(device)
                teacher = torch.as_tensor(teacher_train[batch["index"].numpy()],
                                          dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                out = net(image, timing, statistics, mask)
                ce = F.cross_entropy(out["logits"], labels, weight=weights)
                focal = _class_balanced_focal(out["logits"], labels, weights)
                supcon = _supervised_contrastive(out["projection"], labels)
                align = _view_alignment([out["byte_embedding"], out["timing_embedding"],
                                         out["statistics_embedding"]], mask)
                kd = _distillation_kl(out["logits"], teacher, profile["kd_temperature"])
                per_view = [F.cross_entropy(out["branch_logits"][mask[:, v], v], labels[mask[:, v]], weight=weights)
                            for v in range(3) if mask[:, v].any()]
                aux = torch.stack(per_view).mean() if per_view else ce.new_zeros(())
                branch_kd = _distillation_kl(out["branch_logits"][:, 2], teacher, profile["kd_temperature"])
                loss = (profile["ce"] * ce + profile["focal"] * focal + profile["supcon"] * supcon
                        + profile["kd"] * kd + profile["branch_kd"] * branch_kd + profile["align"] * align + profile["aux"] * aux)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            probs = self._forward_probabilities(net, val_loader)
            score = _macro_f1(y_val, probs.argmax(1), n_classes)
            self.history_.append({"epoch": epoch + 1, "validation_macro_f1": score})
            if score > best + 1e-8:
                best, best_state, no_improve = score, copy.deepcopy(net.state_dict()), 0
                self.best_epoch_ = epoch + 1
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break
        net.load_state_dict(best_state)
        net.eval()
        self.network = net
        self.validation_macro_f1_ = best
        return self

    # -- inference --------------------------------------------------------
    def _forward_probabilities(self, net: TriGuardNet, loader: DataLoader, return_gates: bool = False):
        net.eval()
        probs, gates = [], []
        with torch.no_grad():
            for b in loader:
                out = net(b["image"].to(self.device), b["timing"].to(self.device),
                          b["statistics"].to(self.device), b["view_mask"].to(self.device))
                probs.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
                gates.append(out["gates"].cpu().numpy())
        probs = np.concatenate(probs)
        return (probs, np.concatenate(gates)) if return_gates else probs

    def predict_proba(self, views: dict, return_gates: bool = False):
        """Class posteriors (N, n_classes); optionally also the (N, 3) gate weights."""
        if self.network is None:
            raise RuntimeError("call fit() first")
        loader = DataLoader(_Views(views, None), batch_size=self.batch_size * 2, shuffle=False)
        return self._forward_probabilities(self.network, loader, return_gates)

    def predict(self, views: dict) -> np.ndarray:
        return self.predict_proba(views).argmax(axis=1)

    def score(self, views: dict, y) -> dict:
        """Accuracy, macro/weighted F1 and calibration (ECE, Brier, NLL)."""
        y = np.asarray(y, dtype=np.int64)
        probabilities = self.predict_proba(views)
        pred = probabilities.argmax(axis=1)
        labels = np.arange(self.n_classes)
        return {"accuracy": float(accuracy_score(y, pred)),
                "macro_f1": _macro_f1(y, pred, self.n_classes),
                "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
                **calibration_metrics(y, probabilities)}

    @property
    def parameter_count(self) -> int:
        return self.network.parameter_count
