"""TriGuard loss components; every term is optional and independently logged."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def class_balanced_focal(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None,
                         gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt).pow(gamma) * ce).mean()


def supervised_contrastive(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
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


def distillation_kl(student_logits: torch.Tensor, teacher_probs: torch.Tensor | None, temperature: float = 2.0) -> torch.Tensor:
    if teacher_probs is None:
        return student_logits.new_zeros(())
    return (temperature ** 2) * F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                                         teacher_probs.clamp_min(1e-8).log(), reduction="batchmean", log_target=True)


def view_alignment(views: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            available = mask[:, i] & mask[:, j]
            if available.any():
                losses.append((1 - F.cosine_similarity(views[i][available], views[j][available], dim=-1)).mean())
    return torch.stack(losses).mean() if losses else views[0].new_zeros(())
