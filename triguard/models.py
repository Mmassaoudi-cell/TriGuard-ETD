"""Compact three-view network used by TriGuard-ETD."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                  nn.GELU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class DSResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.bn1(self.dw(x)))
        x = self.se(self.bn2(self.pw(x)))
        return F.gelu(x + residual)


class ByteBranch(nn.Module):
    """Depthwise separable ConvMixer-style image encoder."""
    def __init__(self, embedding_dim: int = 64, channels: int = 40, depth: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.GELU())
        self.blocks = nn.Sequential(*[DSResidualBlock(channels) for _ in range(depth)])
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(channels),
                                 nn.Linear(channels, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.stem(x)))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.dw = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.pw(F.gelu(self.dw(x))).transpose(1, 2)
        return residual + self.dropout(x)


class TimingBranch(nn.Module):
    def __init__(self, input_dim: int = 5, embedding_dim: int = 64, channels: int = 48, depth: int = 2):
        super().__init__()
        self.project = nn.Linear(input_dim, channels)
        self.blocks = nn.Sequential(*[TemporalBlock(channels) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.project(x)).mean(dim=1))


class StatisticsBranch(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden: int = 96, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveGate(nn.Module):
    """Masked, sample-specific convex fusion of three view embeddings."""
    def __init__(self, embedding_dim: int, hidden: int = 96):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(embedding_dim * 3, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor], mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(embeddings, dim=1)
        logits = self.network(torch.cat(embeddings, dim=-1))
        logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1), weights


class TriGuardETD(nn.Module):
    """Tri-view lightweight detector with an optional missing-view mask."""
    def __init__(self, stats_dim: int, n_classes: int, embedding_dim: int = 64,
                 byte_channels: int = 40, byte_depth: int = 3, temporal_depth: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.byte = ByteBranch(embedding_dim, byte_channels, byte_depth)
        self.timing = TimingBranch(5, embedding_dim, 48, temporal_depth)
        self.statistics = StatisticsBranch(stats_dim, embedding_dim, 96, dropout)
        self.gate = AdaptiveGate(embedding_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Dropout(dropout), nn.Linear(embedding_dim, n_classes))
        self.projection = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim))

    def forward(self, image: torch.Tensor, timing: torch.Tensor, statistics: torch.Tensor,
                view_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch = image.shape[0]
        if view_mask is None:
            view_mask = torch.ones(batch, 3, device=image.device, dtype=torch.bool)
        z_b, z_t, z_s = self.byte(image), self.timing(timing), self.statistics(statistics)
        fused, gates = self.gate((z_b, z_t, z_s), view_mask)
        return {"logits": self.classifier(fused), "embedding": fused, "projection": F.normalize(self.projection(fused), dim=-1),
                "byte_embedding": z_b, "timing_embedding": z_t, "statistics_embedding": z_s, "gates": gates}

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
