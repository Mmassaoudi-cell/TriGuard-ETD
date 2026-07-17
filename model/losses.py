"""ntXent (Normalized Temperature-scaled Cross Entropy) contrastive loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss as used in LitCVit pre-training (Equation 1 of paper).

    Given a mini-batch of B samples, for each sample i the positive pair is
    the augmented view z_i+, and all other 2(B-1) representations are negatives.

    Args:
        temperature: τ controls smoothness of the similarity distribution.
                     Paper: τ = 0.3 (best from ablation, Table VIII).
    """

    def __init__(self, temperature: float = 0.3):
        super().__init__()
        self.tau = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (B, d)  embeddings of original views (already L2-normalised)
            z_j: (B, d)  embeddings of augmented views (already L2-normalised)
        Returns:
            scalar loss
        """
        B = z_i.size(0)
        # Concatenate both views: (2B, d)
        z = torch.cat([z_i, z_j], dim=0)   # (2B, d)

        # Pairwise cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.tau  # (2B, 2B)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        # Positive pair indices: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0,     B, device=z.device),
        ])  # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (optional alternative for fine-tuning warm-up).
    Uses ground-truth labels to define positive pairs.
    """

    def __init__(self, temperature: float = 0.3):
        super().__init__()
        self.tau = temperature

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        B = features.size(0)
        sim = torch.mm(features, features.t()) / self.tau
        mask_self = torch.eye(B, device=features.device).bool()

        # Positive mask: same label, different index
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()) & ~mask_self

        sim.masked_fill_(mask_self, float('-inf'))

        log_prob = F.log_softmax(sim, dim=1)
        # Mean over positive pairs
        pos_sum  = (log_prob * pos_mask.float()).sum(1)
        n_pos    = pos_mask.float().sum(1).clamp(min=1)
        loss     = -(pos_sum / n_pos).mean()
        return loss
