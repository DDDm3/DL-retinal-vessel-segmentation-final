"""Loss functions for segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation using logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCEWithLogitsLoss and DiceLoss for binary segmentation."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


def build_loss(
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
) -> BCEDiceLoss:
    """Factory for the default segmentation loss."""
    return BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        smooth=smooth,
    )
