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


class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced binary segmentation using logits."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).contiguous().view(logits.size(0), -1)
        targets = targets.float().contiguous().view(targets.size(0), -1)

        true_positive = (probs * targets).sum(dim=1)
        false_positive = (probs * (1.0 - targets)).sum(dim=1)
        false_negative = ((1.0 - probs) * targets).sum(dim=1)

        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return (1.0 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss that emphasizes hard vessel pixels."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky(logits, targets).pow(self.gamma)


class BCETverskyLoss(nn.Module):
    """BCE + Tversky loss for thin retinal vessel segmentation."""

    def __init__(
        self,
        bce_weight: float = 0.3,
        tversky_weight: float = 0.7,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = self.bce(logits, targets)
        tversky_loss = self.tversky(logits, targets)
        return self.bce_weight * bce_loss + self.tversky_weight * tversky_loss


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


def build_vessel_loss(
    bce_weight: float = 0.3,
    tversky_weight: float = 0.7,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
    pos_weight: torch.Tensor | None = None,
) -> BCETverskyLoss:
    """Factory for the recommended DeepLabV3+ vessel loss."""
    return BCETverskyLoss(
        bce_weight=bce_weight,
        tversky_weight=tversky_weight,
        alpha=alpha,
        beta=beta,
        smooth=smooth,
        pos_weight=pos_weight,
    )
