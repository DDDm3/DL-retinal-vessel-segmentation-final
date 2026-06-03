"""Evaluation metrics for binary retinal vessel segmentation."""

from __future__ import annotations

from typing import Dict

import torch


EPSILON = 1e-7


def _prepare_binary_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert predictions and targets to flattened binary tensors."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Expected pred and target to have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}."
        )
    if pred.dim() != 4 or pred.size(1) != 1:
        raise ValueError(
            f"Expected tensors with shape [B, 1, H, W], got {tuple(pred.shape)}."
        )

    pred = pred.detach().float()
    target = target.detach().float()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    pred_flat = pred_binary.contiguous().view(pred_binary.size(0), -1)
    target_flat = target_binary.contiguous().view(target_binary.size(0), -1)

    return pred_flat, target_flat


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute mean Dice score for a batch of binary segmentation masks."""
    pred_binary, target_binary = _prepare_binary_tensors(pred, target, threshold)

    intersection = (pred_binary * target_binary).sum(dim=1)
    denominator = pred_binary.sum(dim=1) + target_binary.sum(dim=1)
    dice = (2.0 * intersection + EPSILON) / (denominator + EPSILON)

    return float(dice.mean().item())


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute mean intersection-over-union score for a batch."""
    pred_binary, target_binary = _prepare_binary_tensors(pred, target, threshold)

    intersection = (pred_binary * target_binary).sum(dim=1)
    union = pred_binary.sum(dim=1) + target_binary.sum(dim=1) - intersection
    iou = (intersection + EPSILON) / (union + EPSILON)

    return float(iou.mean().item())


def accuracy_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute mean pixel accuracy for a batch."""
    pred_binary, target_binary = _prepare_binary_tensors(pred, target, threshold)

    correct = (pred_binary == target_binary).float().mean(dim=1)

    return float(correct.mean().item())


def precision_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute mean precision for a batch of binary segmentation masks."""
    pred_binary, target_binary = _prepare_binary_tensors(pred, target, threshold)

    true_positive = (pred_binary * target_binary).sum(dim=1)
    false_positive = (pred_binary * (1.0 - target_binary)).sum(dim=1)
    precision = (true_positive + EPSILON) / (
        true_positive + false_positive + EPSILON
    )

    return float(precision.mean().item())


def recall_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute mean recall for a batch of binary segmentation masks."""
    pred_binary, target_binary = _prepare_binary_tensors(pred, target, threshold)

    true_positive = (pred_binary * target_binary).sum(dim=1)
    false_negative = ((1.0 - pred_binary) * target_binary).sum(dim=1)
    recall = (true_positive + EPSILON) / (
        true_positive + false_negative + EPSILON
    )

    return float(recall.mean().item())


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all binary segmentation metrics for a batch."""
    return {
        "dice": dice_score(pred, target, threshold),
        "iou": iou_score(pred, target, threshold),
        "accuracy": accuracy_score(pred, target, threshold),
        "precision": precision_score(pred, target, threshold),
        "recall": recall_score(pred, target, threshold),
    }


if __name__ == "__main__":
    logits = torch.randn(2, 1, 256, 256)
    masks = torch.randint(0, 2, (2, 1, 256, 256)).float()

    metrics = compute_metrics(logits, masks)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
