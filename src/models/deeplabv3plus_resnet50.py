"""DeepLabV3+-ResNet50 wrapper from segmentation-models-pytorch."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "DeepLabV3PlusResNet50Binary requires segmentation-models-pytorch. "
        "Install dependencies with: python -m pip install -r requirements.txt"
    ) from exc


class DeepLabV3PlusResNet50Binary(nn.Module):
    """DeepLabV3+-ResNet50 CNN for binary retinal vessel segmentation.

    This class intentionally calls the library implementation instead of
    reimplementing DeepLabV3+ by hand:

    - model: ``smp.DeepLabV3Plus``
    - encoder: ResNet50
    - encoder pretraining: ImageNet when ``pretrained_backbone=True``
    - output: one-channel logits for binary vessel segmentation

    Input:
        images: [B, 3, H, W]

    Output:
        logits: [B, 1, H, W]

    The model returns logits. Use ``torch.sigmoid(logits)`` to obtain the
    vessel probability map before threshold tuning or visualization.
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        output_stride: int = 16,
        num_classes: int = 1,
        atrous_rates: Optional[tuple[int, int, int]] = None,
        decoder_channels: int = 256,
        safe_small_batch: bool = True,
    ) -> None:
        super().__init__()

        if output_stride not in (8, 16):
            raise ValueError("output_stride must be 8 or 16.")

        decoder_atrous_rates = atrous_rates
        if decoder_atrous_rates is None:
            decoder_atrous_rates = (12, 24, 36) if output_stride == 8 else (6, 12, 18)

        encoder_weights = "imagenet" if pretrained_backbone else None
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            encoder_output_stride=output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        if safe_small_batch:
            self._make_aspp_pooling_branch_small_batch_safe()

    def _make_aspp_pooling_branch_small_batch_safe(self) -> None:
        """Avoid BatchNorm on the 1x1 ASPP image-pooling tensor.

        SMP's DeepLabV3+ follows the standard ASPP pooling branch with
        BatchNorm after global average pooling. That is fine for regular batch
        sizes, but it raises an error when each device sees one image during
        training because the pooled tensor is [1, C, 1, 1]. Replacing only that
        BatchNorm with Identity keeps the library model structure while making
        small-batch training and notebook sanity checks reliable.
        """
        pooling_branch = self.model.decoder.aspp[0].convs[4]
        if isinstance(pooling_branch[2], nn.BatchNorm2d):
            pooling_branch[2] = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input [B, 3, H, W], got {tuple(x.shape)}.")
        if x.size(1) != 3:
            raise ValueError(f"Expected 3-channel input, got {tuple(x.shape)}.")

        return self.model(x)

    @property
    def backbone(self) -> nn.Module:
        """Compatibility alias for existing optimizer/freeze code."""
        return self.model.encoder

    @property
    def classifier(self) -> nn.Module:
        """Compatibility alias for decoder/head parameter groups."""
        return nn.ModuleList([self.model.decoder, self.model.segmentation_head])

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return vessel probability maps [B, 1, H, W]."""
        return torch.sigmoid(self.forward(x))
