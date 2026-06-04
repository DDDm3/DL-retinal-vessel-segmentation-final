"""DeepLabV3+-ResNet50 for binary retinal vessel segmentation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import ASPP


class SeparableConv2d(nn.Sequential):
    """Depthwise separable convolution used in the DeepLabV3+ decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DeepLabV3PlusHead(nn.Module):
    """ASPP head plus low-level feature decoder from DeepLabV3+."""

    def __init__(
        self,
        high_channels: int = 2048,
        low_channels: int = 256,
        low_project_channels: int = 48,
        decoder_channels: int = 256,
        num_classes: int = 1,
        atrous_rates: tuple[int, int, int] = (6, 12, 18),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aspp = ASPP(high_channels, atrous_rates, out_channels=decoder_channels)
        self.low_project = nn.Sequential(
            nn.Conv2d(low_channels, low_project_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_project_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            SeparableConv2d(decoder_channels + low_project_channels, decoder_channels),
            SeparableConv2d(decoder_channels, decoder_channels),
            nn.Dropout(dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        low = features["low"]
        high = features["out"]

        high = self.aspp(high)
        high = F.interpolate(
            high,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low = self.low_project(low)
        return self.decoder(torch.cat([high, low], dim=1))


class DeepLabV3PlusResNet50Binary(nn.Module):
    """DeepLabV3+-ResNet50 CNN for binary vessel segmentation.

    Input:
        images: [B, 3, H, W]

    Output:
        logits: [B, 1, H, W]

    The model returns logits. Use ``torch.sigmoid(logits)`` to get the
    probability map for threshold tuning, RF refinement, or visualization.
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        output_stride: int = 16,
        num_classes: int = 1,
        atrous_rates: Optional[tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()

        if output_stride == 16:
            replace_stride_with_dilation = (False, False, True)
            atrous_rates = atrous_rates or (6, 12, 18)
        elif output_stride == 8:
            replace_stride_with_dilation = (False, True, True)
            atrous_rates = atrous_rates or (12, 24, 36)
        else:
            raise ValueError("output_stride must be 8 or 16.")

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        self.backbone = IntermediateLayerGetter(
            backbone,
            return_layers={"layer1": "low", "layer4": "out"},
        )
        self.classifier = DeepLabV3PlusHead(
            num_classes=num_classes,
            atrous_rates=atrous_rates,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input [B, 3, H, W], got {tuple(x.shape)}.")
        if x.size(1) != 3:
            raise ValueError(f"Expected 3-channel input, got {tuple(x.shape)}.")

        input_size = x.shape[-2:]
        logits = self.classifier(self.backbone(x))
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return vessel probability maps [B, 1, H, W]."""
        return torch.sigmoid(self.forward(x))

