"""DeepLabV3-ResNet50 for binary segmentation."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3ResNet50Binary(nn.Module):
    """DeepLabV3-ResNet50 for binary retinal vessel segmentation.

    Input:
        images: [B, 3, H, W]

    Output:
        logits: [B, 1, H, W]

    Note:
        Output is logits without sigmoid. Apply sigmoid only for inference
        or visualization.
    """

    def __init__(self, pretrained_backbone: bool = True) -> None:
        super().__init__()

        # Use ImageNet-pretrained backbone for transfer learning.
        weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None

        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            aux_loss=False,
        )

        # Replace classifier head to output a single channel for binary masks.
        self.model.classifier = DeepLabHead(
            in_channels=2048,
            num_classes=1,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        out = self.model(x)
        logits = out["out"]

        # Ensure output spatial size matches input.
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits
