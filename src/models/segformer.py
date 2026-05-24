"""SegFormer-B0 for binary segmentation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError as exc:
    raise ImportError(
        "SegFormerB0 requires transformers. Install with `pip install transformers`."
    ) from exc


class SegFormerB0(nn.Module):
    """SegFormer-B0 wrapper with a binary segmentation head."""

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels: int = 1,
        ignore_mismatched_sizes: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            config = SegformerConfig.from_pretrained(pretrained_model_name)
            config.num_labels = num_labels
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
        else:
            config = SegformerConfig(num_labels=num_labels)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected input [B, 3, H, W], got shape {tuple(x.shape)}."
            )
        if x.size(1) != 3:
            raise ValueError(
                f"Expected 3-channel input, got shape {tuple(x.shape)}."
            )

        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if logits.size(1) != 1:
            raise ValueError(
                f"Expected 1-channel logits, got shape {tuple(logits.shape)}."
            )

        return logits


def check_io_shapes(
    model: nn.Module,
    batch_size: int = 2,
    height: int = 512,
    width: int = 512,
    device: Optional[torch.device] = None,
) -> torch.Size:
    """Run a quick forward pass to validate input/output shapes."""
    device = device or torch.device("cpu")
    model = model.to(device)

    x = torch.randn(batch_size, 3, height, width, device=device)
    with torch.no_grad():
        logits = model(x)

    expected = (batch_size, 1, height, width)
    if tuple(logits.shape) != expected:
        raise ValueError(f"Expected logits shape {expected}, got {tuple(logits.shape)}.")

    return logits.shape


if __name__ == "__main__":
    model = SegFormerB0(pretrained=False)
    shape = check_io_shapes(model, batch_size=1, height=256, width=256)
    print("OK", shape)
