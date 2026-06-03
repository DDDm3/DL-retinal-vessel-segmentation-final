"""Visualize SegFormer-B0 predictions against ground-truth masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import extract_state_dict, load_state_dict_robust, resolve_device  # noqa: E402
from src.models import SegFormerB0  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Save SegFormer-B0 prediction visualizations for debugging."
    )
    parser.add_argument(
        "--checkpoint",
        default=Path("src/models/best_segformer_b0.pth"),
        type=Path,
        help="Path to the SegFormer-B0 checkpoint.",
    )
    parser.add_argument(
        "--data",
        default=Path("dataset/drive_test_dataset.pt"),
        type=Path,
        help="Path to the exported .pt test dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/predictions"),
        type=Path,
        help="Directory where PNG visualizations will be saved.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Threshold used to binarize predicted probabilities.",
    )
    parser.add_argument(
        "--num-samples",
        default=5,
        type=int,
        help="Number of dataset samples to visualize.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a torch device string.",
    )

    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> SegFormerB0:
    """Load SegFormer-B0 and its checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = SegFormerB0(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    load_state_dict_robust(model, state_dict)

    model.to(device)
    model.eval()

    return model


def load_dataset(data_path: Path) -> Dict[str, torch.Tensor]:
    """Load an exported dataset dict with images and masks tensors."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data: Any = torch.load(data_path, map_location="cpu")
    if not isinstance(data, dict) or "images" not in data or "masks" not in data:
        raise ValueError("Expected dataset .pt to be a dict with 'images' and 'masks'.")

    images = data["images"].float()
    masks = data["masks"].float()

    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError(f"Expected images shape [B, 3, H, W], got {tuple(images.shape)}.")
    if masks.dim() != 4 or masks.size(1) != 1:
        raise ValueError(f"Expected masks shape [B, 1, H, W], got {tuple(masks.shape)}.")
    if images.size(0) != masks.size(0):
        raise ValueError(
            "Images and masks must have the same number of samples, got "
            f"{images.size(0)} and {masks.size(0)}."
        )

    return {"images": images, "masks": masks}


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Convert an ImageNet-normalized image tensor back to [0, 1]."""
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0.0, 1.0)


def save_visualization(
    image: torch.Tensor,
    mask: torch.Tensor,
    probability: torch.Tensor,
    binary_prediction: torch.Tensor,
    output_path: Path,
) -> None:
    """Save one four-panel prediction visualization."""
    display_image = denormalize_image(image).permute(1, 2, 0).numpy()
    display_mask = mask.detach().cpu().squeeze(0).numpy()
    display_probability = probability.detach().cpu().squeeze(0).numpy()
    display_binary = binary_prediction.detach().cpu().squeeze(0).numpy()

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(display_image)
    axes[0].set_title("Original Image")

    axes[1].imshow(display_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")

    probability_plot = axes[2].imshow(display_probability, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Predicted Probability")
    figure.colorbar(probability_plot, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(display_binary, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Binary Prediction")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def visualize_predictions(
    model: SegFormerB0,
    images: torch.Tensor,
    masks: torch.Tensor,
    output_dir: Path,
    threshold: float,
    num_samples: int,
    device: torch.device,
) -> List[Path]:
    """Run prediction and save visualization PNG files."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(num_samples, images.size(0))
    saved_paths: List[Path] = []

    with torch.no_grad():
        for index in range(sample_count):
            image = images[index]
            mask = masks[index]

            logits = model(image.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits).squeeze(0)
            binary_prediction = (probability >= threshold).float()

            output_path = output_dir / f"prediction_{index:03d}.png"
            save_visualization(
                image=image,
                mask=mask,
                probability=probability,
                binary_prediction=binary_prediction,
                output_path=output_path,
            )
            saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    """Generate prediction visualization images."""
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = load_dataset(args.data)

    saved_paths = visualize_predictions(
        model=model,
        images=dataset["images"],
        masks=dataset["masks"],
        output_dir=args.output_dir,
        threshold=args.threshold,
        num_samples=args.num_samples,
        device=device,
    )

    print("Saved prediction visualizations:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
