"""Visualize SegFormer-B0 predictions against retinal vessel ground truth."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGFORMER_PATH = PROJECT_ROOT / "src" / "models" / "segformer.py"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Save retinal vessel segmentation prediction visualizations."
    )
    parser.add_argument(
        "--model",
        default="segformer_b0",
        choices=("segformer_b0",),
        help="Model architecture to visualize.",
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
        "--threshold",
        default=0.15,
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
        "--output-dir",
        default=Path("outputs/predictions"),
        type=Path,
        help="Directory where PNG visualizations will be saved.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a torch device string.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve auto/cpu/cuda into a torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_segformer_b0_class() -> type[nn.Module]:
    """Load SegFormerB0 directly from src/models/segformer.py."""
    spec = importlib.util.spec_from_file_location("segformer_module", SEGFORMER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SegFormer module from {SEGFORMER_PATH}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    try:
        return module.SegFormerB0
    except AttributeError as exc:
        raise ImportError("Expected SegFormerB0 in src/models/segformer.py.") from exc


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict or a dict containing "
        "'state_dict', 'model_state_dict', or 'model'."
    )


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remove a checkpoint key prefix when present."""
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def add_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Add a checkpoint key prefix when absent."""
    return {
        key if key.startswith(prefix) else f"{prefix}{key}": value
        for key, value in state_dict.items()
    }


def load_state_dict_robust(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load a checkpoint while handling common wrapper prefixes."""
    candidates = [
        state_dict,
        strip_prefix(state_dict, "module."),
        strip_prefix(state_dict, "_orig_mod."),
        add_prefix(state_dict, "model."),
    ]

    errors: List[str] = []
    for candidate in candidates:
        try:
            model.load_state_dict(candidate)
            return
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        "Could not load checkpoint into SegFormer-B0. "
        f"Last load error: {errors[-1]}"
    )


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Instantiate SegFormer-B0 and load checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    segformer_b0 = load_segformer_b0_class()
    model = segformer_b0(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_state_dict_robust(model, extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def as_tensor(value: Any, name: str) -> torch.Tensor:
    """Convert a dataset value to a float tensor."""
    if isinstance(value, torch.Tensor):
        return value.float()

    try:
        return torch.as_tensor(value).float()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not convert {name} to a torch.Tensor.") from exc


def image_to_chw(image: torch.Tensor) -> torch.Tensor:
    """Normalize one image tensor to [3, H, W]."""
    if image.dim() == 2:
        image = image.unsqueeze(0)
    elif image.dim() == 3 and image.shape[-1] in (1, 3, 4):
        image = image.permute(2, 0, 1)

    if image.dim() != 3:
        raise ValueError(
            f"Each image must have shape [C, H, W] or [H, W, C], got {tuple(image.shape)}."
        )

    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    elif image.size(0) == 4:
        image = image[:3]
    elif image.size(0) != 3:
        raise ValueError(f"Expected image to have 1, 3, or 4 channels, got {image.size(0)}.")

    return image.contiguous()


def mask_to_hw(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize one mask-like tensor to [H, W]."""
    if mask.dim() == 3 and mask.shape[-1] == 1:
        mask = mask.permute(2, 0, 1)
    if mask.dim() == 3 and mask.size(0) == 1:
        mask = mask.squeeze(0)

    if mask.dim() != 2:
        raise ValueError(f"{name} must have shape [1, H, W] or [H, W], got {tuple(mask.shape)}.")

    return mask.contiguous()


def normalize_batched_images(images: Any) -> torch.Tensor:
    """Normalize images to [B, 3, H, W]."""
    image_tensor = as_tensor(images, "images")

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.dim() != 4:
        raise ValueError(
            f"Images must have shape [B, C, H, W] or [B, H, W, C], got {tuple(image_tensor.shape)}."
        )

    return torch.stack([image_to_chw(image) for image in image_tensor])


def normalize_batched_masks(masks: Any) -> torch.Tensor:
    """Normalize masks to [B, H, W]."""
    mask_tensor = as_tensor(masks, "masks")

    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if mask_tensor.dim() == 3:
        return torch.stack([mask_to_hw(mask, "mask") for mask in mask_tensor])
    if mask_tensor.dim() == 4:
        return torch.stack([mask_to_hw(mask, "mask") for mask in mask_tensor])

    raise ValueError(
        f"Masks must have shape [B, 1, H, W] or [B, H, W], got {tuple(mask_tensor.shape)}."
    )


def load_dataset(data_path: Path) -> Dict[str, torch.Tensor]:
    """Load a .pt dataset dict with images and masks."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data: Any = torch.load(data_path, map_location="cpu")
    if not isinstance(data, dict) or "images" not in data or "masks" not in data:
        raise ValueError("Expected dataset .pt to be a dict with 'images' and 'masks'.")

    images = normalize_batched_images(data["images"])
    masks = normalize_batched_masks(data["masks"])

    if images.size(0) != masks.size(0):
        raise ValueError(
            "Images and masks must have the same number of samples, got "
            f"{images.size(0)} and {masks.size(0)}."
        )
    if images.shape[-2:] != masks.shape[-2:]:
        raise ValueError(
            "Images and masks must have matching spatial dimensions, got "
            f"{tuple(images.shape[-2:])} and {tuple(masks.shape[-2:])}."
        )

    return {"images": images, "masks": masks}


def probability_from_output(output: torch.Tensor) -> torch.Tensor:
    """Convert model output logits to a vessel probability map [B, H, W]."""
    if output.dim() == 3:
        return torch.sigmoid(output)
    if output.dim() != 4:
        raise ValueError(f"Expected model output [B, C, H, W] or [B, H, W], got {tuple(output.shape)}.")

    if output.size(1) == 1:
        return torch.sigmoid(output[:, 0])
    if output.size(1) >= 2:
        return torch.softmax(output, dim=1)[:, 1]

    raise ValueError(f"Expected at least one output channel, got {tuple(output.shape)}.")


def image_for_display(image: torch.Tensor) -> np.ndarray:
    """Convert one [3, H, W] tensor to a displayable [H, W, 3] array."""
    display = image.detach().cpu()
    if float(display.min()) < 0.0:
        display = display * IMAGENET_STD + IMAGENET_MEAN
    elif float(display.max()) > 1.0:
        display = display / 255.0

    return display.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def mask_for_display(mask: torch.Tensor) -> np.ndarray:
    """Convert a mask tensor to a [0, 1] numpy array."""
    display = mask.detach().cpu().float()
    if float(display.max()) > 1.0:
        display = display / 255.0
    return display.clamp(0.0, 1.0).numpy()


def make_overlay(image: np.ndarray, binary_prediction: np.ndarray) -> np.ndarray:
    """Draw predicted vessels over the original image in readable red."""
    overlay = image.copy()
    vessel_pixels = binary_prediction >= 0.5
    red = np.array([1.0, 0.05, 0.02], dtype=overlay.dtype)
    overlay[vessel_pixels] = (0.35 * overlay[vessel_pixels]) + (0.65 * red)
    return np.clip(overlay, 0.0, 1.0)


def save_visualization(
    image: torch.Tensor,
    mask: torch.Tensor,
    binary_prediction: torch.Tensor,
    output_path: Path,
) -> None:
    """Save one four-panel prediction visualization."""
    display_image = image_for_display(image)
    display_mask = mask_for_display(mask)
    display_prediction = mask_for_display(binary_prediction)
    display_overlay = make_overlay(display_image, display_prediction)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))

    panels = (
        ("Original", display_image, None),
        ("Ground Truth", display_mask, "gray"),
        ("Prediction", display_prediction, "gray"),
        ("Overlay", display_overlay, None),
    )

    for axis, (title, panel, cmap) in zip(axes, panels):
        if cmap is None:
            axis.imshow(panel)
        else:
            axis.imshow(panel, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def visualize_predictions(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    output_dir: Path,
    threshold: float,
    num_samples: int,
    device: torch.device,
) -> List[Path]:
    """Run inference and save prediction visualization PNG files."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")
    if num_samples <= 0:
        raise ValueError(f"num-samples must be positive, got {num_samples}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(num_samples, images.size(0))
    saved_paths: List[Path] = []

    model.eval()
    with torch.no_grad():
        for index in range(sample_count):
            image = images[index]
            mask = masks[index]

            output = model(image.unsqueeze(0).to(device))
            probability = probability_from_output(output).squeeze(0).cpu()
            binary_prediction = (probability >= threshold).float()

            output_path = output_dir / f"sample_{index:03d}.png"
            save_visualization(
                image=image,
                mask=mask,
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
