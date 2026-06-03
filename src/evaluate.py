"""Evaluation entry point for binary retinal vessel segmentation models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import AttentionUNet, SegFormerB0, UNet  # noqa: E402
from src.utils.metrics import dice_score, iou_score  # noqa: E402


TensorPair = Tuple[torch.Tensor, torch.Tensor]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation models for retinal vessel segmentation."
    )
    parser.add_argument(
        "--model",
        choices=("unet", "attention_unet", "segformer_b0"),
        required=True,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a .pth checkpoint file.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to a .pt test dataset.",
    )
    parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Threshold used to binarize predictions.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a torch device string.",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="DataLoader worker count. Default 0 is Windows-friendly.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a user-provided device argument to a torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def build_model(model_name: str) -> nn.Module:
    """Instantiate the selected model and validate that it is runnable."""
    if model_name == "unet":
        # UNet evaluation requires a trained checkpoint, which is not currently
        # included in the repository.
        model = UNet()
    elif model_name == "attention_unet":
        # AttentionUNet evaluation requires a trained checkpoint, which is not
        # currently included in the repository.
        model = AttentionUNet()
    elif model_name == "segformer_b0":
        model = SegFormerB0(pretrained=False)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    if not isinstance(model, nn.Module):
        raise NotImplementedError(
            f"{model.__class__.__name__} is not implemented as torch.nn.Module yet. "
            "Complete the model class before running evaluation."
        )
    if type(model).forward is nn.Module.forward:
        raise NotImplementedError(
            f"{model.__class__.__name__}.forward() is not implemented yet. "
            "Complete the model forward pass before running evaluation."
        )

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load model weights from a checkpoint file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    load_state_dict_robust(model, state_dict)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict or a dict "
        "containing 'state_dict', 'model_state_dict', or 'model'."
    )


def load_state_dict_robust(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Load a state_dict while handling common wrapper key prefixes."""
    candidate_state_dicts = [
        state_dict,
        strip_prefix_from_state_dict(state_dict, "module."),
        strip_prefix_from_state_dict(state_dict, "_orig_mod."),
        add_prefix_to_state_dict(state_dict, "model."),
    ]

    errors: List[str] = []
    for candidate in candidate_state_dicts:
        try:
            model.load_state_dict(candidate)
            return
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        "Could not load checkpoint into the selected model. The checkpoint may "
        "belong to a different architecture or use incompatible layer names. "
        f"Last load error: {errors[-1]}"
    )


def strip_prefix_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Remove a prefix from all state_dict keys when present."""
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def add_prefix_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Add a prefix to keys that do not already have it."""
    return {
        key if key.startswith(prefix) else f"{prefix}{key}": value
        for key, value in state_dict.items()
    }


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    """Convert a dataset value to a torch tensor."""
    if isinstance(value, torch.Tensor):
        return value

    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not convert {name} to a torch.Tensor.") from exc


def _normalize_image_mask_pair(image: Any, mask: Any) -> TensorPair:
    """Convert one image/mask pair to float tensors with channel dimensions."""
    image_tensor = _as_tensor(image, "image").float()
    mask_tensor = _as_tensor(mask, "mask").float()

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)

    if image_tensor.dim() != 3:
        raise ValueError(
            f"Each image must have shape [C, H, W] or [H, W], got "
            f"{tuple(image_tensor.shape)}."
        )
    if mask_tensor.dim() != 3:
        raise ValueError(
            f"Each mask must have shape [1, H, W] or [H, W], got "
            f"{tuple(mask_tensor.shape)}."
        )
    if mask_tensor.size(0) != 1:
        raise ValueError(f"Each mask must have one channel, got {mask_tensor.size(0)}.")

    return image_tensor, mask_tensor


def _dataset_from_pairs(samples: Sequence[Any]) -> TensorDataset:
    """Build a TensorDataset from a sequence of image/mask samples."""
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for index, sample in enumerate(samples):
        if isinstance(sample, dict):
            image = _get_first_existing(sample, ("image", "images", "x", "input"))
            mask = _get_first_existing(sample, ("mask", "masks", "y", "target", "label"))
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            image, mask = sample[0], sample[1]
        else:
            raise ValueError(
                "Unsupported sample format at index "
                f"{index}. Expected (image, mask) or a dict with image/mask keys."
            )

        image_tensor, mask_tensor = _normalize_image_mask_pair(image, mask)
        images.append(image_tensor)
        masks.append(mask_tensor)

    if not images:
        raise ValueError("Dataset is empty.")

    return TensorDataset(torch.stack(images), torch.stack(masks))


def _get_first_existing(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first value found for a list of candidate keys."""
    for key in keys:
        if key in data:
            return data[key]

    raise ValueError(f"Dataset dict is missing expected keys: {tuple(keys)}.")


def _dataset_from_tensor_pair(images: Any, masks: Any) -> TensorDataset:
    """Build a TensorDataset from batched image and mask tensors."""
    image_tensor = _as_tensor(images, "images").float()
    mask_tensor = _as_tensor(masks, "masks").float()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(1)
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(1)

    if image_tensor.dim() != 4:
        raise ValueError(
            f"Images must have shape [B, C, H, W] or [B, H, W], got "
            f"{tuple(image_tensor.shape)}."
        )
    if mask_tensor.dim() != 4:
        raise ValueError(
            f"Masks must have shape [B, 1, H, W] or [B, H, W], got "
            f"{tuple(mask_tensor.shape)}."
        )
    if mask_tensor.size(1) != 1:
        raise ValueError(f"Masks must have one channel, got {mask_tensor.size(1)}.")
    if image_tensor.size(0) != mask_tensor.size(0):
        raise ValueError(
            "Images and masks must have the same batch dimension, got "
            f"{image_tensor.size(0)} and {mask_tensor.size(0)}."
        )

    return TensorDataset(image_tensor, mask_tensor)


def load_test_dataset(data_path: Path) -> TensorDataset:
    """Load a .pt test dataset in one of the supported formats."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data = torch.load(data_path, map_location="cpu")

    if isinstance(data, dict):
        images = _get_first_existing(data, ("images", "image", "x", "inputs"))
        masks = _get_first_existing(data, ("masks", "mask", "y", "targets", "labels"))
        return _dataset_from_tensor_pair(images, masks)

    if isinstance(data, tuple) and len(data) == 2:
        return _dataset_from_tensor_pair(data[0], data[1])

    if isinstance(data, list):
        return _dataset_from_pairs(data)

    raise ValueError(
        "Unsupported dataset format. Expected a list of (image, mask), a tuple "
        "of (images, masks), or a dict with images/masks keys."
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a model and return average Dice and IoU over all batches."""
    model.to(device)
    model.eval()

    dice_scores: List[float] = []
    iou_scores: List[float] = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            dice_scores.append(dice_score(outputs, masks, threshold=threshold))
            iou_scores.append(iou_score(outputs, masks, threshold=threshold))

    if not dice_scores:
        raise ValueError("No batches were evaluated. Check the dataset and batch size.")

    return {
        "dice": sum(dice_scores) / len(dice_scores),
        "iou": sum(iou_scores) / len(iou_scores),
    }


def format_model_name(model_name: str) -> str:
    """Return a readable display name for a model argument."""
    if model_name == "unet":
        return "UNet"
    if model_name == "attention_unet":
        return "AttentionUNet"
    if model_name == "segformer_b0":
        return "SegFormer-B0"
    return model_name


def main() -> None:
    """Run model evaluation from command-line arguments."""
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.data}")

    device = resolve_device(args.device)
    model = build_model(args.model)

    load_checkpoint(model, args.checkpoint, device)
    dataset = load_test_dataset(args.data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
    )

    print("Evaluation Results")
    print(f"Model: {format_model_name(args.model)}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    print(f"Dice Score: {results['dice']:.4f}")
    print(f"IoU Score : {results['iou']:.4f}")


if __name__ == "__main__":
    main()
