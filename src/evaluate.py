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

from src.models import (  # noqa: E402
    AttentionUNet,
    DeepLabV3PlusResNet50Binary,
    DeepLabV3ResNet50Binary,
    SegFormerB0,
    UNet,
)
from src.utils.metrics import (  # noqa: E402
    accuracy_score,
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)


TensorPair = Tuple[torch.Tensor, torch.Tensor]
DEFAULT_CHECKPOINTS = {
    "segformer_b0": Path("src/models/best_segformer_b0.pth"),
    "deeplabv3_resnet50": Path("src/models/best_deeplabv3_resnet50.pth"),
    "deeplabv3plus_resnet50": Path("src/models/best_deeplabv3plus_resnet50.pth"),
}
COMPARISON_MODELS = ("segformer_b0", "deeplabv3_resnet50", "deeplabv3plus_resnet50")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation models for retinal vessel segmentation."
    )
    parser.add_argument(
        "--model",
        choices=(
            "unet",
            "attention_unet",
            "segformer_b0",
            "deeplabv3_resnet50",
            "deeplabv3plus_resnet50",
            "all",
        ),
        required=True,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        type=Path,
        help="Path to a .pth checkpoint file. Not needed when --model all.",
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
        "--tune-threshold",
        action="store_true",
        help="Search thresholds on the provided dataset and report the best Dice threshold.",
    )
    parser.add_argument(
        "--threshold-min",
        default=0.3,
        type=float,
        help="Minimum threshold for --tune-threshold.",
    )
    parser.add_argument(
        "--threshold-max",
        default=0.85,
        type=float,
        help="Maximum threshold for --tune-threshold.",
    )
    parser.add_argument(
        "--threshold-step",
        default=0.05,
        type=float,
        help="Threshold step for --tune-threshold.",
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
    elif model_name == "deeplabv3_resnet50":
        model = DeepLabV3ResNet50Binary(pretrained_backbone=False)
    elif model_name == "deeplabv3plus_resnet50":
        model = DeepLabV3PlusResNet50Binary(pretrained_backbone=False)
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
    """Evaluate a model and return average Dice, IoU, and accuracy."""
    model.to(device)
    model.eval()

    dice_scores: List[float] = []
    iou_scores: List[float] = []
    accuracy_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            dice_scores.append(dice_score(outputs, masks, threshold=threshold))
            iou_scores.append(iou_score(outputs, masks, threshold=threshold))
            accuracy_scores.append(accuracy_score(outputs, masks, threshold=threshold))
            precision_scores.append(precision_score(outputs, masks, threshold=threshold))
            recall_scores.append(recall_score(outputs, masks, threshold=threshold))

    if not dice_scores:
        raise ValueError("No batches were evaluated. Check the dataset and batch size.")

    return {
        "dice": sum(dice_scores) / len(dice_scores),
        "iou": sum(iou_scores) / len(iou_scores),
        "accuracy": sum(accuracy_scores) / len(accuracy_scores),
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
    }


def collect_outputs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> TensorPair:
    """Run one inference pass and return CPU tensors for threshold search."""
    model.to(device)
    model.eval()

    outputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            batch_outputs = model(images).detach().cpu()
            outputs.append(batch_outputs)
            targets.append(masks.detach().cpu())

    if not outputs:
        raise ValueError("No batches were evaluated. Check the dataset and batch size.")

    return torch.cat(outputs, dim=0), torch.cat(targets, dim=0)


def evaluate_cached_outputs(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate already-computed model outputs at one threshold."""
    return {
        "dice": dice_score(outputs, masks, threshold=threshold),
        "iou": iou_score(outputs, masks, threshold=threshold),
        "accuracy": accuracy_score(outputs, masks, threshold=threshold),
        "precision": precision_score(outputs, masks, threshold=threshold),
        "recall": recall_score(outputs, masks, threshold=threshold),
    }


def threshold_candidates(
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> List[float]:
    """Build an inclusive threshold grid."""
    if threshold_step <= 0:
        raise ValueError("--threshold-step must be positive.")
    if threshold_min > threshold_max:
        raise ValueError("--threshold-min must be <= --threshold-max.")

    values: List[float] = []
    value = threshold_min
    while value <= threshold_max + 1e-9:
        values.append(round(value, 6))
        value += threshold_step
    return values


def tune_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
) -> Tuple[float, Dict[str, float]]:
    """Find the threshold with the best Dice score on a dataset."""
    outputs, masks = collect_outputs(model, dataloader, device)
    best_threshold = float(thresholds[0])
    best_results: Dict[str, float] = {}
    best_dice = -1.0

    for threshold in thresholds:
        results = evaluate_cached_outputs(outputs, masks, threshold=threshold)
        print(
            f"threshold={threshold:.3f} "
            f"dice={results['dice']:.4f} "
            f"iou={results['iou']:.4f} "
            f"accuracy={results['accuracy']:.4f} "
            f"precision={results['precision']:.4f} "
            f"recall={results['recall']:.4f}",
            flush=True,
        )
        if results["dice"] > best_dice:
            best_dice = results["dice"]
            best_threshold = float(threshold)
            best_results = results

    return best_threshold, best_results


def format_model_name(model_name: str) -> str:
    """Return a readable display name for a model argument."""
    if model_name == "unet":
        return "UNet"
    if model_name == "attention_unet":
        return "AttentionUNet"
    if model_name == "segformer_b0":
        return "SegFormer-B0"
    if model_name == "deeplabv3_resnet50":
        return "DeepLabV3-ResNet50"
    if model_name == "deeplabv3plus_resnet50":
        return "DeepLabV3+-ResNet50"
    return model_name


def print_results(model_name: str, checkpoint_path: Path, results: Dict[str, float]) -> None:
    """Print evaluation results for one model."""
    print("Evaluation Results")
    print(f"Model: {format_model_name(model_name)}")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    if "threshold" in results:
        print(f"Best Threshold: {results['threshold']:.4f}")
    print(f"Dice Score: {results['dice']:.4f}")
    print(f"IoU Score : {results['iou']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    if "precision" in results:
        print(f"Precision : {results['precision']:.4f}")
    if "recall" in results:
        print(f"Recall    : {results['recall']:.4f}")


def evaluate_one_model(
    model_name: str,
    checkpoint_path: Path,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    tune_threshold_args: argparse.Namespace | None = None,
) -> Dict[str, float]:
    """Load and evaluate one model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = build_model(model_name)
    load_checkpoint(model, checkpoint_path, device)

    if tune_threshold_args is not None and tune_threshold_args.tune_threshold:
        thresholds = threshold_candidates(
            tune_threshold_args.threshold_min,
            tune_threshold_args.threshold_max,
            tune_threshold_args.threshold_step,
        )
        best_threshold, best_results = tune_threshold(model, dataloader, device, thresholds)
        best_results["threshold"] = best_threshold
        return best_results

    return evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=threshold,
    )


def main() -> None:
    """Run model evaluation from command-line arguments."""
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.data}")

    device = resolve_device(args.device)
    dataset = load_test_dataset(args.data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.model == "all":
        for index, model_name in enumerate(COMPARISON_MODELS):
            checkpoint_path = DEFAULT_CHECKPOINTS[model_name]
            results = evaluate_one_model(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                dataloader=dataloader,
                device=device,
                threshold=args.threshold,
                tune_threshold_args=args,
            )
            if index > 0:
                print()
            print_results(model_name, checkpoint_path, results)
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --model all is used.")

    results = evaluate_one_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        tune_threshold_args=args,
    )
    print_results(args.model, args.checkpoint, results)


if __name__ == "__main__":
    main()
