"""Export a DRIVE test dataset .pt file for model evaluation."""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
PT_EXTENSION = ".pt"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FOV_MASK_KEYS = {"mask", "masks", "roi", "fov", "field_of_view"}
VESSEL_MASK_KEYS = (
    "manual",
    "manual_1",
    "manual1",
    "manual_2",
    "manual2",
    "vessel",
    "vessels",
    "vessel_mask",
    "groundtruth",
    "ground_truth",
    "gt",
    "label",
    "labels",
    "target",
    "targets",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a DRIVE test dataset .pt file from a processed zip."
    )
    parser.add_argument(
        "--zip",
        default=Path("dataset/DRIVE_processed_dataset.zip"),
        type=Path,
        help="Path to the processed DRIVE dataset zip.",
    )
    parser.add_argument(
        "--output",
        default=Path("dataset/drive_test_dataset.pt"),
        type=Path,
        help="Path where the exported .pt dataset will be saved.",
    )
    parser.add_argument(
        "--image-size",
        default=512,
        type=int,
        help="Square image size used for exported tensors.",
    )
    parser.add_argument(
        "--debug-output",
        default=Path("outputs/drive_test_dataset_debug.png"),
        type=Path,
        help="Path for one image/mask debug visualization.",
    )

    return parser.parse_args()


def is_file_entry(name: str) -> bool:
    """Return True when a zip entry looks like a file."""
    return not name.endswith("/")


def split_tokens(path: str) -> List[str]:
    """Split a zip path into lowercase searchable tokens."""
    return [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", path.lower())
        if token
    ]


def is_test_entry(path: str) -> bool:
    """Return True when a path appears to belong to a test split."""
    tokens = split_tokens(path)
    return any(token in {"test", "testing"} for token in tokens)


def is_mask_entry(path: str) -> bool:
    """Return True when a path appears to contain a vessel mask."""
    tokens = split_tokens(path)
    name = Path(path).stem.lower()
    if any(token in {"fov", "roi"} for token in tokens):
        return False

    mask_tokens = {
        "groundtruth",
        "ground",
        "gt",
        "label",
        "labels",
        "manual",
        "manual1",
        "manual2",
        "target",
        "targets",
        "vessel",
        "vessels",
    }

    return any(token in mask_tokens for token in tokens) or "manual" in name


def is_image_entry(path: str) -> bool:
    """Return True when a path appears to contain a fundus image."""
    suffix = Path(path).suffix.lower()
    return suffix in IMAGE_EXTENSIONS and not is_mask_entry(path)


def sample_key(path: str) -> str:
    """Create a loose pairing key for matching images with masks."""
    stem = Path(path).stem.lower()
    number_match = re.search(r"\d+", stem)
    if number_match:
        return str(int(number_match.group(0)))

    key = re.sub(
        r"(image|images|mask|masks|manual|manual1|manual2|groundtruth|gt|"
        r"label|labels|target|targets|test|training|train|vessel|vessels)",
        "",
        stem,
    )
    key = re.sub(r"[^a-z0-9]+", "", key)
    if not key:
        raise ValueError(f"Could not derive a sample key from path: {path}")

    return key


def prefer_test_split(entries: Sequence[str]) -> List[str]:
    """Prefer test split entries when present, otherwise return all entries."""
    test_entries = [entry for entry in entries if is_test_entry(entry)]
    return test_entries or list(entries)


def classify_entry(path: str) -> str:
    """Classify a zip entry for inspection logs."""
    if path.endswith("/"):
        return "folder"
    if Path(path).suffix.lower() == PT_EXTENSION:
        return "torch sample"
    if is_image_entry(path):
        return "retinal image candidate"
    if is_mask_entry(path):
        return "vessel mask candidate"
    if any(token in {"fov", "roi"} for token in split_tokens(path)):
        return "FOV/ROI mask candidate (excluded)"

    return "other file"


def print_zip_structure(zip_file: zipfile.ZipFile) -> None:
    """Print the dataset structure and candidate file types inside the zip."""
    print("Dataset zip structure")
    for entry in zip_file.infolist():
        print(f"- {entry.filename} [{classify_entry(entry.filename)}]")


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Convert a PIL image to a normalized [3, H, W] float tensor."""
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image_bytes = torch.frombuffer(image.tobytes(), dtype=torch.uint8).clone()
    image_tensor = image_bytes.float().view(image_size, image_size, 3).permute(2, 0, 1)
    image_tensor = image_tensor / 255.0

    return (image_tensor - IMAGENET_MEAN) / IMAGENET_STD


def mask_to_tensor(mask: Image.Image, image_size: int) -> torch.Tensor:
    """Convert a PIL mask to a binary [1, H, W] float tensor."""
    mask = mask.convert("L").resize((image_size, image_size), Image.NEAREST)
    mask_bytes = torch.frombuffer(mask.tobytes(), dtype=torch.uint8).clone()
    mask_tensor = mask_bytes.float().view(1, image_size, image_size)

    return (mask_tensor > 127).float()


def tensor_image_to_export(image: torch.Tensor, image_size: int) -> torch.Tensor:
    """Normalize an existing image tensor to SegFormer evaluation format."""
    image = image.detach().float()

    if image.dim() == 3 and image.size(-1) in {1, 3}:
        image = image.permute(2, 0, 1)
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.dim() != 3:
        raise ValueError(f"Expected image tensor [C, H, W], got {tuple(image.shape)}.")

    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    elif image.size(0) != 3:
        raise ValueError(f"Expected 1 or 3 image channels, got {image.size(0)}.")

    min_value = float(image.min().item())
    max_value = float(image.max().item())
    is_imagenet_normalized = min_value < 0.0

    if not is_imagenet_normalized and max_value > 1.0:
        image = image / 255.0

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if is_imagenet_normalized:
        return image

    return (image - IMAGENET_MEAN) / IMAGENET_STD


def tensor_mask_to_export(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    """Normalize an existing mask tensor to binary [1, H, W] format."""
    mask = mask.detach().float()

    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.permute(2, 0, 1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"Expected mask tensor [1, H, W], got {tuple(mask.shape)}.")
    if mask.size(0) != 1:
        mask = mask[:1]

    if mask.max() > 1.0:
        mask = mask / 255.0

    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0),
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0)

    return (mask >= 0.5).float()


def read_image_from_zip(zip_file: zipfile.ZipFile, entry: str) -> Image.Image:
    """Read an image entry from a zip file."""
    with zip_file.open(entry) as file_obj:
        image_data = file_obj.read()

    return Image.open(io.BytesIO(image_data))


def load_pt_sample_from_zip(zip_file: zipfile.ZipFile, entry: str) -> Any:
    """Load a torch .pt sample stored inside a zip file."""
    with zip_file.open(entry) as file_obj:
        payload = io.BytesIO(file_obj.read())

    return torch.load(payload, map_location="cpu")


def get_first_existing(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    """Return the first value found in a dict for candidate keys."""
    for key in keys:
        if key in data:
            return data[key]

    return None


def extract_pair_from_pt_sample(sample: Any) -> Optional[Tuple[Any, Any]]:
    """Extract an image/mask pair from common .pt sample formats."""
    if isinstance(sample, dict):
        image = get_first_existing(sample, ("image", "images", "x", "input"))
        mask = get_first_existing(sample, VESSEL_MASK_KEYS)
        if image is not None and mask is not None:
            return image, mask

    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
        return sample[0], sample[1]

    return None


def describe_pt_mask_candidates(sample: Any) -> List[str]:
    """Return candidate mask descriptions from a .pt sample."""
    if not isinstance(sample, dict):
        return []

    descriptions: List[str] = []
    for key, value in sample.items():
        if key not in VESSEL_MASK_KEYS and key not in FOV_MASK_KEYS:
            continue
        if not torch.is_tensor(value):
            descriptions.append(f"{key}: non-tensor candidate")
            continue

        tensor = value.detach().float()
        positive_ratio = float((tensor > 0.5).float().mean().item())
        if key in FOV_MASK_KEYS:
            mask_type = "FOV/ROI mask (excluded)"
        else:
            mask_type = "vessel annotation candidate"
        descriptions.append(
            f"{key}: {mask_type}, shape={tuple(tensor.shape)}, "
            f"positive_ratio={positive_ratio:.4f}"
        )

    return descriptions


def export_from_pt_entries(
    zip_file: zipfile.ZipFile,
    entries: Sequence[str],
    image_size: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Export tensors from .pt entries when the zip stores sample files."""
    pt_entries = [entry for entry in entries if Path(entry).suffix.lower() == PT_EXTENSION]
    pt_entries = prefer_test_split(pt_entries)

    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for entry in sorted(pt_entries):
        sample = load_pt_sample_from_zip(zip_file, entry)
        pair = extract_pair_from_pt_sample(sample)
        if pair is None:
            continue

        image, mask = pair
        mask_key = select_vessel_mask_key(sample)
        print("Exporting sample")
        print(f"Image: {entry}::image")
        print(f"Mask: {entry}::{mask_key}")
        print("Mask type: vessel annotation")
        images.append(tensor_image_to_export(torch.as_tensor(image), image_size))
        masks.append(tensor_mask_to_export(torch.as_tensor(mask), image_size))

    if not images:
        return None

    return torch.stack(images), torch.stack(masks)


def select_vessel_mask_key(sample: Any) -> str:
    """Return the vessel annotation key selected from a .pt sample."""
    if not isinstance(sample, dict):
        return "tuple_index_1"

    for key in VESSEL_MASK_KEYS:
        if key in sample:
            return key

    raise ValueError(
        "No vessel annotation key found. Refusing to export FOV/ROI mask keys "
        f"{sorted(FOV_MASK_KEYS)}."
    )


def pair_image_and_mask_entries(entries: Sequence[str]) -> List[Tuple[str, str]]:
    """Pair image entries with corresponding mask entries using loose keys."""
    image_entries = prefer_test_split([entry for entry in entries if is_image_entry(entry)])
    mask_entries = prefer_test_split(
        [
            entry
            for entry in entries
            if Path(entry).suffix.lower() in IMAGE_EXTENSIONS and is_mask_entry(entry)
        ]
    )

    if not image_entries:
        raise ValueError("No image files were found in the dataset zip.")
    if not mask_entries:
        raise ValueError("No mask or ground-truth files were found in the dataset zip.")

    masks_by_key: Dict[str, str] = {}
    for mask_entry in sorted(mask_entries):
        masks_by_key.setdefault(sample_key(mask_entry), mask_entry)

    pairs: List[Tuple[str, str]] = []
    missing_masks: List[str] = []

    for image_entry in sorted(image_entries):
        key = sample_key(image_entry)
        mask_entry = masks_by_key.get(key)
        if mask_entry is None:
            missing_masks.append(image_entry)
            continue

        pairs.append((image_entry, mask_entry))

    if not pairs:
        raise ValueError(
            "No matching image/mask pairs were found in the dataset zip. "
            f"Images without masks: {missing_masks[:5]}"
        )

    return pairs


def export_from_image_entries(
    zip_file: zipfile.ZipFile,
    entries: Sequence[str],
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Export tensors from image/mask files stored inside a zip."""
    pairs = pair_image_and_mask_entries(entries)

    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for image_entry, mask_entry in pairs:
        image = read_image_from_zip(zip_file, image_entry)
        mask = read_image_from_zip(zip_file, mask_entry)

        images.append(image_to_tensor(image, image_size))
        masks.append(mask_to_tensor(mask, image_size))

    return torch.stack(images), torch.stack(masks)


def build_test_tensors(zip_path: Path, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build image and mask tensors from a processed DRIVE dataset zip."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")

    with zipfile.ZipFile(zip_path) as zip_file:
        print_zip_structure(zip_file)
        entries = [entry.filename for entry in zip_file.infolist() if is_file_entry(entry.filename)]
        if not entries:
            raise ValueError(f"No files were found in zip: {zip_path}")

        print_mask_candidates(zip_file, entries)

        pt_export = export_from_pt_entries(zip_file, entries, image_size)
        if pt_export is not None:
            return pt_export

        return export_from_image_entries(zip_file, entries, image_size)


def print_mask_candidates(zip_file: zipfile.ZipFile, entries: Sequence[str]) -> None:
    """Print candidate mask types found in image files or .pt samples."""
    print()
    print("Candidate mask inspection")
    image_mask_entries = [
        entry
        for entry in entries
        if Path(entry).suffix.lower() in IMAGE_EXTENSIONS
        and (is_mask_entry(entry) or any(token in {"fov", "roi"} for token in split_tokens(entry)))
    ]
    for entry in image_mask_entries:
        print(f"- {entry}: {classify_entry(entry)}")

    pt_entries = prefer_test_split(
        [entry for entry in entries if Path(entry).suffix.lower() == PT_EXTENSION]
    )
    for entry in sorted(pt_entries)[:5]:
        sample = load_pt_sample_from_zip(zip_file, entry)
        descriptions = describe_pt_mask_candidates(sample)
        if descriptions:
            print(f"- {entry}")
            for description in descriptions:
                print(f"  {description}")


def save_dataset(images: torch.Tensor, masks: torch.Tensor, output_path: Path) -> None:
    """Save exported tensors to a .pt dataset file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "masks": masks}, output_path)


def save_debug_visualization(
    image: torch.Tensor,
    mask: torch.Tensor,
    output_path: Path,
) -> None:
    """Save one debug image showing original image and vessel mask."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_image = (image * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    display_image = display_image.permute(1, 2, 0).cpu().numpy()
    display_mask = mask.squeeze(0).cpu().numpy()

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(display_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(display_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Vessel Mask")
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Export the DRIVE test dataset and print a summary."""
    args = parse_args()
    images, masks = build_test_tensors(args.zip, args.image_size)
    save_dataset(images, masks, args.output)
    save_debug_visualization(images[0], masks[0], args.debug_output)

    print("Test Dataset Export")
    print(f"Samples: {images.size(0)}")
    print(f"Image tensor shape: {tuple(images.shape)}")
    print(f"Mask tensor shape : {tuple(masks.shape)}")
    print(f"Output path: {args.output}")
    print(f"Debug visualization: {args.debug_output}")


if __name__ == "__main__":
    main()
