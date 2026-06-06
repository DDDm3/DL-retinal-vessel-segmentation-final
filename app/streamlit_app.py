"""Streamlit demo for retinal vessel segmentation and vessel analysis."""

from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn

try:
    import cv2
except ImportError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SEGFORMER_PATH = SRC_DIR / "models" / "segformer.py"
DEEPLABPLUS_PATH = SRC_DIR / "models" / "deeplabv3plus_resnet50.py"
VES_FUNC_PATH = SRC_DIR / "ves_func.py"
MODEL_INPUT_SIZE = (512, 512)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODEL_CONFIGS = {
    "SegFormer-B0": {
        "module_path": SEGFORMER_PATH,
        "class_name": "SegFormerB0",
        "checkpoint": PROJECT_ROOT / "src" / "models" / "best_segformer_b0.pth",
        "default_threshold": 0.15,
        "init_kwargs": {"pretrained": False},
        "normalize": True,
        "auto_crop": True,
    },
    "DeepLabV3+-ResNet50": {
        "module_path": DEEPLABPLUS_PATH,
        "class_name": "DeepLabV3PlusResNet50Binary",
        "checkpoint": PROJECT_ROOT / "src" / "models" / "best_deeplabv3plus_resnet50.pth",
        "default_threshold": 0.50,
        "init_kwargs": {"pretrained_backbone": False},
        "normalize": True,
        "auto_crop": True,
        "postprocess": False,
        "min_area": 20,
    },
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class(module_path: Path, class_name: str, module_name: str) -> type[nn.Module]:
    """Load a model class directly from a source file."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Expected {class_name} in {module_path}.") from exc


@st.cache_resource(show_spinner=False)
def load_vessel_module() -> Optional[Any]:
    """Load vessel analysis helpers when available."""
    if not VES_FUNC_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location("streamlit_ves_func", VES_FUNC_PATH)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract model weights from common checkpoint formats."""
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
    """Load weights while handling common wrapper prefixes."""
    candidates = [
        state_dict,
        strip_prefix(state_dict, "module."),
        strip_prefix(state_dict, "_orig_mod."),
        add_prefix(state_dict, "model."),
    ]

    errors = []
    for candidate in candidates:
        try:
            model.load_state_dict(candidate)
            return
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        "Could not load checkpoint into the selected model. "
        f"Last load error: {errors[-1]}"
    )


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, checkpoint_path: str) -> Tuple[nn.Module, torch.device]:
    """Load and cache the selected model for inference."""
    config = MODEL_CONFIGS[model_name]
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    device = resolve_device()
    model_class = load_class(
        module_path=config["module_path"],
        class_name=config["class_name"],
        module_name=f"streamlit_{config['class_name']}",
    )
    model = model_class(**config["init_kwargs"])

    checkpoint = torch.load(path, map_location=device)
    load_state_dict_robust(model, extract_state_dict(checkpoint))

    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image, normalize: bool) -> torch.Tensor:
    """Convert a PIL RGB image to [1, 3, H, W] float tensor."""
    resized = image.resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0).contiguous()


def crop_retina_region(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int], bool]:
    """Crop the visible fundus region from black-padded uploads."""
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    max_channel = array.max(axis=2)
    min_channel = array.min(axis=2)
    color_spread = max_channel - min_channel

    # Prefer the orange/red fundus content when the upload is a report figure
    # containing labels, white canvas, and a separate black/white GT panel.
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    fundus_like = (
        (max_channel > 35)
        & (max_channel < 245)
        & (color_spread > 18)
        & (red >= green - 20)
        & (red > blue + 10)
    )

    if fundus_like.mean() >= 0.015:
        ys, xs = np.where(fundus_like)
        left = int(xs.min())
        right = int(xs.max()) + 1
        top = int(ys.min())
        bottom = int(ys.max()) + 1

        box_width = right - left
        box_height = bottom - top
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        side = int(max(box_width, box_height) * 1.08)
        side = min(max(side, 1), max(width, height))

        crop_left = int(round(center_x - side / 2.0))
        crop_top = int(round(center_y - side / 2.0))
        crop_right = crop_left + side
        crop_bottom = crop_top + side

        if crop_left < 0:
            crop_right -= crop_left
            crop_left = 0
        if crop_top < 0:
            crop_bottom -= crop_top
            crop_top = 0
        if crop_right > width:
            crop_left -= crop_right - width
            crop_right = width
        if crop_bottom > height:
            crop_top -= crop_bottom - height
            crop_bottom = height

        crop_left = max(crop_left, 0)
        crop_top = max(crop_top, 0)
        crop_right = min(crop_right, width)
        crop_bottom = min(crop_bottom, height)
        crop_box = (crop_left, crop_top, crop_right, crop_bottom)
        return image.crop(crop_box), crop_box, True

    brightness = max_channel
    non_black = brightness > 18

    if non_black.mean() < 0.02:
        return image, (0, 0, width, height), False

    ys, xs = np.where(non_black)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    box_width = right - left
    box_height = bottom - top
    if box_width >= width * 0.92 and box_height >= height * 0.92:
        return image, (0, 0, width, height), False

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    side = int(max(box_width, box_height) * 1.12)
    side = min(max(side, 1), max(width, height))

    crop_left = int(round(center_x - side / 2.0))
    crop_top = int(round(center_y - side / 2.0))
    crop_right = crop_left + side
    crop_bottom = crop_top + side

    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > width:
        crop_left -= crop_right - width
        crop_right = width
    if crop_bottom > height:
        crop_top -= crop_bottom - height
        crop_bottom = height

    crop_left = max(crop_left, 0)
    crop_top = max(crop_top, 0)
    crop_right = min(crop_right, width)
    crop_bottom = min(crop_bottom, height)

    crop_box = (crop_left, crop_top, crop_right, crop_bottom)
    return image.crop(crop_box), crop_box, True


def logits_from_output(output: Any) -> torch.Tensor:
    """Extract logits from tensor outputs or torchvision-style dict outputs."""
    if isinstance(output, dict):
        if "out" in output:
            output = output["out"]
        elif "logits" in output:
            output = output["logits"]
        else:
            raise ValueError("Model output dict did not contain 'out' or 'logits'.")

    if output is None:
        raise ValueError("Model output did not contain logits.")
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor model output, got {type(output).__name__}.")

    return output


def probability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits into a [B, H, W] vessel probability tensor."""
    if logits.dim() == 3:
        return torch.sigmoid(logits)
    if logits.dim() != 4:
        raise ValueError(f"Expected output [B, C, H, W] or [B, H, W], got {tuple(logits.shape)}.")

    if logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.size(1) >= 2:
        return torch.softmax(logits, dim=1)[:, 1]

    raise ValueError(f"Expected at least one output channel, got {tuple(logits.shape)}.")


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove small 8-connected foreground components from a binary mask."""
    foreground = mask.astype(bool)
    visited = np.zeros(foreground.shape, dtype=bool)
    cleaned = np.zeros(foreground.shape, dtype=bool)
    height, width = foreground.shape

    for start_y, start_x in np.argwhere(foreground):
        start_y = int(start_y)
        start_x = int(start_x)
        if visited[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        component = []
        visited[start_y, start_x] = True

        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    next_y = y + dy
                    next_x = x + dx
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and foreground[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

        if len(component) >= min_area:
            for y, x in component:
                cleaned[y, x] = True

    return cleaned.astype(np.uint8)


def postprocess_binary_mask(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Apply the same light cleanup used by the DeepLabV3 notebook."""
    binary = mask.astype(np.uint8)

    if cv2 is not None:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        cleaned = np.zeros_like(binary)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == label_id] = 1
        return cleaned.astype(np.uint8)

    return remove_small_components(binary, min_area=min_area)


def predict_mask(
    model_name: str,
    model: nn.Module,
    device: torch.device,
    image: Image.Image,
    threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run inference and return an aligned binary mask plus debug stats."""
    config = MODEL_CONFIGS[model_name]
    normalize = bool(config["normalize"])
    if bool(config.get("auto_crop", False)):
        inference_image, crop_box, was_cropped = crop_retina_region(image)
    else:
        inference_image = image
        crop_box = (0, 0, image.width, image.height)
        was_cropped = False

    input_tensor = preprocess_image(inference_image, normalize=normalize).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        logits = logits_from_output(output)
        probability = probability_from_logits(logits)
        probability_max = float(probability.detach().max().cpu().item())
        probability_np = probability.detach().cpu().squeeze(0).numpy()

    probability_image = Image.fromarray(
        np.clip(probability_np * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    probability_image = probability_image.resize(inference_image.size, Image.Resampling.BILINEAR)
    probability_crop = np.asarray(probability_image, dtype=np.float32) / 255.0
    probability_resized = np.zeros((image.height, image.width), dtype=np.float32)
    left, top, right, bottom = crop_box
    probability_resized[top:bottom, left:right] = probability_crop
    binary = (probability_resized >= threshold).astype(np.uint8)

    raw_vessel_pixels = int(binary.sum())
    postprocess = bool(config.get("postprocess", False))
    if postprocess:
        binary = postprocess_binary_mask(
            binary,
            min_area=int(config.get("min_area", 20)),
        )

    binary_mask = binary.astype(np.uint8)

    debug_stats = {
        "raw_output_shape": tuple(logits.shape),
        "raw_output_min": float(logits.detach().min().cpu().item()),
        "raw_output_max": float(logits.detach().max().cpu().item()),
        "probability_min": float(probability.detach().min().cpu().item()),
        "probability_max": probability_max,
        "probability_map": probability_resized,
        "selected_threshold": float(threshold),
        "raw_vessel_pixels": raw_vessel_pixels,
        "vessel_pixels": int(binary_mask.sum()),
        "total_pixels": int(binary_mask.size),
        "crop_box": crop_box,
        "input_size": inference_image.size,
        "input_image": inference_image,
        "was_cropped": was_cropped,
        "preprocessing": "ImageNet normalized" if normalize else "RGB [0, 1]",
        "postprocessing": "Morphology + small component cleanup" if postprocess else "None",
    }

    return binary_mask, debug_stats


def make_overlay(image: Image.Image, binary_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a red vessel mask over the original RGB image."""
    original = np.asarray(image, dtype=np.float32)
    overlay = original.copy()
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    vessel_pixels = binary_mask > 0
    overlay[vessel_pixels] = (1.0 - alpha) * original[vessel_pixels] + alpha * red
    return np.clip(overlay, 0, 255).astype(np.uint8)


def connected_component_sizes(binary_image: np.ndarray) -> list[int]:
    """Fallback 8-connected component sizes for binary images."""
    visited = np.zeros(binary_image.shape, dtype=bool)
    sizes: list[int] = []
    height, width = binary_image.shape

    for start_y, start_x in np.argwhere(binary_image):
        start_y = int(start_y)
        start_x = int(start_x)
        if visited[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        size = 0

        while stack:
            y, x = stack.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    next_y = y + dy
                    next_x = x + dx
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and binary_image[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

        sizes.append(size)

    return sizes


def fallback_neighbor_count(binary_image: np.ndarray) -> np.ndarray:
    """Fallback 8-neighbor count for skeleton analysis."""
    padded = np.pad(binary_image.astype(np.uint8), 1, mode="constant", constant_values=0)
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )


def analyze_binary_mask(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Compute vessel analysis metrics from a predicted binary mask."""
    mask = binary_mask.astype(bool)
    vessel_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    vessel_density = vessel_pixels / total_pixels if total_pixels else 0.0

    analysis: Dict[str, Any] = {
        "vessel_pixels": vessel_pixels,
        "total_pixels": total_pixels,
        "vessel_density": vessel_density,
        "vessel_density_percent": vessel_density * 100.0,
        "skeleton": None,
        "branch_count": None,
        "junction_count": None,
        "endpoint_count": None,
        "connected_components": None,
        "risk_score": None,
    }

    ves_func = load_vessel_module()
    try:
        if ves_func is not None and hasattr(ves_func, "skeletonize_zhang_suen"):
            skeleton = ves_func.skeletonize_zhang_suen(mask)
        else:
            skeleton = mask

        if ves_func is not None and hasattr(ves_func, "count_neighbors"):
            neighbor_count = ves_func.count_neighbors(skeleton)
        else:
            neighbor_count = fallback_neighbor_count(skeleton)

        if ves_func is not None and hasattr(ves_func, "count_neighbor_groups"):
            neighbor_groups = ves_func.count_neighbor_groups(skeleton)
        else:
            neighbor_groups = neighbor_count

        endpoint_pixels = skeleton & (neighbor_count == 1)
        junction_pixels = skeleton & (neighbor_groups >= 3)
        branch_pixels = skeleton & ~junction_pixels

        component_fn = connected_component_sizes
        if ves_func is not None and hasattr(ves_func, "connected_component_sizes"):
            component_fn = ves_func.connected_component_sizes

        branch_sizes = [size for size in component_fn(branch_pixels) if size >= 2]
        junction_sizes = component_fn(junction_pixels)
        vessel_component_sizes = component_fn(mask)

        branch_count = len(branch_sizes)
        junction_count = len(junction_sizes)
        endpoint_count = int(endpoint_pixels.sum())
        connected_components = len(vessel_component_sizes)

        analysis.update(
            {
                "skeleton": skeleton.astype(np.uint8),
                "branch_count": branch_count,
                "junction_count": junction_count,
                "endpoint_count": endpoint_count,
                "connected_components": connected_components,
            }
        )

        if ves_func is not None and hasattr(ves_func, "calculate_risk_score"):
            risk_score, _, _ = ves_func.calculate_risk_score(
                vessel_density,
                branch_count,
                junction_count,
                endpoint_count,
                connected_components,
            )
            analysis["risk_score"] = risk_score
    except Exception:
        return analysis

    return analysis


def image_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes for Streamlit downloads."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def mask_to_image(binary_mask: np.ndarray) -> Image.Image:
    """Convert a binary mask array to a displayable PIL image."""
    return Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")


def display_metric(label: str, value: Any, suffix: str = "") -> None:
    """Render a metric and show N/A when the value is unavailable."""
    if value is None:
        st.metric(label, "N/A")
    elif isinstance(value, float):
        st.metric(label, f"{value:.2f}{suffix}")
    else:
        st.metric(label, f"{value}{suffix}")


def density_interpretation(vessel_density: float) -> str:
    """Return a simple non-medical interpretation for vessel density."""
    if vessel_density < 0.045:
        return "Low vessel density"
    if vessel_density > 0.18:
        return "High vessel density"
    return "Moderate vessel density"


def render_sidebar(model_name: str, threshold: float) -> bool:
    """Render sidebar model information and controls."""
    checkpoint = MODEL_CONFIGS[model_name]["checkpoint"]

    st.sidebar.header("Model information")
    st.sidebar.write(f"Model: {model_name}")
    st.sidebar.write(f"Checkpoint: `{checkpoint.relative_to(PROJECT_ROOT)}`")
    st.sidebar.write(f"Device: {resolve_device().type.upper()}")

    if checkpoint.exists():
        st.sidebar.success("Checkpoint found")
    else:
        st.sidebar.error("Checkpoint missing")

    st.sidebar.divider()
    st.sidebar.write(f"Threshold: {threshold:.2f}")
    return st.sidebar.button("Run Model", type="primary", width="stretch")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Retinal Vessel Segmentation Demo",
        layout="wide",
    )

    st.title("Retinal Vessel Segmentation Demo")
    st.caption("This demo is for academic purposes only and is not a medical diagnostic tool.")

    model_name = st.sidebar.selectbox(
        "Model",
        options=list(MODEL_CONFIGS.keys()),
        index=0,
        key="selected_model_name",
    )
    default_threshold = MODEL_CONFIGS[model_name]["default_threshold"]

    if st.session_state.get("last_selected_model_name") != model_name:
        st.session_state["prediction_threshold"] = float(default_threshold)
        st.session_state["last_selected_model_name"] = model_name

    threshold = st.sidebar.slider(
        "Prediction threshold",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        key="prediction_threshold",
    )
    run_model = render_sidebar(model_name, threshold)

    uploaded_file = st.file_uploader(
        "Upload retinal fundus image",
        type=("png", "jpg", "jpeg"),
    )

    if uploaded_file is None:
        st.info("Upload a retinal image to begin.")
        return

    original_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded image")
    st.image(original_image, caption="Original image", width="stretch")

    if not run_model:
        return

    checkpoint = MODEL_CONFIGS[model_name]["checkpoint"]
    try:
        with st.spinner(f"Running {model_name} inference..."):
            model, device = load_model(model_name, str(checkpoint))
            binary_mask, debug_stats = predict_mask(
                model_name,
                model,
                device,
                original_image,
                threshold,
            )
            overlay = make_overlay(original_image, binary_mask)
            analysis = analyze_binary_mask(binary_mask)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not run inference: {exc}")
        return

    model_input_image = debug_stats.get("input_image", original_image)
    mask_image = mask_to_image(binary_mask)
    probability_image = Image.fromarray(
        np.clip(debug_stats["probability_map"] * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    overlay_image = Image.fromarray(overlay, mode="RGB")
    skeleton = analysis.get("skeleton")
    skeleton_image = mask_to_image(skeleton) if skeleton is not None else None

    st.subheader("Segmentation results")
    result_columns = st.columns(6)
    with result_columns[0]:
        st.image(original_image, caption="Original image", width="stretch")
    with result_columns[1]:
        st.image(model_input_image, caption="Model input crop", width="stretch")
    with result_columns[2]:
        st.image(probability_image, caption="Probability map", width="stretch")
    with result_columns[3]:
        st.image(mask_image, caption="Predicted vessel mask", width="stretch")
    with result_columns[4]:
        st.image(overlay_image, caption="Overlay", width="stretch")
    with result_columns[5]:
        if skeleton_image is not None:
            st.image(skeleton_image, caption="Skeletonized vessel map", width="stretch")
        else:
            st.info("Skeleton map unavailable.")

    st.subheader("Vessel analysis")
    metric_cols = st.columns(6)
    with metric_cols[0]:
        display_metric("Vessel Density", analysis.get("vessel_density_percent"), "%")
    with metric_cols[1]:
        display_metric("Branch Count", analysis.get("branch_count"))
    with metric_cols[2]:
        display_metric("Junction Count", analysis.get("junction_count"))
    with metric_cols[3]:
        display_metric("Endpoint Count", analysis.get("endpoint_count"))
    with metric_cols[4]:
        display_metric("Connected Components", analysis.get("connected_components"))
    with metric_cols[5]:
        display_metric("Risk Score", analysis.get("risk_score"))

    st.write(density_interpretation(float(analysis["vessel_density"])))
    st.caption(
        f"{analysis['vessel_pixels']:,} vessel pixels / "
        f"{analysis['total_pixels']:,} total pixels"
    )

    st.subheader("Inference debug")
    debug_cols = st.columns(8)
    with debug_cols[0]:
        st.metric("Raw Output Min", f"{debug_stats['raw_output_min']:.6f}")
    with debug_cols[1]:
        st.metric("Raw Output Max", f"{debug_stats['raw_output_max']:.6f}")
    with debug_cols[2]:
        st.metric("Probability Min", f"{debug_stats['probability_min']:.6f}")
    with debug_cols[3]:
        st.metric("Probability Max", f"{debug_stats['probability_max']:.6f}")
    with debug_cols[4]:
        st.metric("Threshold", f"{debug_stats['selected_threshold']:.6f}")
    with debug_cols[5]:
        st.metric("Raw Vessel Pixels", f"{debug_stats['raw_vessel_pixels']:,}")
    with debug_cols[6]:
        st.metric("Vessel Pixels", f"{debug_stats['vessel_pixels']:,}")
    with debug_cols[7]:
        st.metric("Total Pixels", f"{debug_stats['total_pixels']:,}")
    st.caption(
        f"Raw output shape: {debug_stats['raw_output_shape']} | "
        f"Preprocessing: {debug_stats['preprocessing']} | "
        f"Postprocessing: {debug_stats['postprocessing']} | "
        f"Auto crop: {debug_stats['was_cropped']} | "
        f"Model input size: {debug_stats['input_size']} | "
        f"Crop box: {debug_stats['crop_box']}"
    )
    if debug_stats["vessel_pixels"] == 0:
        suggested_threshold = max(debug_stats["probability_max"] * 0.75, 0.000001)
        st.warning(
            "No vessel pixels passed the selected threshold. Check probability max "
            "above to decide whether the threshold is too high or the model output "
            "contains little vessel signal. "
            f"For inspection only, try a threshold near {suggested_threshold:.6f}."
        )

    download_mask_col, download_overlay_col = st.columns(2)
    with download_mask_col:
        st.download_button(
            "Download prediction mask PNG",
            data=image_bytes(mask_image),
            file_name="predicted_vessel_mask.png",
            mime="image/png",
            width="stretch",
        )
    with download_overlay_col:
        st.download_button(
            "Download overlay PNG",
            data=image_bytes(overlay_image),
            file_name="retinal_vessel_overlay.png",
            mime="image/png",
            width="stretch",
        )


if __name__ == "__main__":
    main()
