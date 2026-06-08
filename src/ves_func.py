"""
Analyze a segmented retinal vessel mask.

Input: a binary/gray mask image where vessel pixels are bright and background
pixels are dark.

Output: vessel density, branch count, junction count, endpoint count, and a
simple screening risk score.
"""
from collections import deque

import numpy as np
from PIL import Image


def load_binary_mask(image_path, threshold=127, invert=False):
    """Load an image and convert it to a binary vessel mask."""
    image = Image.open(image_path).convert("L")
    gray_image = np.asarray(image)

    mask = gray_image > threshold

    if invert:
        mask = ~mask

    return mask


def skeletonize_zhang_suen(mask, max_iterations=1000):
    """Skeletonize a binary mask using the Zhang-Suen thinning algorithm."""
    skeleton = mask.astype(bool).copy()

    if skeleton.ndim != 2:
        raise ValueError("Mask must be a 2D image.")

    for _ in range(max_iterations):
        has_changed = False

        for step in [0, 1]:
            padded = np.pad(skeleton, 1, mode="constant", constant_values=False)

            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            neighbor_count = sum(neighbors)
            transition_count = 0

            for i in range(8):
                current_neighbor = neighbors[i]
                next_neighbor = neighbors[(i + 1) % 8]
                transition_count += (~current_neighbor) & next_neighbor

            if step == 0:
                removable_pixels = (
                    skeleton
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transition_count == 1)
                    & ~(p2 & p4 & p6)
                    & ~(p4 & p6 & p8)
                )
            else:
                removable_pixels = (
                    skeleton
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transition_count == 1)
                    & ~(p2 & p4 & p8)
                    & ~(p2 & p6 & p8)
                )

            if np.any(removable_pixels):
                skeleton[removable_pixels] = False
                has_changed = True

        if not has_changed:
            break

    return skeleton


def count_neighbors(binary_image):
    """Count 8-connected neighbors around each foreground pixel."""
    padded = np.pad(binary_image.astype(np.uint8), 1, mode="constant", constant_values=0)

    neighbor_count = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )

    return neighbor_count


def count_neighbor_groups(binary_image):
    """Count separated groups in the 8-neighborhood of each skeleton pixel."""
    padded = np.pad(binary_image.astype(bool), 1, mode="constant", constant_values=False)

    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]

    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
    group_count = 0

    for i in range(8):
        current_neighbor = neighbors[i]
        next_neighbor = neighbors[(i + 1) % 8]
        group_count += (~current_neighbor) & next_neighbor

    return group_count


def get_neighbors_8(y, x, height, width):
    """Return valid 8-neighbor coordinates for one pixel."""
    neighbors = []

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue

            next_y = y + dy
            next_x = x + dx

            if 0 <= next_y < height and 0 <= next_x < width:
                neighbors.append((next_y, next_x))

    return neighbors


def connected_component_sizes(binary_image):
    """Find sizes of all 8-connected components in a binary image."""
    visited = np.zeros(binary_image.shape, dtype=bool)
    component_sizes = []
    height, width = binary_image.shape

    foreground_pixels = np.argwhere(binary_image)

    for pixel in foreground_pixels:
        start_y = int(pixel[0])
        start_x = int(pixel[1])

        if visited[start_y, start_x]:
            continue

        component_size = 0
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True

        while queue:
            current_y, current_x = queue.popleft()
            component_size += 1

            neighbors = get_neighbors_8(current_y, current_x, height, width)
            for next_y, next_x in neighbors:
                if binary_image[next_y, next_x] and not visited[next_y, next_x]:
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))

        component_sizes.append(component_size)

    return component_sizes


def calculate_risk_score(vessel_density, branch_count, junction_count, endpoint_count, connected_components):
    """Calculate a simple screening risk score from extracted features."""
    score = 0
    reasons = []

    if vessel_density < 0.045:
        score += 35
        reasons.append("vessel density is very low")
    elif vessel_density < 0.065:
        score += 15
        reasons.append("vessel density is low")
    elif vessel_density > 0.18:
        score += 15
        reasons.append("vessel density is unusually high")

    if branch_count < 25:
        score += 20
        reasons.append("branch count is low")
    elif branch_count > 300:
        score += 10
        reasons.append("branch count is unusually high")

    if junction_count < 10:
        score += 15
        reasons.append("junction count is low")
    elif junction_count > 300:
        score += 10
        reasons.append("junction count is unusually high")

    if endpoint_count > max(150, branch_count * 4):
        score += 15
        reasons.append("many endpoints suggest fragmented vessels")

    if connected_components > 35:
        score += 15
        reasons.append("many disconnected components suggest segmentation fragmentation")

    score = min(score, 100)

    if score < 30:
        risk_level = "low"
        interpretation = "Low screening risk based on the vessel mask features."
    elif score < 60:
        risk_level = "moderate"
        interpretation = "Moderate screening risk; review the mask and original image."
    else:
        risk_level = "high"
        interpretation = "High screening risk; recommend clinician review."

    if reasons:
        interpretation += " Main signals: " + "; ".join(reasons) + "."

    return score, risk_level, interpretation


def analyze_vessel_mask(image_path, threshold=127, invert=False):
    """Analyze a segmented retinal vessel mask and return a result dictionary."""
    mask = load_binary_mask(image_path, threshold, invert)
    skeleton = skeletonize_zhang_suen(mask)

    neighbor_count = count_neighbors(skeleton)
    neighbor_groups = count_neighbor_groups(skeleton)

    endpoint_pixels = skeleton & (neighbor_count == 1)
    junction_pixels = skeleton & (neighbor_groups >= 3)
    branch_pixels = skeleton & ~junction_pixels

    branch_sizes = connected_component_sizes(branch_pixels)
    branch_sizes = [size for size in branch_sizes if size >= 2]

    junction_sizes = connected_component_sizes(junction_pixels)
    vessel_component_sizes = connected_component_sizes(mask)

    vessel_density = float(mask.mean())
    branch_count = len(branch_sizes)
    junction_count = len(junction_sizes)
    endpoint_count = int(endpoint_pixels.sum())
    connected_components = len(vessel_component_sizes)

    risk_score, risk_level, interpretation = calculate_risk_score(
        vessel_density,
        branch_count,
        junction_count,
        endpoint_count,
        connected_components,
    )

    height, width = mask.shape

    result = {
        "image_path": str(image_path),
        "width": int(width),
        "height": int(height),
        "vessel_density": vessel_density,
        "skeleton_pixels": int(skeleton.sum()),
        "branch_count": branch_count,
        "junction_count": junction_count,
        "endpoint_count": endpoint_count,
        "connected_components": connected_components,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "interpretation": interpretation,
    }

    return result
