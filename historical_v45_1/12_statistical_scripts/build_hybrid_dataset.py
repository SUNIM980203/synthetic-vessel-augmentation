#!/usr/bin/env python3
"""Build a real-background hybrid dataset from isolated Unity object cutouts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


TYPE_TO_CATEGORY = {18: 1, 19: 2, 23: 3, 64: 4, 40: 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-coco", type=Path, required=True)
    parser.add_argument("--real-images", type=Path, required=True)
    parser.add_argument("--synthetic-coco", type=Path)
    parser.add_argument("--synthetic-images", type=Path)
    parser.add_argument("--unity-cutouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutouts-per-class", type=int, default=100)
    parser.add_argument("--insertions-per-image", type=int, default=2)
    parser.add_argument(
        "--target-counts",
        help="Optional class targets, e.g. 1:99,2:98,3:98,4:100,5:94",
    )
    parser.add_argument("--max-insertions-per-image", type=int, default=4)
    parser.add_argument("--placement-retries", type=int, default=4)
    parser.add_argument(
        "--target-reserve-window",
        type=int,
        default=0,
        help="Add one catch-up slot within the final N anchor images for each class.",
    )
    parser.add_argument(
        "--target-schedule-buffer",
        type=int,
        default=0,
        help="Schedule this many extra attempts early while still hard-capping the saved target count.",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--semantic-placement", action="store_true")
    parser.add_argument("--sensor-aware", action="store_true")
    parser.add_argument(
        "--vessel-render-v2",
        action="store_true",
        help="Use the preregistered color/contrast/PSF recipe for vessel cutouts.",
    )
    parser.add_argument(
        "--vessel-min-bbox-scale",
        type=float,
        default=0.0,
        help="Minimum sqrt(width*height) in pixels for inserted vessel boxes; 0 disables.",
    )
    parser.add_argument(
        "--vessel-max-bbox-scale",
        type=float,
        default=0.0,
        help="Exclusive maximum sqrt(width*height) in pixels for inserted vessel boxes; 0 disables.",
    )
    parser.add_argument(
        "--strict-vessel-water",
        action="store_true",
        help="Require vessel boxes to remain almost entirely on the anchor-connected water surface.",
    )
    return parser.parse_args()


def parse_target_counts(value: str | None) -> dict[int, int] | None:
    if not value:
        return None
    result: dict[int, int] = {}
    for item in value.split(","):
        category_text, count_text = item.split(":", 1)
        category = int(category_text)
        count = int(count_text)
        if category not in range(1, 6) or count < 0:
            raise ValueError(f"Invalid target count: {item}")
        result[category] = count
    if set(result) != set(range(1, 6)):
        raise ValueError("--target-counts must define categories 1 through 5")
    return result


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def link_or_copy(source: Path, destination: Path) -> None:
    """Preserve unmodified counterpart pixels without a JPEG re-encode."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def intersection_over_union(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / max(union, 1e-9)


def overlaps_other_annotation(
    annotation: dict[str, object],
    annotations: list[dict[str, object]],
    crop_box: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = crop_box
    annotation_id = annotation["id"]
    for other in annotations:
        if other["id"] == annotation_id:
            continue
        ox, oy, ow, oh = [float(value) for value in other["bbox"]]
        center_x = ox + ow * 0.5
        center_y = oy + oh * 0.5
        if x0 <= center_x < x1 and y0 <= center_y < y1:
            return True
    return False


def extract_cutout(
    image: np.ndarray,
    annotation: dict[str, object],
    image_annotations: list[dict[str, object]],
) -> Image.Image | None:
    height, width = image.shape[:2]
    x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
    if box_width < 4 or box_height < 4 or box_width > 90 or box_height > 110:
        return None
    padding = max(4, round(max(box_width, box_height) * 0.18))
    x0 = max(0, math.floor(x) - padding)
    y0 = max(0, math.floor(y) - padding)
    x1 = min(width, math.ceil(x + box_width) + padding)
    y1 = min(height, math.ceil(y + box_height) + padding)
    if overlaps_other_annotation(annotation, image_annotations, (x0, y0, x1, y1)):
        return None
    crop = image[y0:y1, x0:x1].copy()
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    border = np.concatenate(
        [crop[0:2].reshape(-1, 3), crop[-2:].reshape(-1, 3), crop[:, 0:2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3)],
        axis=0,
    )
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(crop.astype(np.float32) - background, axis=2)
    mask = (distance > 22.0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if components <= 1:
        return None
    crop_center = np.array([crop.shape[1] * 0.5, crop.shape[0] * 0.5])
    component_scores = []
    for component in range(1, components):
        area = stats[component, cv2.CC_STAT_AREA]
        if area < 8:
            continue
        distance_to_center = np.linalg.norm(centroids[component] - crop_center)
        component_scores.append((area / (1.0 + distance_to_center), component))
    if not component_scores:
        return None
    primary = max(component_scores)[1]
    primary_centroid = centroids[primary]
    selected = np.zeros_like(mask)
    for component in range(1, components):
        area = stats[component, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        if np.linalg.norm(centroids[component] - primary_centroid) <= max(crop.shape[:2]) * 0.55:
            selected[labels == component] = 255
    selected = cv2.GaussianBlur(selected, (3, 3), 0.65)
    nonzero = cv2.findNonZero((selected > 20).astype(np.uint8))
    if nonzero is None:
        return None
    tight_x, tight_y, tight_width, tight_height = cv2.boundingRect(nonzero)
    alpha_area = int(np.count_nonzero(selected > 64))
    tight_area = tight_width * tight_height
    if tight_width < 3 or tight_height < 3 or alpha_area < 0.12 * tight_area:
        return None
    rgba = cv2.cvtColor(crop, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = selected
    rgba = rgba[tight_y : tight_y + tight_height, tight_x : tight_x + tight_width]
    return Image.fromarray(rgba, mode="RGBA")


def build_cutout_library(
    coco: dict[str, object],
    images_directory: Path,
    output: Path,
    limit: int,
    rng: random.Random,
) -> dict[int, list[Image.Image]]:
    images = {int(record["id"]): record for record in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    image_ids = list(images)
    rng.shuffle(image_ids)
    library: dict[int, list[Image.Image]] = defaultdict(list)
    for image_id in image_ids:
        if all(len(library[category]) >= limit for category in range(1, 6)):
            break
        record = images[image_id]
        image = np.asarray(Image.open(images_directory / record["file_name"]).convert("RGB"))
        image_annotations = annotations_by_image[image_id]
        shuffled = list(image_annotations)
        rng.shuffle(shuffled)
        for annotation in shuffled:
            category = int(annotation["category_id"])
            if len(library[category]) >= limit:
                continue
            cutout = extract_cutout(image, annotation, image_annotations)
            if cutout is None:
                continue
            category_directory = output / "cutouts" / str(category)
            category_directory.mkdir(parents=True, exist_ok=True)
            cutout_path = category_directory / f"cutout_{len(library[category]):04d}.png"
            cutout.save(cutout_path)
            library[category].append(cutout)
    return library


def load_unity_cutouts(
    directory: Path,
    limit: int,
    rng: random.Random,
) -> dict[int, list[Image.Image]]:
    library: dict[int, list[Image.Image]] = defaultdict(list)
    for category in range(1, 6):
        paths = list((directory / str(category)).glob("*.png"))
        rng.shuffle(paths)
        for path in paths[:limit]:
            cutout = Image.open(path).convert("RGBA")
            alpha_bbox = cutout.getchannel("A").getbbox()
            if alpha_bbox is None:
                continue
            cutout = cutout.crop(alpha_bbox)
            if cutout.width >= 3 and cutout.height >= 3:
                library[category].append(cutout.copy())
    return library


def choose_position(
    anchor: dict[str, object],
    category: int,
    cutout_size: tuple[int, int],
    image_size: tuple[int, int],
    occupied: list[tuple[float, float, float, float]],
    background: Image.Image,
    rng: random.Random,
    semantic_placement: bool = False,
    surface_labels: np.ndarray | None = None,
    strict_vessel_water: bool = False,
) -> tuple[int, int, dict[str, float | int]] | None:
    image_width, image_height = image_size
    cutout_width, cutout_height = cutout_size
    ax, ay, aw, ah = [float(value) for value in anchor["bbox"]]
    anchor_center_x = ax + aw * 0.5
    anchor_center_y = ay + ah * 0.5
    base_distance = max(8.0, max(aw, ah) * 1.2)
    pixels = np.asarray(background.convert("RGB"), dtype=np.float32)
    context_padding = max(5, round(max(aw, ah) * 0.6))
    context_x0 = max(0, math.floor(ax) - context_padding)
    context_y0 = max(0, math.floor(ay) - context_padding)
    context_x1 = min(image_width, math.ceil(ax + aw) + context_padding)
    context_y1 = min(image_height, math.ceil(ay + ah) + context_padding)
    context = pixels[context_y0:context_y1, context_x0:context_x1]
    ring_mask = np.ones(context.shape[:2], dtype=bool)
    inner_x0 = max(0, math.floor(ax) - context_x0)
    inner_y0 = max(0, math.floor(ay) - context_y0)
    inner_x1 = min(context.shape[1], math.ceil(ax + aw) - context_x0)
    inner_y1 = min(context.shape[0], math.ceil(ay + ah) - context_y0)
    ring_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
    ring_pixels = context[ring_mask]
    if not len(ring_pixels):
        ring_pixels = context.reshape(-1, 3)
    anchor_median = np.median(ring_pixels, axis=0)
    anchor_std = np.std(ring_pixels, axis=0)
    anchor_features = surface_features(ring_pixels)
    dominant_surface: int | None = None
    connected_surface: np.ndarray | None = None
    anchor_components: set[int] = set()
    if semantic_placement and surface_labels is not None:
        anchor_surface_labels = surface_labels[context_y0:context_y1, context_x0:context_x1][ring_mask]
        if len(anchor_surface_labels):
            dominant_surface = int(np.bincount(anchor_surface_labels).argmax())
            surface_mask = (surface_labels == dominant_surface).astype(np.uint8)
            surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            _, connected_surface = cv2.connectedComponents(surface_mask, 8)
            ring_components = connected_surface[context_y0:context_y1, context_x0:context_x1][ring_mask]
            anchor_components = {int(value) for value in ring_components if int(value) > 0}
    best: tuple[float, int, int] | None = None
    valid_candidates = 0
    semantic_rejections = 0
    trials = 360 if strict_vessel_water and category == 5 else (240 if semantic_placement else 120)
    for _ in range(trials):
        angle = rng.uniform(0.0, math.tau)
        distance_extension = (
            12.0
            if strict_vessel_water and category == 5
            else (18.0 if semantic_placement and category == 5 else 25.0)
        )
        distance = rng.uniform(base_distance, base_distance + distance_extension)
        center_x = anchor_center_x + math.cos(angle) * distance
        center_y = anchor_center_y + math.sin(angle) * distance
        x = round(center_x - cutout_width * 0.5)
        y = round(center_y - cutout_height * 0.5)
        if x < 1 or y < 1 or x + cutout_width >= image_width - 1 or y + cutout_height >= image_height - 1:
            continue
        candidate = (float(x), float(y), float(cutout_width), float(cutout_height))
        if all(intersection_over_union(candidate, existing) < 0.02 for existing in occupied):
            candidate_pixels = pixels[y : y + cutout_height, x : x + cutout_width].reshape(-1, 3)
            candidate_box_features = surface_features(candidate_pixels)
            semantic_padding = max(4, round(max(cutout_width, cutout_height) * 0.8))
            semantic_x0 = max(0, x - semantic_padding)
            semantic_y0 = max(0, y - semantic_padding)
            semantic_x1 = min(image_width, x + cutout_width + semantic_padding)
            semantic_y1 = min(image_height, y + cutout_height + semantic_padding)
            semantic_pixels = pixels[semantic_y0:semantic_y1, semantic_x0:semantic_x1].reshape(-1, 3)
            candidate_features = surface_features(semantic_pixels)
            if dominant_surface is not None and connected_surface is not None:
                candidate_surface = surface_labels[y : y + cutout_height, x : x + cutout_width]
                match_fraction = float(np.mean(candidate_surface == dominant_surface))
                required_fraction = (
                    0.88
                    if strict_vessel_water and category == 5
                    else (0.62 if category == 5 else 0.50)
                )
                center_component = int(
                    connected_surface[
                        min(image_height - 1, y + cutout_height // 2),
                        min(image_width - 1, x + cutout_width // 2),
                    ]
                )
                if match_fraction < required_fraction or center_component not in anchor_components:
                    semantic_rejections += 1
                    continue
            if semantic_placement and not semantic_surface_allowed(
                category, anchor_features, candidate_features
            ):
                semantic_rejections += 1
                continue
            if (
                strict_vessel_water
                and category == 5
                and not strict_vessel_surface_allowed(anchor_features, candidate_box_features)
            ):
                semantic_rejections += 1
                continue
            valid_candidates += 1
            candidate_median = np.median(candidate_pixels, axis=0)
            candidate_std = np.std(candidate_pixels, axis=0)
            score = float(
                np.linalg.norm(candidate_median - anchor_median)
                + 0.35 * np.linalg.norm(candidate_std - anchor_std)
            )
            if semantic_placement:
                score += semantic_surface_distance(anchor_features, candidate_features)
            if best is None or score < best[0]:
                best = (score, x, y)
    if best is None:
        return None
    return (
        best[1],
        best[2],
        {
            "placement_score": best[0],
            "valid_candidates": valid_candidates,
            "semantic_rejections": semantic_rejections,
        },
    )


def surface_features(flat_rgb: np.ndarray) -> dict[str, float]:
    """Compute compact surface cues used for class-aware placement."""
    rgb = np.asarray(flat_rgb, dtype=np.float32).reshape(-1, 3)
    if not len(rgb):
        return {
            "gray_std": 0.0,
            "green_fraction": 0.0,
            "saturation": 0.0,
            "range_mean": 0.0,
            "red_median": 0.0,
            "blue_median": 0.0,
            "brightness": 0.0,
        }
    maximum = rgb.max(axis=1)
    minimum = rgb.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
    excess_green = 2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    return {
        "gray_std": float(np.std(gray)),
        "green_fraction": float(np.mean(excess_green > 18.0)),
        "saturation": float(np.median(saturation)),
        "range_mean": float(np.mean(maximum - minimum)),
        "red_median": float(np.median(rgb[:, 0])),
        "blue_median": float(np.median(rgb[:, 2])),
        "brightness": float(np.median(gray)),
    }


def semantic_surface_allowed(
    category: int,
    anchor: dict[str, float],
    candidate: dict[str, float],
) -> bool:
    """Reject vegetation/high-texture surfaces inconsistent with the class anchor."""
    if category == 5:  # vessels: preserve the anchor's water/harbor appearance
        if candidate["green_fraction"] > max(0.14, anchor["green_fraction"] + 0.06):
            return False
        if candidate["gray_std"] > max(24.0, anchor["gray_std"] * 1.65 + 3.0):
            return False
        if abs(candidate["saturation"] - anchor["saturation"]) > 0.24:
            return False
        blue_red_ratio = (candidate["blue_median"] + 8.0) / (candidate["red_median"] + 8.0)
        if blue_red_ratio < 0.78:
            return False
    else:  # vehicles/equipment: avoid vegetation and highly heterogeneous roof edges
        if candidate["green_fraction"] > max(0.22, anchor["green_fraction"] + 0.10):
            return False
        if candidate["gray_std"] > max(38.0, anchor["gray_std"] * 2.0 + 5.0):
            return False
    return True


def semantic_surface_distance(
    anchor: dict[str, float], candidate: dict[str, float]
) -> float:
    return float(
        0.55 * abs(candidate["gray_std"] - anchor["gray_std"])
        + 42.0 * abs(candidate["green_fraction"] - anchor["green_fraction"])
        + 28.0 * abs(candidate["saturation"] - anchor["saturation"])
        + 0.18 * abs(candidate["range_mean"] - anchor["range_mean"])
    )


def strict_vessel_surface_allowed(
    anchor: dict[str, float], candidate: dict[str, float]
) -> bool:
    """Tight local appearance gate used only by the vessel dose experiment."""
    if candidate["green_fraction"] > max(0.10, anchor["green_fraction"] + 0.035):
        return False
    if candidate["gray_std"] > max(18.0, anchor["gray_std"] * 1.30 + 2.0):
        return False
    if abs(candidate["saturation"] - anchor["saturation"]) > 0.16:
        return False
    if abs(candidate["brightness"] - anchor["brightness"]) > 42.0:
        return False
    anchor_ratio = (anchor["blue_median"] + 8.0) / (anchor["red_median"] + 8.0)
    candidate_ratio = (candidate["blue_median"] + 8.0) / (candidate["red_median"] + 8.0)
    # Brown/gray docks can match a docked vessel's local cluster.  Requiring
    # neutral-to-blue reflectance makes the vessel mask conservative and keeps
    # those hard surfaces out even when the anchor itself touches a quay.
    if candidate_ratio < max(0.98, anchor_ratio - 0.18):
        return False
    return True


def build_surface_labels(background: Image.Image, seed: int) -> np.ndarray:
    """Quantize image surfaces in Lab space for connected-surface placement."""
    rgb = np.asarray(background.convert("RGB"))
    height, width = rgb.shape[:2]
    small_width = min(160, width)
    small_height = min(160, height)
    small = cv2.resize(rgb, (small_width, small_height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(int(seed & 0x7FFFFFFF))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.4)
    _, labels, _ = cv2.kmeans(lab, 10, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    label_image = labels.reshape(small_height, small_width).astype(np.uint8)
    return cv2.resize(label_image, (width, height), interpolation=cv2.INTER_NEAREST)


def paste_cutout(
    background: Image.Image,
    cutout: Image.Image,
    position: tuple[int, int],
    category: int,
    rng: random.Random,
    sensor_aware: bool = False,
    vessel_render_v2: bool = False,
) -> None:
    x, y = position
    local = background.crop((x, y, x + cutout.width, y + cutout.height)).convert("RGB")
    local_brightness = float(np.asarray(local, dtype=np.float32).mean())
    cutout_rgb = np.asarray(cutout.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(cutout.getchannel("A"), dtype=np.float32) / 255.0
    object_pixels = cutout_rgb[alpha > 0.35]
    object_brightness = float(object_pixels.mean()) if len(object_pixels) else local_brightness
    brightness_factor = max(0.78, min(1.22, local_brightness / max(object_brightness, 1.0)))
    adjusted = ImageEnhance.Brightness(cutout).enhance(brightness_factor)
    if vessel_render_v2 and category == 5:
        adjusted = ImageEnhance.Color(adjusted).enhance(rng.uniform(0.72, 0.98))
        adjusted = ImageEnhance.Contrast(adjusted).enhance(rng.uniform(0.68, 0.84))
    else:
        adjusted = ImageEnhance.Color(adjusted).enhance(rng.uniform(0.38, 0.68))
        adjusted = ImageEnhance.Contrast(adjusted).enhance(rng.uniform(0.82, 0.96))

    if sensor_aware:
        adjusted = match_local_color(adjusted, local)
        gray = cv2.cvtColor(np.asarray(local), cv2.COLOR_RGB2GRAY)
        local_sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if vessel_render_v2 and category == 5:
            # The Unity cutout is already rendered at target GSD. Avoid the v5
            # second resize cycle, which erased deck-scale structure, and use a
            # single weak PSF after appearance matching.
            blur_sigma = max(0.08, min(0.32, 0.38 - 0.04 * math.log1p(local_sharpness)))
        else:
            blur_sigma = max(0.28, min(0.88, 0.98 - 0.10 * math.log1p(local_sharpness)))
        if min(adjusted.size) >= 8 and not (vessel_render_v2 and category == 5):
            downscale = rng.uniform(0.82, 0.94)
            small = adjusted.resize(
                (max(3, round(adjusted.width * downscale)), max(3, round(adjusted.height * downscale))),
                Image.Resampling.BILINEAR,
            )
            adjusted = small.resize(adjusted.size, Image.Resampling.BILINEAR)
        adjusted = adjusted.filter(ImageFilter.GaussianBlur(blur_sigma))
    else:
        adjusted = adjusted.filter(ImageFilter.GaussianBlur(rng.uniform(0.35, 0.75)))

    shadow_strength = {1: 0.12, 2: 0.13, 3: 0.14, 4: 0.16, 5: 0.055}[category]
    shadow_blur = max(0.8, min(2.2, max(adjusted.size) * 0.07))
    shadow_alpha = adjusted.getchannel("A").filter(ImageFilter.GaussianBlur(shadow_blur)).point(
        lambda value: int(value * shadow_strength)
    )
    shadow = Image.new("RGBA", adjusted.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    offset = 1 if category == 5 else max(1, round(max(adjusted.size) * 0.07))
    background.alpha_composite(shadow, (x + offset, y + offset))
    background.alpha_composite(adjusted, (x, y))


def match_local_color(cutout: Image.Image, local: Image.Image) -> Image.Image:
    rgba = np.asarray(cutout, dtype=np.float32).copy()
    alpha = rgba[:, :, 3] / 255.0
    mask = alpha > 0.35
    if not np.any(mask):
        return cutout
    local_rgb = np.asarray(local.convert("RGB"), dtype=np.float32)
    object_median = np.median(rgba[:, :, :3][mask], axis=0)
    local_median = np.median(local_rgb.reshape(-1, 3), axis=0)
    shift = np.clip((local_median - object_median) * 0.12, -14.0, 14.0)
    rgba[:, :, :3] = np.clip(rgba[:, :, :3] + shift.reshape(1, 1, 3) * alpha[:, :, None], 0, 255)
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def main() -> int:
    args = parse_args()
    if args.vessel_min_bbox_scale < 0 or args.vessel_max_bbox_scale < 0:
        raise ValueError("Vessel bbox-scale limits must be non-negative")
    if (
        args.vessel_max_bbox_scale > 0
        and args.vessel_max_bbox_scale <= args.vessel_min_bbox_scale
    ):
        raise ValueError("--vessel-max-bbox-scale must exceed the minimum")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    appearance_rng = random.Random(args.seed ^ 0x5EEDBEEF)
    target_counts = parse_target_counts(args.target_counts)
    real = load_json(args.real_coco)
    if args.unity_cutouts:
        library = load_unity_cutouts(args.unity_cutouts, args.cutouts_per_class, rng)
    else:
        if not args.synthetic_coco or not args.synthetic_images:
            raise ValueError("Provide --unity-cutouts or both synthetic COCO/image inputs.")
        synthetic = load_json(args.synthetic_coco)
        library = build_cutout_library(
            synthetic,
            args.synthetic_images,
            args.output,
            args.cutouts_per_class,
            rng,
        )
    if any(not library[category] for category in range(1, 6)):
        raise RuntimeError(f"Incomplete cutout library: { {key: len(value) for key, value in library.items()} }")

    real_annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in real["annotations"]:
        real_annotations_by_image[int(annotation["image_id"])].append(annotation)
    output_images = args.output / "images" / "train"
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations = []
    output_image_records = []
    next_annotation_id = 1
    added_counts = Counter()
    placement_scores: list[float] = []
    inserted_bbox_scales: list[float] = []
    vessel_scale_rejections = 0
    semantic_rejections = 0
    valid_candidate_count = 0
    augmented_image_count = 0
    remaining_anchor_images = Counter(
        TYPE_TO_CATEGORY.get(int(record.get("anchor_xview_type_id", 0)))
        for record in real["images"]
    )

    for output_image_id, image_record in enumerate(real["images"], start=1):
        source_path = args.real_images / image_record["file_name"]
        background = Image.open(source_path).convert("RGBA")
        placement_background = background.copy()
        surface_labels = (
            build_surface_labels(placement_background, args.seed + output_image_id)
            if args.semantic_placement
            else None
        )
        image_annotations = real_annotations_by_image[int(image_record["id"])]
        occupied = [tuple(float(value) for value in annotation["bbox"]) for annotation in image_annotations]
        anchor_category = TYPE_TO_CATEGORY.get(int(image_record.get("anchor_xview_type_id", 0)))
        anchors = [annotation for annotation in image_annotations if int(annotation["category_id"]) == anchor_category]
        if not anchors:
            anchors = list(image_annotations)
        image_insertions = 0

        desired_insertions = args.insertions_per_image
        if target_counts and anchor_category in target_counts:
            remaining_target = max(
                0,
                target_counts[anchor_category]
                + args.target_schedule_buffer
                - added_counts[anchor_category],
            )
            remaining_images = max(1, remaining_anchor_images[anchor_category])
            desired_insertions = min(
                args.max_insertions_per_image,
                math.ceil(remaining_target / remaining_images),
            )
            if (
                remaining_target > 0
                and args.target_reserve_window > 0
                and remaining_images <= args.target_reserve_window
            ):
                desired_insertions = min(
                    args.max_insertions_per_image, desired_insertions + 1
                )

        for insertion_index in range(desired_insertions):
            if not anchors:
                break
            placement = None
            anchor = None
            category = 0
            cutout = None
            for _ in range(args.placement_retries):
                anchor = rng.choice(anchors)
                category = int(anchor["category_id"])
                if target_counts and added_counts[category] >= target_counts[category]:
                    break
                source_cutout = rng.choice(library[category])
                scale = rng.uniform(0.88, 1.12)
                cutout = source_cutout.resize(
                    (
                        max(3, round(source_cutout.width * scale)),
                        max(3, round(source_cutout.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                bbox_scale = math.sqrt(float(cutout.width * cutout.height))
                if category == 5 and (
                    (
                        args.vessel_min_bbox_scale > 0
                        and bbox_scale < args.vessel_min_bbox_scale
                    )
                    or (
                        args.vessel_max_bbox_scale > 0
                        and bbox_scale >= args.vessel_max_bbox_scale
                    )
                ):
                    vessel_scale_rejections += 1
                    continue
                placement = choose_position(
                    anchor,
                    category,
                    cutout.size,
                    placement_background.size,
                    occupied,
                    placement_background,
                    rng,
                    args.semantic_placement,
                    surface_labels,
                    args.strict_vessel_water,
                )
                if placement is not None:
                    break
            if placement is None:
                continue
            assert anchor is not None and cutout is not None
            position = (placement[0], placement[1])
            placement_diagnostics = placement[2]
            placement_scores.append(float(placement_diagnostics["placement_score"]))
            semantic_rejections += int(placement_diagnostics["semantic_rejections"])
            valid_candidate_count += int(placement_diagnostics["valid_candidates"])
            paste_cutout(
                background,
                cutout,
                position,
                category,
                appearance_rng,
                args.sensor_aware,
                args.vessel_render_v2,
            )
            bbox = (float(position[0]), float(position[1]), float(cutout.width), float(cutout.height))
            inserted_bbox_scales.append(math.sqrt(bbox[2] * bbox[3]))
            occupied.append(bbox)
            output_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": output_image_id,
                    "category_id": category,
                    "bbox": list(bbox),
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "source": "unity_cutout",
                    "placement_score": float(placement_diagnostics["placement_score"]),
                }
            )
            next_annotation_id += 1
            added_counts[category] += 1
            image_insertions += 1

        if anchor_category in remaining_anchor_images:
            remaining_anchor_images[anchor_category] -= 1

        for annotation in image_annotations:
            copied = {
                "id": next_annotation_id,
                "image_id": output_image_id,
                "category_id": int(annotation["category_id"]),
                "bbox": [float(value) for value in annotation["bbox"]],
                "area": float(annotation["area"]),
                "iscrowd": int(annotation.get("iscrowd", 0)),
                "source": "xview_real",
            }
            output_annotations.append(copied)
            next_annotation_id += 1

        file_name = f"hyb_{output_image_id:06d}.jpg"
        destination_path = output_images / file_name
        if image_insertions:
            background.convert("RGB").save(destination_path, quality=95, subsampling=0)
            augmented_image_count += 1
        else:
            link_or_copy(source_path, destination_path)
        output_record = dict(image_record)
        output_record.update(
            {
                "id": output_image_id,
                "file_name": file_name,
                "width": int(image_record["width"]),
                "height": int(image_record["height"]),
                "source_patch": image_record["file_name"],
            }
        )
        output_image_records.append(output_record)

    coco = {
        "info": {
            "description": "xView real backgrounds with Unity object cutout augmentation",
            "version": "1.0.0",
            "seed": args.seed,
            "semantic_placement": args.semantic_placement,
            "sensor_aware": args.sensor_aware,
            "vessel_render_v2": args.vessel_render_v2,
            "vessel_min_bbox_scale": args.vessel_min_bbox_scale,
            "vessel_max_bbox_scale": args.vessel_max_bbox_scale,
            "strict_vessel_water": args.strict_vessel_water,
        },
        "images": output_image_records,
        "annotations": output_annotations,
        "categories": real["categories"],
    }
    annotation_directory = args.output / "annotations"
    annotation_directory.mkdir(parents=True, exist_ok=True)
    (annotation_directory / "instances_train.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "image_count": len(output_image_records),
        "annotation_count": len(output_annotations),
        "real_annotation_count": len(real["annotations"]),
        "unity_annotation_count": sum(added_counts.values()),
        "augmented_image_count": augmented_image_count,
        "unchanged_image_count": len(output_image_records) - augmented_image_count,
        "unity_annotation_count_by_category": dict(sorted(added_counts.items())),
        "cutout_count_by_category": {str(category): len(library[category]) for category in range(1, 6)},
        "semantic_placement": args.semantic_placement,
        "sensor_aware": args.sensor_aware,
        "vessel_render_v2": args.vessel_render_v2,
        "vessel_min_bbox_scale": args.vessel_min_bbox_scale,
        "vessel_max_bbox_scale": args.vessel_max_bbox_scale,
        "vessel_scale_rejection_count": vessel_scale_rejections,
        "unity_bbox_scale_min": float(np.min(inserted_bbox_scales)) if inserted_bbox_scales else None,
        "unity_bbox_scale_p50": float(np.median(inserted_bbox_scales)) if inserted_bbox_scales else None,
        "unity_bbox_scale_p95": float(np.percentile(inserted_bbox_scales, 95)) if inserted_bbox_scales else None,
        "unity_bbox_scale_max": float(np.max(inserted_bbox_scales)) if inserted_bbox_scales else None,
        "strict_vessel_water": args.strict_vessel_water,
        "placement_score_mean": float(np.mean(placement_scores)) if placement_scores else None,
        "placement_score_p95": float(np.percentile(placement_scores, 95)) if placement_scores else None,
        "semantic_rejection_count": semantic_rejections,
        "valid_candidate_count": valid_candidate_count,
        "target_count_by_category": target_counts,
        "target_schedule_buffer": args.target_schedule_buffer,
        "target_count_met": (
            all(added_counts[category] == target_counts[category] for category in range(1, 6))
            if target_counts
            else None
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
