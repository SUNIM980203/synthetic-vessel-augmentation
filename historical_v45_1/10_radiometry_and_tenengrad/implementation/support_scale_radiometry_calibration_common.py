"""Shared, training-free machinery for prospective radiometry calibration.

This module deliberately contains no detector training or prediction call.  The
only model operation is frozen intermediate-feature extraction for the support
metric that was specified before this calibration study.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import wasserstein_distance
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_vessel_feature_support import extract_embeddings  # noqa: E402


METRICS = (
    "absolute_weber_contrast",
    "local_cnr",
    "tenengrad_ratio",
    "laplacian_variance_ratio",
    "lab_delta_e76",
)
EXTENDED_METRICS = (
    "foreground_luminance_median",
    "background_luminance_median",
    "foreground_background_luminance_ratio",
    "signed_weber_contrast",
    *METRICS,
    "foreground_gradient_mean",
    "background_gradient_mean",
    "edge_energy_ratio",
    "boundary_sharpness_ratio",
)

FROZEN_THRESHOLDS = {
    "group_standardized_wasserstein_max": 0.50,
    "paired_median_absolute_standardized_difference_max": 0.10,
    "pair_absolute_log_scale_max": 0.02,
    "pair_absolute_log_aspect_max": 0.05,
    "pair_absolute_alpha_occupancy_max": 0.05,
    "pair_each_radiometry_standardized_difference_max": 0.15,
    "support_required_hosts": 30,
    "support_median_gap_min": 0.06,
    "support_fraction_gap_ge_0_04_min": 0.80,
    "support_minimum_gap": 0.005,
}

# The full parameter bank is prespecified before any calibration result is
# calculated.  It is intentionally small and interpretable: one local-contrast
# control and one local high-frequency control.
CALIBRATION_PARAMETER_BANK = tuple(
    {
        "profile_id": f"contrast_{gain:.2f}_unsharp_{amount:.2f}",
        "local_contrast_gain": gain,
        "unsharp_amount": amount,
        "unsharp_sigma": 0.85,
        "difference_floor_rgb": 4.0,
        "difference_full_rgb": 18.0,
        "mask_close_kernel": 3,
        "mask_dilate_kernel": 3,
        "mask_feather_sigma": 0.80,
    }
    for gain in (1.00, 1.25, 1.50, 1.75)
    for amount in (0.00, 0.15, 0.30)
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clipped_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(x1 + 1, min(width, int(round(x + w))))
    y2 = max(y1 + 1, min(height, int(round(y + h))))
    return x1, y1, x2, y2


def object_and_ring_masks(
    shape: tuple[int, int], bbox: list[float], other_annotations: list[dict] | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = shape
    x1, y1, x2, y2 = clipped_box(bbox, width, height)
    pad = max(6, int(round(0.30 * max(x2 - x1, y2 - y1))))
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(width, x2 + pad), min(height, y2 + pad)
    obj = np.zeros((height, width), dtype=bool)
    obj[y1:y2, x1:x2] = True
    ring = np.zeros((height, width), dtype=bool)
    ring[oy1:oy2, ox1:ox2] = True
    ring[obj] = False
    for annotation in other_annotations or []:
        ax1, ay1, ax2, ay2 = clipped_box(annotation["bbox"], width, height)
        ring[ay1:ay2, ax1:ax2] = False
    if int(ring.sum()) < 50:
        raise ValueError("Insufficient local background ring pixels")
    return obj, ring, (x1, y1, x2, y2)


def extended_appearance_metrics(
    rgb: np.ndarray, bbox: list[float], other_annotations: list[dict] | None = None
) -> dict[str, float]:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    obj, ring, (x1, y1, x2, y2) = object_and_ring_masks(rgb.shape[:2], bbox, other_annotations)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[:, :, 0] * (100.0 / 255.0)
    obj_l, bg_l = luminance[obj], luminance[ring]
    obj_med, bg_med = float(np.median(obj_l)), float(np.median(bg_l))
    bg_mad = float(np.median(np.abs(bg_l - bg_med)))
    signed_weber = (obj_med - bg_med) / max(abs(bg_med), 1.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    boundary = np.zeros_like(obj)
    boundary[y1:y2, x1:x2] = True
    inner = cv2.erode(boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    boundary &= ~inner
    obj_lab = np.median(lab[obj], axis=0)
    bg_lab = np.median(lab[ring], axis=0)
    obj_grad = float(np.mean(grad[obj]))
    bg_grad = float(np.mean(grad[ring]))
    obj_energy = float(np.mean((gx[obj] ** 2 + gy[obj] ** 2)))
    bg_energy = float(np.mean((gx[ring] ** 2 + gy[ring] ** 2)))
    boundary_grad = float(np.mean(grad[boundary])) if bool(boundary.any()) else math.nan
    return {
        "foreground_luminance_median": obj_med,
        "background_luminance_median": bg_med,
        "foreground_background_luminance_ratio": obj_med / max(bg_med, 1.0),
        "signed_weber_contrast": signed_weber,
        "absolute_weber_contrast": abs(signed_weber),
        "local_cnr": abs(obj_med - bg_med) / max(1.4826 * bg_mad, 0.5),
        "tenengrad_ratio": obj_grad / max(bg_grad, 1e-5),
        "laplacian_variance_ratio": float(np.var(lap[obj], ddof=1) / max(np.var(lap[ring], ddof=1), 1e-7)),
        "lab_delta_e76": float(np.linalg.norm(obj_lab - bg_lab)),
        "foreground_gradient_mean": obj_grad,
        "background_gradient_mean": bg_grad,
        "edge_energy_ratio": obj_energy / max(bg_energy, 1e-7),
        "boundary_sharpness_ratio": boundary_grad / max(bg_grad, 1e-5),
    }


def difference_soft_mask(
    candidate_rgb: np.ndarray, clean_rgb: np.ndarray, bbox: list[float], parameters: dict
) -> np.ndarray:
    height, width = candidate_rgb.shape[:2]
    x1, y1, x2, y2 = clipped_box(bbox, width, height)
    difference = np.max(
        np.abs(candidate_rgb.astype(np.float32) - clean_rgb.astype(np.float32)), axis=2
    )
    floor = float(parameters["difference_floor_rgb"])
    full = float(parameters["difference_full_rgb"])
    soft = np.clip((difference - floor) / max(full - floor, 1e-6), 0.0, 1.0)
    binary = (soft > 0.08).astype(np.uint8)
    close_kernel = int(parameters["mask_close_kernel"])
    dilate_kernel = int(parameters["mask_dilate_kernel"])
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if dilate_kernel > 1:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
    soft = np.maximum(soft, 0.65 * binary.astype(np.float32))
    feather = float(parameters["mask_feather_sigma"])
    if feather > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), feather)
    support = np.zeros((height, width), dtype=np.float32)
    support[y1:y2, x1:x2] = 1.0
    return np.clip(soft * support, 0.0, 1.0)


def apply_frozen_calibration(
    candidate_rgb: np.ndarray, clean_rgb: np.ndarray, bbox: list[float], parameters: dict
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the same label-blind, deterministic local rule to any candidate."""
    if candidate_rgb.shape != clean_rgb.shape:
        raise ValueError("Candidate and clean host shapes differ")
    obj, ring, _ = object_and_ring_masks(candidate_rgb.shape[:2], bbox)
    soft_mask = difference_soft_mask(candidate_rgb, clean_rgb, bbox, parameters)
    lab = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_l = float(np.median(lab[:, :, 0][ring]))
    luminance = lab[:, :, 0]
    contrast = background_l + float(parameters["local_contrast_gain"]) * (luminance - background_l)
    sigma = float(parameters["unsharp_sigma"])
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigma)
    sharpened = contrast + float(parameters["unsharp_amount"]) * (contrast - blurred)
    calibrated_l = luminance * (1.0 - soft_mask) + sharpened * soft_mask
    calibrated_lab = lab.copy()
    calibrated_lab[:, :, 0] = np.clip(calibrated_l, 0, 255)
    output = cv2.cvtColor(calibrated_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    audit = {
        "derived_mask_fraction_in_bbox": float(np.mean(soft_mask[obj] > 0.05)),
        "derived_mask_mean_in_bbox": float(np.mean(soft_mask[obj])),
        "host_background_lab_l": background_l,
        "maximum_absolute_rgb_change": float(
            np.max(np.abs(output.astype(np.int16) - candidate_rgb.astype(np.int16)))
        ),
        "mean_absolute_rgb_change_in_bbox": float(
            np.mean(np.abs(output.astype(np.float32) - candidate_rgb.astype(np.float32))[obj])
        ),
    }
    return output, audit


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_final_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, quality=95, subsampling=0)


def native_metrics(path: Path) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source"] == "Native real":
                rows.append([float(row[key]) for key in METRICS])
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (150, len(METRICS)):
        raise ValueError(f"Expected 150 native proxy rows, got {values.shape}")
    summary: dict[str, dict[str, float]] = {}
    for index, key in enumerate(METRICS):
        column = values[:, index]
        summary[key] = {
            "mean": float(np.mean(column)),
            "sample_sd": float(np.std(column, ddof=1)),
            "median": float(np.median(column)),
            "q1": float(np.quantile(column, 0.25)),
            "q3": float(np.quantile(column, 0.75)),
        }
    return values, summary


def distribution_gate(values: np.ndarray, native: np.ndarray, native_summary: dict) -> dict:
    result: dict[str, dict] = {}
    for index, key in enumerate(METRICS):
        column = values[:, index]
        q1, median, q3 = [float(np.quantile(column, q)) for q in (0.25, 0.50, 0.75)]
        sd = max(float(native_summary[key]["sample_sd"]), 1e-12)
        w = float(wasserstein_distance(column, native[:, index]) / sd)
        median_inside = float(native_summary[key]["q1"]) <= median <= float(native_summary[key]["q3"])
        overlap = max(q1, float(native_summary[key]["q1"])) <= min(q3, float(native_summary[key]["q3"]))
        passed = bool(median_inside and overlap and w <= FROZEN_THRESHOLDS["group_standardized_wasserstein_max"])
        result[key] = {
            "mean": float(np.mean(column)),
            "sample_sd": float(np.std(column, ddof=1)),
            "q1": q1,
            "median": median,
            "q3": q3,
            "native_q1": float(native_summary[key]["q1"]),
            "native_median": float(native_summary[key]["median"]),
            "native_q3": float(native_summary[key]["q3"]),
            "signed_median_difference": median - float(native_summary[key]["median"]),
            "standardized_wasserstein": w,
            "median_inside_native_iqr": bool(median_inside),
            "iqr_overlap": bool(overlap),
            "pass": passed,
        }
    return {"metrics": result, "pass": bool(all(row["pass"] for row in result.values()))}


def evaluate_selected_pairs(
    pairs: list[dict], candidates: dict[int, dict], native: np.ndarray, native_summary: dict
) -> dict:
    near = np.asarray(
        [[float(candidates[int(row["near_candidate_id"])]["calibrated_metrics"][key]) for key in METRICS] for row in pairs],
        dtype=np.float64,
    )
    far = np.asarray(
        [[float(candidates[int(row["far_candidate_id"])]["calibrated_metrics"][key]) for key in METRICS] for row in pairs],
        dtype=np.float64,
    )
    gaps = np.asarray([float(row["support_gap"]) for row in pairs], dtype=np.float64)
    support = {
        "selected_host_count": len(pairs),
        "all_required_hosts_matched": len(pairs) == int(FROZEN_THRESHOLDS["support_required_hosts"]),
        "median_gap": float(np.median(gaps)) if len(gaps) else None,
        "fraction_gap_ge_0_04": float(np.mean(gaps >= 0.04)) if len(gaps) else 0.0,
        "minimum_gap": float(np.min(gaps)) if len(gaps) else None,
    }
    support["pass"] = bool(
        support["all_required_hosts_matched"]
        and support["median_gap"] >= FROZEN_THRESHOLDS["support_median_gap_min"]
        and support["fraction_gap_ge_0_04"] >= FROZEN_THRESHOLDS["support_fraction_gap_ge_0_04_min"]
        and support["minimum_gap"] >= FROZEN_THRESHOLDS["support_minimum_gap"]
    )
    paired_metrics: dict[str, dict] = {}
    for index, key in enumerate(METRICS):
        sd = max(float(native_summary[key]["sample_sd"]), 1e-12)
        value = float(np.median(np.abs(near[:, index] - far[:, index])) / sd)
        paired_metrics[key] = {
            "median_absolute_standardized_difference": value,
            "pass": bool(value <= FROZEN_THRESHOLDS["paired_median_absolute_standardized_difference_max"]),
        }
    near_gate = distribution_gate(near, native, native_summary)
    far_gate = distribution_gate(far, native, native_summary)
    radiometry = {
        "relative_near": near_gate,
        "relative_far": far_gate,
        "paired_standardized_difference": {
            "metrics": paired_metrics,
            "pass": bool(all(row["pass"] for row in paired_metrics.values())),
        },
    }
    radiometry["pass"] = bool(
        near_gate["pass"] and far_gate["pass"] and radiometry["paired_standardized_difference"]["pass"]
    )
    failed = 0
    ratios: list[float] = []
    for group in (near_gate, far_gate):
        for row in group["metrics"].values():
            failed += int(not row["pass"])
            ratios.append(float(row["standardized_wasserstein"]) / 0.50)
    for row in paired_metrics.values():
        failed += int(not row["pass"])
        ratios.append(float(row["median_absolute_standardized_difference"]) / 0.10)
    failed += int(not support["pass"])
    if len(gaps):
        ratios.extend((
            0.06 / max(float(np.median(gaps)), 1e-12),
            0.80 / max(float(np.mean(gaps >= 0.04)), 1e-12),
            0.005 / max(float(np.min(gaps)), 1e-12),
        ))
    return {
        "support_gate": support,
        "radiometry_gate": radiometry,
        "failed_condition_count": int(failed),
        "maximum_constraint_ratio": float(max(ratios)) if ratios else math.inf,
        "joint_pass": bool(support["pass"] and radiometry["pass"]),
    }


def load_frozen_model(weights: Path, device: str = "0") -> tuple[torch.nn.Module, torch.device]:
    torch_device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu")
    model = YOLO(str(weights)).model.to(torch_device).eval()
    return model, torch_device


def records_for_candidates(
    candidate_ids: list[int], coco: dict, image_path_by_id: dict[int, Path]
) -> list[dict]:
    images = {int(row["id"]): row for row in coco["images"]}
    annotations = {int(row["id"]): row for row in coco["annotations"]}
    records = []
    for candidate_id in candidate_ids:
        annotation = annotations[candidate_id]
        image = images[int(annotation["image_id"])]
        bbox = [float(value) for value in annotation["bbox"]]
        records.append({
            "annotation_id": candidate_id,
            "image_id": int(annotation["image_id"]),
            "file_name": str(image_path_by_id[candidate_id].name),
            "image_path": str(image_path_by_id[candidate_id]),
            "source_scene": str(image.get("source_scene", image["file_name"])),
            "image_width": int(image["width"]),
            "image_height": int(image["height"]),
            "bbox": bbox,
            "bbox_area": float(bbox[2] * bbox[3]),
            "bbox_scale": float(math.sqrt(max(bbox[2] * bbox[3], 1.0))),
        })
    return records


def scene_lookup(coco: dict) -> dict[int, str]:
    images = {int(row["id"]): row for row in coco["images"]}
    return {
        int(row["id"]): str(images[int(row["image_id"])].get("source_scene", images[int(row["image_id"])]["file_name"]))
        for row in coco["annotations"] if int(row["category_id"]) == 5
    }


def nearest_real_distances(
    candidate_ids: list[int], candidate_embeddings: np.ndarray, candidate_coco: dict,
    real_npz_path: Path, real_coco_path: Path,
) -> dict[int, float]:
    real_npz = np.load(real_npz_path)
    real_ids = real_npz["annotation_ids"].astype(int)
    real_scales = real_npz["bbox_scales"].astype(float)
    real_embeddings = real_npz["embeddings"].astype(np.float32)
    medium = (real_scales >= 32.0) & (real_scales < 64.0)
    real_ids, real_embeddings = real_ids[medium], real_embeddings[medium]
    candidate_scenes = scene_lookup(candidate_coco)
    real_scenes_lookup = scene_lookup(load_json(real_coco_path))
    real_scenes = np.asarray([real_scenes_lookup[int(annotation_id)] for annotation_id in real_ids])
    nearest = np.full(len(candidate_ids), np.inf, dtype=np.float32)
    for start in range(0, len(candidate_ids), 512):
        stop = min(start + 512, len(candidate_ids))
        distances = np.clip(1.0 - candidate_embeddings[start:stop] @ real_embeddings.T, 0.0, 2.0)
        for local_index, candidate_id in enumerate(candidate_ids[start:stop]):
            distances[local_index, real_scenes == candidate_scenes[int(candidate_id)]] = np.inf
        nearest[start:stop] = np.min(distances, axis=1)
    if not bool(np.all(np.isfinite(nearest))):
        raise RuntimeError("At least one candidate has no cross-scene real reference")
    return {int(candidate_id): float(value) for candidate_id, value in zip(candidate_ids, nearest)}


def extract_candidate_support(
    model: torch.nn.Module, device: torch.device, candidate_ids: list[int], candidate_coco: dict,
    image_path_by_id: dict[int, Path], real_npz_path: Path, real_coco_path: Path,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[int, float]]:
    records = records_for_candidates(candidate_ids, candidate_coco, image_path_by_id)
    embeddings = extract_embeddings(model, records, 640, batch_size, device)
    distances = nearest_real_distances(
        candidate_ids, embeddings, candidate_coco, real_npz_path, real_coco_path
    )
    return embeddings, distances


def fixed_pair_from_ids(
    first_id: int, second_id: int, distances: dict[int, float], candidates: dict[int, dict]
) -> dict:
    if distances[first_id] <= distances[second_id]:
        near_id, far_id = first_id, second_id
    else:
        near_id, far_id = second_id, first_id
    return {
        "host_index": int(candidates[near_id]["host_index"]),
        "near_candidate_id": near_id,
        "far_candidate_id": far_id,
        "near_distance": float(distances[near_id]),
        "far_distance": float(distances[far_id]),
        "support_gap": float(distances[far_id] - distances[near_id]),
    }


def pair_geometry(candidate: dict) -> tuple:
    bbox = tuple(float(value) for value in candidate["bbox"])
    return bbox + (bool(candidate.get("rotated_90", False)),)


def exact_pair_eligible(left: dict, right: dict, native_summary: dict) -> tuple[bool, dict]:
    exact_geometry = pair_geometry(left) == pair_geometry(right)
    occupancy = abs(float(left["alpha_occupancy"]) - float(right["alpha_occupancy"]))
    standardized = {}
    for key in METRICS:
        sd = max(float(native_summary[key]["sample_sd"]), 1e-12)
        standardized[key] = abs(
            float(left["calibrated_metrics"][key]) - float(right["calibrated_metrics"][key])
        ) / sd
    passed = bool(
        exact_geometry
        and occupancy <= FROZEN_THRESHOLDS["pair_absolute_alpha_occupancy_max"]
        and all(value <= FROZEN_THRESHOLDS["pair_each_radiometry_standardized_difference_max"] for value in standardized.values())
    )
    return passed, {
        "exact_geometry_orientation": exact_geometry,
        "absolute_occupancy_difference": occupancy,
        "standardized_metric_differences": standardized,
    }


def select_exact_pairs(
    candidates: list[dict], distances: dict[int, float], native: np.ndarray, native_summary: dict
) -> tuple[list[dict], list[dict]]:
    """Apply one frozen exact-matching selector; there is no fresh-data grid."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[int(row["host_index"])].append(row)
    pools: dict[int, list[dict]] = {}
    host_audit: list[dict] = []
    for host in sorted(grouped):
        rows = sorted(grouped[host], key=lambda row: int(row["candidate_id"]))
        feasible = []
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                passed, audit = exact_pair_eligible(left, right, native_summary)
                if not passed:
                    continue
                left_id, right_id = int(left["candidate_id"]), int(right["candidate_id"])
                pair = fixed_pair_from_ids(left_id, right_id, distances, {left_id: left, right_id: right})
                if pair["support_gap"] < FROZEN_THRESHOLDS["support_minimum_gap"]:
                    continue
                pair.update(audit)
                pair["near"] = left if pair["near_candidate_id"] == left_id else right
                pair["far"] = right if pair["far_candidate_id"] == right_id else left
                pair["target_cost"] = sum(
                    abs(
                        0.5 * (float(pair["near"]["calibrated_metrics"][key]) + float(pair["far"]["calibrated_metrics"][key]))
                        - float(pair["near"]["winsorized_target_metrics"][key])
                    ) / max(float(native_summary[key]["sample_sd"]), 1e-12)
                    for key in METRICS
                )
                pair["matching_cost"] = occupancy_matching_cost(pair, native_summary)
                feasible.append(pair)
        tier_006 = [row for row in feasible if row["support_gap"] >= 0.06]
        tier_004 = [row for row in feasible if row["support_gap"] >= 0.04]
        tier = tier_006 or tier_004 or feasible
        tier_name = "gap_ge_0_06" if tier_006 else ("gap_ge_0_04" if tier_004 else "gap_ge_0_005")
        tier = sorted(
            tier,
            key=lambda row: (
                row["target_cost"] + row["matching_cost"] - 4.0 * row["support_gap"],
                row["near_candidate_id"], row["far_candidate_id"],
            ),
        )[:192]
        for row in tier:
            row["selection_tier"] = tier_name
        if tier:
            pools[host] = tier
        host_audit.append({
            "host_index": host,
            "candidate_count": len(rows),
            "eligible_exact_pair_count": len(feasible),
            "gap_ge_0_04_pair_count": len(tier_004),
            "gap_ge_0_06_pair_count": len(tier_006),
            "selected_tier": tier_name if tier else None,
            "search_option_count": len(tier),
        })
    if sorted(pools) != sorted(grouped):
        return [], host_audit
    selected = [pools[host][0] for host in sorted(pools)]
    native_quantiles = np.quantile(native, (0.10, 0.25, 0.50, 0.75, 0.90), axis=0)
    for _sweep in range(4):
        changed = 0
        for index, host in enumerate(sorted(pools)):
            incumbent = selected[index]
            best = incumbent
            best_key = (
                selection_proxy(selected, native_quantiles, native_summary),
                incumbent["near_candidate_id"], incumbent["far_candidate_id"],
            )
            for option in pools[host]:
                trial = list(selected)
                trial[index] = option
                key = (
                    selection_proxy(trial, native_quantiles, native_summary),
                    option["near_candidate_id"], option["far_candidate_id"],
                )
                if key < best_key:
                    best, best_key = option, key
            if best is not incumbent:
                selected[index] = best
                changed += 1
        if changed == 0:
            break
    serial = []
    for row in selected:
        serial.append({
            key: value for key, value in row.items() if key not in ("near", "far")
        })
    return serial, host_audit


def occupancy_matching_cost(pair: dict, native_summary: dict) -> float:
    value = float(pair["absolute_occupancy_difference"]) / 0.05
    for key, difference in pair["standardized_metric_differences"].items():
        value += float(difference) / 0.15
    return value


def selection_proxy(selected: list[dict], native_quantiles: np.ndarray, native_summary: dict) -> float:
    near = np.asarray([[float(row["near"]["calibrated_metrics"][key]) for key in METRICS] for row in selected])
    far = np.asarray([[float(row["far"]["calibrated_metrics"][key]) for key in METRICS] for row in selected])
    gaps = np.asarray([float(row["support_gap"]) for row in selected])
    distribution = 0.0
    paired = 0.0
    hard = 0.0
    for group in (near, far):
        quantiles = np.quantile(group, (0.10, 0.25, 0.50, 0.75, 0.90), axis=0)
        for index, key in enumerate(METRICS):
            sd = max(float(native_summary[key]["sample_sd"]), 1e-12)
            distribution += float(np.mean(np.abs(quantiles[:, index] - native_quantiles[:, index])) / sd)
    for index, key in enumerate(METRICS):
        sd = max(float(native_summary[key]["sample_sd"]), 1e-12)
        value = float(np.median(np.abs(near[:, index] - far[:, index])) / sd)
        paired += value
        hard += max(0.0, value / 0.10 - 1.0) ** 2
    hard += max(0.0, 0.06 / max(float(np.median(gaps)), 1e-12) - 1.0) ** 2
    hard += max(0.0, 0.80 / max(float(np.mean(gaps >= 0.04)), 1e-12) - 1.0) ** 2
    # Fixed v47-development choice: radiometry weight 1, gap weight 4.
    return 1000.0 * hard + distribution + paired - 4.0 * float(np.median(gaps))


def summarize_rows(rows: list[dict], group_key: str, metric_keys: Iterable[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    output = []
    for group, values in sorted(grouped.items()):
        for metric in metric_keys:
            column = np.asarray([float(row[metric]) for row in values], dtype=np.float64)
            output.append({
                group_key: group,
                "metric": metric,
                "n": len(column),
                "mean": float(np.mean(column)),
                "sample_sd": float(np.std(column, ddof=1)) if len(column) > 1 else math.nan,
                "q1": float(np.quantile(column, 0.25)),
                "median": float(np.median(column)),
                "q3": float(np.quantile(column, 0.75)),
            })
    return output
