"""Shared calibrated appearance-pool generation for support-diversity study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from support_scale_radiometry_calibration_common import (
    METRICS,
    apply_frozen_calibration,
    difference_soft_mask,
    extended_appearance_metrics,
    load_json,
    read_rgb,
    save_final_jpeg,
    sha256,
    write_json,
)


APPEARANCE_VARIANTS: tuple[dict[str, Any], ...] = (
    {"rank": 0, "name": "original", "family": "baseline", "amplitude": 0.0},
    {"rank": 1, "name": "major_axis_reflection", "family": "texture_realization", "amplitude": 0.45},
    {"rank": 2, "name": "longitudinal_phase_0", "family": "spatial_frequency", "amplitude": 6.0},
    {"rank": 3, "name": "longitudinal_phase_pi2", "family": "spatial_frequency", "amplitude": 6.0},
    {"rank": 4, "name": "strip_permutation", "family": "texture_realization", "amplitude": 0.55},
    {"rank": 5, "name": "phase_scramble", "family": "texture_realization", "amplitude": 0.50},
    {"rank": 6, "name": "chroma_microstructure", "family": "material_chroma", "amplitude": 3.0},
    {"rank": 7, "name": "microcontrast_detail", "family": "fine_detail", "amplitude": 0.65},
)

# Prespecified development-only 2,048 extension.  It may be invoked only after
# the uniform 1,024 sweep has been evaluated and failed.  Every host receives
# all eight additional variants; there is no failed-host-specific branch.
APPEARANCE_EXTENSION_VARIANTS: tuple[dict[str, Any], ...] = (
    {"rank": 8, "name": "major_axis_reflection_strong", "family": "texture_realization", "amplitude": 0.75},
    {"rank": 9, "name": "longitudinal_phase_pi", "family": "spatial_frequency", "amplitude": 8.0},
    {"rank": 10, "name": "longitudinal_phase_3pi2", "family": "spatial_frequency", "amplitude": 8.0},
    {"rank": 11, "name": "diagonal_crosshatch", "family": "spatial_frequency", "amplitude": 5.0},
    {"rank": 12, "name": "strip_permutation_alt", "family": "texture_realization", "amplitude": 0.80},
    {"rank": 13, "name": "phase_scramble_alt", "family": "texture_realization", "amplitude": 0.82},
    {"rank": 14, "name": "chroma_microstructure_alt", "family": "material_chroma", "amplitude": 4.5},
    {"rank": 15, "name": "microcontrast_detail_soft", "family": "fine_detail", "amplitude": 0.40},
)


def _masked_match(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match source mean/SD to target within mask, retaining spatial rearrangement."""
    if not np.any(mask):
        return target.copy()
    output = source.astype(np.float32).copy()
    for channel in range(source.shape[2]):
        src = source[:, :, channel][mask].astype(np.float32)
        ref = target[:, :, channel][mask].astype(np.float32)
        src_sd = max(float(np.std(src)), 1e-6)
        ref_sd = float(np.std(ref))
        output[:, :, channel] = (
            (source[:, :, channel].astype(np.float32) - float(np.mean(src)))
            * (ref_sd / src_sd)
            + float(np.mean(ref))
        )
    return output


def apply_appearance_variant(
    raw_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    bbox: list[float],
    variant: dict[str, Any],
    seed: int,
    calibration_parameters: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Change only vessel-interior appearance; preserve outline, bbox and placement."""
    if int(variant["rank"]) == 0:
        return raw_rgb.copy(), {"changed_pixel_fraction": 0.0, "interior_pixel_count": 0}
    soft = difference_soft_mask(raw_rgb, clean_rgb, bbox, calibration_parameters)
    x, y, width, height = [int(round(float(value))) for value in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(raw_rgb.shape[1], x + width), min(raw_rgb.shape[0], y + height)
    roi_soft = soft[y0:y1, x0:x1]
    base_mask = (roi_soft >= 0.30).astype(np.uint8)
    interior = cv2.erode(base_mask, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    if int(np.sum(interior)) < 12:
        interior = base_mask.astype(bool)
    roi = raw_rgb[y0:y1, x0:x1]
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    styled = lab.copy()
    name = str(variant["name"])
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(interior.shape, dtype=np.float32)
    major = xx / max(interior.shape[1] - 1, 1) if width >= height else yy / max(interior.shape[0] - 1, 1)
    minor = yy / max(interior.shape[0] - 1, 1) if width >= height else xx / max(interior.shape[1] - 1, 1)

    if name.startswith("major_axis_reflection"):
        reflected = np.flip(lab, axis=1 if width >= height else 0)
        reflected_mask = np.flip(interior, axis=1 if width >= height else 0)
        valid = interior & reflected_mask
        matched = _masked_match(reflected, lab, valid)
        blend = float(variant["amplitude"])
        styled[valid] = (1.0 - blend) * lab[valid] + blend * matched[valid]
    elif name.startswith("longitudinal_phase"):
        phase = {
            "longitudinal_phase_0": 0.0,
            "longitudinal_phase_pi2": math.pi / 2.0,
            "longitudinal_phase_pi": math.pi,
            "longitudinal_phase_3pi2": 3.0 * math.pi / 2.0,
        }[name]
        pattern = np.sin(major * math.pi * 8.0 + phase) * np.cos((minor - 0.5) * math.pi)
        pattern -= float(np.mean(pattern[interior]))
        styled[:, :, 0][interior] += float(variant["amplitude"]) * pattern[interior]
    elif name.startswith("strip_permutation"):
        residual = lab - cv2.GaussianBlur(lab, (0, 0), 1.0)
        axis_length = interior.shape[1] if width >= height else interior.shape[0]
        boundaries = np.linspace(0, axis_length, 7, dtype=int)
        order = rng.permutation(6)
        pieces = []
        for index in range(6):
            if width >= height:
                pieces.append(residual[:, boundaries[index] : boundaries[index + 1]])
            else:
                pieces.append(residual[boundaries[index] : boundaries[index + 1], :])
        rebuilt = np.concatenate([pieces[index] for index in order], axis=1 if width >= height else 0)
        rebuilt = cv2.resize(rebuilt, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
        blend = float(variant["amplitude"])
        styled[interior] = lab[interior] + blend * rebuilt[interior]
    elif name.startswith("phase_scramble"):
        luminance = lab[:, :, 0]
        smooth = cv2.GaussianBlur(luminance, (0, 0), 1.2)
        residual = luminance - smooth
        spectrum = np.fft.rfft2(residual)
        phases = rng.uniform(-math.pi, math.pi, spectrum.shape)
        scrambled = np.fft.irfft2(np.abs(spectrum) * np.exp(1j * phases), s=residual.shape).real.astype(np.float32)
        src_sd = max(float(np.std(scrambled[interior])), 1e-6)
        ref_sd = float(np.std(residual[interior]))
        scrambled = (scrambled - float(np.mean(scrambled[interior]))) * (ref_sd / src_sd)
        styled[:, :, 0][interior] = smooth[interior] + (
            (1.0 - float(variant["amplitude"])) * residual[interior]
            + float(variant["amplitude"]) * scrambled[interior]
        )
    elif name.startswith("chroma_microstructure"):
        pattern = np.sin(major * math.pi * 6.0 + 0.7) * np.cos(minor * math.pi * 4.0 - 0.4)
        pattern -= float(np.mean(pattern[interior]))
        amplitude = float(variant["amplitude"])
        styled[:, :, 1][interior] += amplitude * pattern[interior]
        styled[:, :, 2][interior] -= 0.8 * amplitude * pattern[interior]
    elif name.startswith("microcontrast_detail"):
        highpass = lab[:, :, 0] - cv2.GaussianBlur(lab[:, :, 0], (0, 0), 0.75)
        styled[:, :, 0][interior] += float(variant["amplitude"]) * highpass[interior]
    elif name == "diagonal_crosshatch":
        pattern = np.sin((major + minor) * math.pi * 7.0) * np.sin((major - minor) * math.pi * 5.0)
        pattern -= float(np.mean(pattern[interior]))
        styled[:, :, 0][interior] += float(variant["amplitude"]) * pattern[interior]
    else:
        raise ValueError(f"Unknown appearance variant: {name}")

    styled = np.clip(styled, 0, 255).astype(np.uint8)
    styled_rgb = cv2.cvtColor(styled, cv2.COLOR_LAB2RGB)
    alpha = np.clip(roi_soft, 0.0, 1.0)[:, :, None]
    output = raw_rgb.copy()
    blended = np.rint(alpha * styled_rgb.astype(np.float32) + (1.0 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    output[y0:y1, x0:x1] = blended
    changed = np.any(output != raw_rgb, axis=2)
    return output, {
        "changed_pixel_fraction": float(np.mean(changed)),
        "interior_pixel_count": int(np.sum(interior)),
        "valid_color_range": bool(output.min() >= 0 and output.max() <= 255),
        "geometry_pixels_outside_bbox_changed": bool(np.any(changed[:y0]) or np.any(changed[y1:]) or np.any(changed[:, :x0]) or np.any(changed[:, x1:])),
    }


def generate_pool(
    base_raw_root: Path,
    input_root: Path,
    output_root: Path,
    protocol: dict[str, Any],
    original_host_offset: int,
    source_label: str,
    variants: tuple[dict[str, Any], ...] = APPEARANCE_VARIANTS,
    candidate_id_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[int, Path], dict[str, Any]]:
    """Generate the 8x nested appearance pool and apply the frozen calibrator."""
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite candidate pool: {output_root}")
    images_root = output_root / "images" / "train"
    annotations_root = output_root / "annotations"
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)
    base_candidates = load_json(base_raw_root / "candidate_manifest.json")["candidates"]
    base_coco = load_json(base_raw_root / "annotations" / "instances_train.json")
    annotations_by_id = {int(row["id"]): row for row in base_coco["annotations"]}
    source_scene_by_id = {int(row["id"]): row["source_scene"] for row in base_coco["images"]}
    clean_coco = load_json(input_root / "annotations" / "base_instances_train.json")
    clean_paths = {
        int(row["id"]): input_root / "images" / "train" / str(row["file_name"])
        for row in clean_coco["images"]
    }
    clean_images = {key: read_rgb(path) for key, path in clean_paths.items()}
    params = protocol["calibration_function"]["parameters"]
    output_candidates: list[dict[str, Any]] = []
    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    image_paths: dict[int, Path] = {}
    original_hash_matches = 0
    previous_final = base_raw_root.parent / "support_scale_radiometry_fresh_medium_calibrated"
    if not previous_final.exists():
        previous_final = Path("__missing__")

    total = len(base_candidates) * len(variants)
    for base_index, base in enumerate(sorted(base_candidates, key=lambda row: int(row["candidate_id"])), start=1):
        base_id = int(base["candidate_id"])
        host = int(base["host_index"])
        raw_path = base_raw_root / "images" / "train" / f"cand_{base_id:06d}.jpg"
        raw = read_rgb(raw_path)
        clean = clean_images[int(base["reference_image_id"])]
        for local_variant_index, variant in enumerate(variants):
            rank = int(variant["rank"])
            candidate_id = candidate_id_offset + (base_id - 1) * len(variants) + local_variant_index + 1
            appearance_seed = int.from_bytes(
                hashlib.sha256(f"support-diversity-v1|{source_label}|{base_id}|{rank}".encode()).digest()[:8],
                "big",
            ) & 0x7FFFFFFF
            transformed, appearance_audit = apply_appearance_variant(
                raw, clean, base["bbox"], variant, appearance_seed, params
            )
            precalibration = extended_appearance_metrics(transformed, base["bbox"])
            final, calibration_audit = apply_frozen_calibration(transformed, clean, base["bbox"], params)
            output_path = images_root / f"cand_{candidate_id:07d}.jpg"
            save_final_jpeg(output_path, final)
            decoded = read_rgb(output_path)
            metrics = extended_appearance_metrics(decoded, base["bbox"])
            row = copy.deepcopy(base)
            row.update(
                {
                    "candidate_id": candidate_id,
                    "base_candidate_id": base_id,
                    "appearance_variant_rank": rank,
                    "appearance_variant": str(variant["name"]),
                    "appearance_family": str(variant["family"]),
                    "appearance_amplitude": float(variant["amplitude"]),
                    "appearance_seed": appearance_seed,
                    "original_v47_host_id": original_host_offset + host,
                    "precalibration_metrics": {key: float(precalibration[key]) for key in METRICS},
                    "calibrated_metrics": {key: float(metrics[key]) for key in METRICS},
                    "appearance_audit": appearance_audit,
                    "prospective_calibration": copy.deepcopy(params),
                    "prospective_calibration_audit": calibration_audit,
                    "base_raw_image_sha256": sha256(raw_path),
                    "image_sha256": sha256(output_path),
                }
            )
            output_candidates.append(row)
            image_paths[candidate_id] = output_path
            output_images.append(
                {
                    "id": candidate_id,
                    "file_name": output_path.name,
                    "width": int(decoded.shape[1]),
                    "height": int(decoded.shape[0]),
                    "source_scene": source_scene_by_id[base_id],
                    "host_index": host,
                    "original_v47_host_id": original_host_offset + host,
                }
            )
            ann = copy.deepcopy(annotations_by_id[base_id])
            ann.update({"id": candidate_id, "image_id": candidate_id})
            output_annotations.append(ann)
            if rank == 0:
                prior_path = previous_final / "images" / "train" / f"cand_{base_id:06d}.jpg"
                if prior_path.exists() and sha256(prior_path) == sha256(output_path):
                    original_hash_matches += 1
        if base_index % 128 == 0 or base_index == len(base_candidates):
            print(f"expanded calibrated pool {base_index * len(variants)}/{total}", flush=True)

    output_coco = {
        "info": {
            "description": f"Calibrated nested support-diversity pool: {source_label}",
            "frozen_calibration_profile": "contrast_1.25_unsharp_0.00",
            "candidate_images_are_final_decoded_jpegs": True,
        },
        "images": output_images,
        "annotations": output_annotations,
        "categories": base_coco["categories"],
    }
    write_json(annotations_root / "instances_train.json", output_coco)
    write_json(output_root / "candidate_manifest.json", {"candidates": output_candidates})
    summary = {
        "source_label": source_label,
        "base_candidate_count": len(base_candidates),
        "expanded_candidate_count": len(output_candidates),
        "hosts": len({int(row["host_index"]) for row in output_candidates}),
        "generated_candidates_per_host": 128 * len(variants),
        "appearance_variants": list(variants),
        "original_rank_hash_matches_previous_final": original_hash_matches,
        "support_calculated_after_final_calibration": True,
        "calibration_parameters": copy.deepcopy(params),
        "candidate_manifest_sha256": sha256(output_root / "candidate_manifest.json"),
        "candidate_coco_sha256": sha256(annotations_root / "instances_train.json"),
    }
    write_json(output_root / "summary.json", summary)
    return output_candidates, image_paths, summary


def budget_candidates(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    ranks = budget // 128
    return [row for row in candidates if int(row["appearance_variant_rank"]) < ranks]
