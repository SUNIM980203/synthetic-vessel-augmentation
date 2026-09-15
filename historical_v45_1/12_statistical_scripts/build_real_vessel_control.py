#!/usr/bin/env python3
"""Build an xView-training-only real-vessel cutout control matched to a Unity reference dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from build_hybrid_dataset import extract_cutout, link_or_copy, paste_cutout


VESSEL_CATEGORY = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-coco", type=Path, required=True)
    parser.add_argument("--real-images", type=Path, required=True)
    parser.add_argument("--reference-coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-limit", type=int, default=300)
    parser.add_argument("--source-min-bbox-scale", type=float, default=24.0)
    parser.add_argument("--source-max-bbox-scale", type=float, default=80.0)
    parser.add_argument("--aspect-shortlist", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_library(
    coco: dict[str, object],
    images_directory: Path,
    output: Path,
    limit: int,
    min_scale: float,
    max_scale: float,
    rng: random.Random,
) -> list[dict[str, object]]:
    images = {int(record["id"]): record for record in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    image_ids = list(images)
    rng.shuffle(image_ids)
    library: list[dict[str, object]] = []
    cutout_directory = output / "cutouts" / str(VESSEL_CATEGORY)
    cutout_directory.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        if len(library) >= limit:
            break
        record = images[image_id]
        vessel_annotations = [
            annotation
            for annotation in annotations_by_image[image_id]
            if int(annotation["category_id"]) == VESSEL_CATEGORY
        ]
        rng.shuffle(vessel_annotations)
        if not vessel_annotations:
            continue
        rgb = np.asarray(Image.open(images_directory / record["file_name"]).convert("RGB"))
        for annotation in vessel_annotations:
            width, height = [float(value) for value in annotation["bbox"][2:4]]
            bbox_scale = math.sqrt(width * height)
            if bbox_scale < min_scale or bbox_scale >= max_scale:
                continue
            cutout = extract_cutout(rgb, annotation, annotations_by_image[image_id])
            if cutout is None:
                continue
            alpha = np.asarray(cutout.getchannel("A"), dtype=np.uint8)
            alpha_occupancy = float(np.mean(alpha > 64))
            path = cutout_directory / f"real_vessel_{len(library):04d}.png"
            cutout.save(path)
            library.append(
                {
                    "image": cutout,
                    "path": str(path),
                    "source_scene": str(record.get("source_scene", record["file_name"])),
                    "source_image_id": image_id,
                    "source_annotation_id": int(annotation["id"]),
                    "source_bbox_scale": bbox_scale,
                    "aspect": cutout.width / max(cutout.height, 1),
                    "alpha_occupancy": alpha_occupancy,
                    "uses": 0,
                }
            )
            if len(library) >= limit:
                break
    return library


def choose_cutout(
    library: list[dict[str, object]],
    target_scene: str,
    target_aspect: float,
    shortlist_size: int,
    rng: random.Random,
) -> dict[str, object]:
    eligible = [record for record in library if str(record["source_scene"]) != target_scene]
    if not eligible:
        raise RuntimeError(f"No cross-scene vessel cutout is available for scene {target_scene}")
    eligible.sort(key=lambda record: abs(math.log(float(record["aspect"]) / target_aspect)))
    shortlist = eligible[: max(1, min(shortlist_size, len(eligible)))]
    minimum_uses = min(int(record["uses"]) for record in shortlist)
    balanced = [record for record in shortlist if int(record["uses"]) == minimum_uses]
    selected = rng.choice(balanced)
    selected["uses"] = int(selected["uses"]) + 1
    return selected


def main() -> int:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    if args.source_min_bbox_scale <= 0 or args.source_max_bbox_scale <= args.source_min_bbox_scale:
        raise ValueError("Invalid source bbox-scale interval")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    appearance_rng = random.Random(args.seed ^ 0x5EEDBEEF)

    real = load_json(args.real_coco)
    reference = load_json(args.reference_coco)
    library = build_library(
        real,
        args.real_images,
        args.output,
        args.library_limit,
        args.source_min_bbox_scale,
        args.source_max_bbox_scale,
        rng,
    )
    if not library:
        raise RuntimeError("No valid real-vessel cutouts were extracted")

    real_images_by_name = {str(record["file_name"]): record for record in real["images"]}
    real_annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in real["annotations"]:
        real_annotations_by_image[int(annotation["image_id"])].append(annotation)
    reference_annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in reference["annotations"]:
        reference_annotations_by_image[int(annotation["image_id"])].append(annotation)

    output_images_directory = args.output / "images" / "train"
    output_images_directory.mkdir(parents=True, exist_ok=True)
    output_images: list[dict[str, object]] = []
    output_annotations: list[dict[str, object]] = []
    next_annotation_id = 1
    inserted_scales: list[float] = []
    reference_scales: list[float] = []
    aspect_log_distortions: list[float] = []
    source_scenes: Counter[str] = Counter()
    target_scenes: Counter[str] = Counter()
    modified_images = 0

    for output_image_id, reference_image in enumerate(reference["images"], start=1):
        source_name = str(reference_image.get("source_patch", ""))
        if source_name not in real_images_by_name:
            raise KeyError(f"Reference source patch is missing from real training COCO: {source_name}")
        real_image = real_images_by_name[source_name]
        target_scene = str(real_image.get("source_scene", source_name))
        source_path = args.real_images / source_name
        background = Image.open(source_path).convert("RGBA")
        inserted = [
            annotation
            for annotation in reference_annotations_by_image[int(reference_image["id"])]
            if int(annotation["category_id"]) == VESSEL_CATEGORY
            and str(annotation.get("source", "")) == "unity_cutout"
        ]
        if len(inserted) > 1:
            raise ValueError(f"Reference image {reference_image['id']} has more than one Unity vessel")

        if inserted:
            target = inserted[0]
            x, y, width, height = [float(value) for value in target["bbox"]]
            target_width = int(round(width))
            target_height = int(round(height))
            if target_width < 3 or target_height < 3:
                raise ValueError(f"Invalid reference vessel bbox: {target['bbox']}")
            target_aspect = target_width / target_height
            selected = choose_cutout(
                library,
                target_scene,
                target_aspect,
                args.aspect_shortlist,
                rng,
            )
            source_cutout = selected["image"]
            assert isinstance(source_cutout, Image.Image)
            resized = source_cutout.resize((target_width, target_height), Image.Resampling.LANCZOS)
            position = (int(round(x)), int(round(y)))
            paste_cutout(
                background,
                resized,
                position,
                VESSEL_CATEGORY,
                appearance_rng,
                sensor_aware=True,
                vessel_render_v2=True,
            )
            bbox = [float(position[0]), float(position[1]), float(target_width), float(target_height)]
            inserted_scale = math.sqrt(target_width * target_height)
            reference_scale = math.sqrt(width * height)
            inserted_scales.append(inserted_scale)
            reference_scales.append(reference_scale)
            aspect_log_distortions.append(abs(math.log(float(selected["aspect"]) / target_aspect)))
            source_scenes[str(selected["source_scene"])] += 1
            target_scenes[target_scene] += 1
            if str(selected["source_scene"]) == target_scene:
                raise AssertionError("Same-scene cutout placement escaped the exclusion rule")
            output_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": output_image_id,
                    "category_id": VESSEL_CATEGORY,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "source": "xview_real_cutout",
                    "cutout_source_scene": str(selected["source_scene"]),
                    "cutout_source_image_id": int(selected["source_image_id"]),
                    "cutout_source_annotation_id": int(selected["source_annotation_id"]),
                    "reference_unity_annotation_id": int(target["id"]),
                }
            )
            next_annotation_id += 1
            modified_images += 1

        real_image_annotations = real_annotations_by_image[int(real_image["id"])]
        for annotation in real_image_annotations:
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

        file_name = f"realv_{output_image_id:06d}.jpg"
        destination = output_images_directory / file_name
        if inserted:
            background.convert("RGB").save(destination, quality=95, subsampling=0)
        else:
            link_or_copy(source_path, destination)
        output_record = dict(real_image)
        output_record.update(
            {
                "id": output_image_id,
                "file_name": file_name,
                "source_patch": source_name,
            }
        )
        output_images.append(output_record)

    reference_insertions = sum(
        1
        for annotation in reference["annotations"]
        if int(annotation["category_id"]) == VESSEL_CATEGORY
        and str(annotation.get("source", "")) == "unity_cutout"
    )
    if modified_images != reference_insertions:
        raise AssertionError((modified_images, reference_insertions))

    coco = {
        "info": {
            "description": "xView real-vessel cutout control matched to Unity Medium-150 hosts, positions, and boxes",
            "version": "1.0.0",
            "seed": args.seed,
            "source_split": "train only",
            "same_scene_cutouts_excluded": True,
            "reference_coco": str(args.reference_coco),
            "sensor_aware": True,
            "vessel_render_v2": True,
        },
        "images": output_images,
        "annotations": output_annotations,
        "categories": real["categories"],
    }
    annotations_directory = args.output / "annotations"
    annotations_directory.mkdir(parents=True, exist_ok=True)
    (annotations_directory / "instances_train.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    use_counts = [int(record["uses"]) for record in library]
    source_bbox_scales = [float(record["source_bbox_scale"]) for record in library]
    alpha_occupancies = [float(record["alpha_occupancy"]) for record in library]
    scale_errors = np.abs(np.asarray(inserted_scales) - np.asarray(reference_scales))
    summary = {
        "image_count": len(output_images),
        "annotation_count": len(output_annotations),
        "real_annotation_count": len(real["annotations"]),
        "real_cutout_annotation_count": modified_images,
        "modified_image_count": modified_images,
        "reference_insertion_count": reference_insertions,
        "library_count": len(library),
        "library_source_scene_count": len({str(record["source_scene"]) for record in library}),
        "target_scene_count": len(target_scenes),
        "same_scene_insertion_count": 0,
        "unique_used_cutout_count": sum(count > 0 for count in use_counts),
        "maximum_cutout_reuse": max(use_counts, default=0),
        "source_bbox_scale_min": min(source_bbox_scales, default=None),
        "source_bbox_scale_p50": float(np.median(source_bbox_scales)) if source_bbox_scales else None,
        "source_bbox_scale_max": max(source_bbox_scales, default=None),
        "library_alpha_occupancy_p05": float(np.percentile(alpha_occupancies, 5)) if alpha_occupancies else None,
        "library_alpha_occupancy_p50": float(np.median(alpha_occupancies)) if alpha_occupancies else None,
        "inserted_bbox_scale_min": min(inserted_scales, default=None),
        "inserted_bbox_scale_p50": float(np.median(inserted_scales)) if inserted_scales else None,
        "inserted_bbox_scale_p95": float(np.percentile(inserted_scales, 95)) if inserted_scales else None,
        "inserted_bbox_scale_max": max(inserted_scales, default=None),
        "reference_scale_max_absolute_error": float(np.max(scale_errors)) if len(scale_errors) else None,
        "aspect_log_distortion_p50": float(np.median(aspect_log_distortions)) if aspect_log_distortions else None,
        "aspect_log_distortion_p95": float(np.percentile(aspect_log_distortions, 95)) if aspect_log_distortions else None,
        "different_scene_constraint_met": True,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
