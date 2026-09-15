"""Identity-aware calibrated candidate generation and deterministic deduplication.

Only identity, RNG, oversupply, and integrity mechanics live here.  Rendering,
appearance transforms, calibration, geometry, and metrics are imported from
their frozen scientific implementations.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from candidate_uid import (
    candidate_uid,
    derive_unique_rng_seed64,
    identity_document,
)
from support_diversity_candidate_common import (
    APPEARANCE_EXTENSION_VARIANTS,
    APPEARANCE_VARIANTS,
    apply_appearance_variant,
)
from support_scale_radiometry_calibration_common import (
    METRICS,
    apply_frozen_calibration,
    extended_appearance_metrics,
    load_json,
    read_rgb,
    save_final_jpeg,
    sha256,
    write_csv,
    write_json,
)


SCIENTIFIC_VARIANTS: tuple[dict[str, Any], ...] = (
    APPEARANCE_VARIANTS + APPEARANCE_EXTENSION_VARIANTS
)
TARGET_UNIQUE_PER_HOST = 2048
DEVELOPMENT_BUDGETS = (2048, 2080, 2112, 2176)

# Only existing stochastic variants are reused for the appended sequence.
# No transformation, family, amplitude, or parameter range is introduced.
EXTRA_SEQUENCE_VARIANT_RANKS = (4, 5, 12, 13)


def _variant_by_rank() -> dict[int, dict[str, Any]]:
    return {int(row["rank"]): copy.deepcopy(row) for row in SCIENTIFIC_VARIANTS}


def _host_candidate_sequence(
    base_rows: list[dict[str, Any]],
    oversupply_count: int,
) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    """Return (base, frozen variant, repeat) in canonical host-local order."""
    if len(base_rows) != 128:
        raise ValueError(f"Expected 128 base renders per host, got {len(base_rows)}")
    if oversupply_count not in DEVELOPMENT_BUDGETS:
        raise ValueError(f"Unsupported prospectively bounded budget: {oversupply_count}")
    variants = _variant_by_rank()
    sequence: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for base in sorted(base_rows, key=lambda row: int(row["candidate_id"])):
        for rank in range(16):
            sequence.append((base, variants[rank], 0))
    for extra_index in range(oversupply_count - TARGET_UNIQUE_PER_HOST):
        base = sorted(base_rows, key=lambda row: int(row["candidate_id"]))[extra_index]
        rank = EXTRA_SEQUENCE_VARIANT_RANKS[extra_index % len(EXTRA_SEQUENCE_VARIANT_RANKS)]
        sequence.append((base, variants[rank], 1))
    return sequence


def generate_oversupply_pool(
    *,
    base_raw_root: Path,
    input_root: Path,
    output_root: Path,
    radiometry_protocol: dict[str, Any],
    original_host_offset: int,
    cohort_id: str,
    oversupply_count: int,
    reserved_rng_seeds: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """Generate one fixed oversupply budget for every host.

    Final-image hashes are computed from decoded calibrated JPEG files.  The
    function refuses to overwrite any existing output.
    """
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite candidate pool: {output_root}")
    images_root = output_root / "images" / "train"
    annotations_root = output_root / "annotations"
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    base_candidates = load_json(base_raw_root / "candidate_manifest.json")["candidates"]
    base_coco = load_json(base_raw_root / "annotations" / "instances_train.json")
    annotations_by_id = {int(row["id"]): row for row in base_coco["annotations"]}
    source_scene_by_id = {int(row["id"]): str(row["source_scene"]) for row in base_coco["images"]}
    clean_coco = load_json(input_root / "annotations" / "base_instances_train.json")
    clean_paths = {
        int(row["id"]): input_root / "images" / "train" / str(row["file_name"])
        for row in clean_coco["images"]
    }
    clean_images = {key: read_rgb(path) for key, path in clean_paths.items()}
    params = radiometry_protocol["calibration_function"]["parameters"]
    experiment_namespace = "support-scale-deconfounding-medium"
    reserved = {int(value) for value in reserved_rng_seeds}
    used_rng_seeds: set[int] = set()
    seen_uids: set[str] = set()

    by_host: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in base_candidates:
        by_host[int(row["host_index"])].append(row)
    if len(by_host) != 30:
        raise ValueError(f"Expected 30 hosts, got {len(by_host)}")

    output_candidates: list[dict[str, Any]] = []
    raw_images: list[dict[str, Any]] = []
    raw_annotations: list[dict[str, Any]] = []
    uid_csv: list[dict[str, Any]] = []
    total = len(by_host) * oversupply_count

    for host_position, host in enumerate(sorted(by_host)):
        original_host_id = original_host_offset + host
        sequence = _host_candidate_sequence(by_host[host], oversupply_count)
        raw_cache: dict[int, Any] = {}
        for host_generation_index, (base, variant, repeat) in enumerate(sequence):
            base_id = int(base["candidate_id"])
            if base_id not in raw_cache:
                raw_cache[base_id] = read_rgb(
                    base_raw_root / "images" / "train" / f"cand_{base_id:06d}.jpg"
                )
            raw = raw_cache[base_id]
            clean = clean_images[int(base["reference_image_id"])]
            candidate_id = host_position * oversupply_count + host_generation_index + 1
            identity = identity_document(
                experiment_namespace=experiment_namespace,
                cohort_id=cohort_id,
                original_host_id=original_host_id,
                base_render_id=base_id,
                appearance_variant_id=str(variant["name"]),
                appearance_variant_parameters={
                    "rank": int(variant["rank"]),
                    "family": str(variant["family"]),
                    "amplitude": float(variant["amplitude"]),
                    "sequence_repeat": int(repeat),
                },
                candidate_index=host_generation_index,
            )
            uid, uid_preimage = candidate_uid(identity)
            if uid in seen_uids:
                raise RuntimeError(f"Candidate UID collision before rendering: {uid}")
            seen_uids.add(uid)
            seed = derive_unique_rng_seed64(uid, used_rng_seeds, reserved)
            transformed, appearance_audit = apply_appearance_variant(
                raw,
                clean,
                base["bbox"],
                variant,
                seed.rng_seed_64,
                params,
            )
            precalibration = extended_appearance_metrics(transformed, base["bbox"])
            final, calibration_audit = apply_frozen_calibration(
                transformed, clean, base["bbox"], params
            )
            output_path = images_root / f"cand_{candidate_id:07d}.jpg"
            save_final_jpeg(output_path, final)
            decoded = read_rgb(output_path)
            metrics = extended_appearance_metrics(decoded, base["bbox"])
            image_hash = sha256(output_path)
            valid = bool(
                not appearance_audit.get("geometry_pixels_outside_bbox_changed", False)
                and appearance_audit.get("valid_color_range", True)
                and image_hash
            )
            row = copy.deepcopy(base)
            row.update(
                {
                    "candidate_id": candidate_id,
                    "base_candidate_id": base_id,
                    "host_generation_index": host_generation_index,
                    "canonical_generation_index": candidate_id - 1,
                    "appearance_variant_rank": int(variant["rank"]),
                    "appearance_variant": str(variant["name"]),
                    "appearance_family": str(variant["family"]),
                    "appearance_amplitude": float(variant["amplitude"]),
                    "appearance_sequence_repeat": int(repeat),
                    "candidate_uid": uid,
                    "candidate_uid_preimage": uid_preimage,
                    "rng_seed_64": int(seed.rng_seed_64),
                    "rng_collision_nonce": int(seed.collision_nonce),
                    "original_v47_host_id": original_host_id,
                    "source_scene": source_scene_by_id[base_id],
                    "precalibration_metrics": {
                        key: float(precalibration[key]) for key in METRICS
                    },
                    "calibrated_metrics": {key: float(metrics[key]) for key in METRICS},
                    "appearance_audit": appearance_audit,
                    "prospective_calibration": copy.deepcopy(params),
                    "prospective_calibration_audit": calibration_audit,
                    "base_raw_image_sha256": sha256(
                        base_raw_root / "images" / "train" / f"cand_{base_id:06d}.jpg"
                    ),
                    "final_image_sha256": image_hash,
                    "image_sha256": image_hash,
                    "valid_candidate": valid,
                }
            )
            output_candidates.append(row)
            raw_images.append(
                {
                    "id": candidate_id,
                    "file_name": output_path.name,
                    "width": int(decoded.shape[1]),
                    "height": int(decoded.shape[0]),
                    "source_scene": source_scene_by_id[base_id],
                    "host_index": host,
                    "original_v47_host_id": original_host_id,
                    "candidate_uid": uid,
                }
            )
            ann = copy.deepcopy(annotations_by_id[base_id])
            ann.update({"id": candidate_id, "image_id": candidate_id})
            raw_annotations.append(ann)
            uid_csv.append(
                {
                    "candidate_id": candidate_id,
                    "host_index": host,
                    "original_v47_host_id": original_host_id,
                    "host_generation_index": host_generation_index,
                    "base_render_id": base_id,
                    "appearance_variant": str(variant["name"]),
                    "appearance_family": str(variant["family"]),
                    "sequence_repeat": int(repeat),
                    "candidate_uid": uid,
                    "rng_seed_64": int(seed.rng_seed_64),
                    "rng_collision_nonce": int(seed.collision_nonce),
                    "final_image_sha256": image_hash,
                    "valid_candidate": valid,
                }
            )
        print(
            f"identity-calibrated pool host {host_position + 1}/30 "
            f"({(host_position + 1) * oversupply_count}/{total})",
            flush=True,
        )

    write_json(output_root / "candidate_manifest_raw.json", {"candidates": output_candidates})
    write_csv(output_root / "candidate_uid_manifest.csv", uid_csv)
    write_json(
        annotations_root / "instances_raw.json",
        {
            "info": {
                "description": f"Identity-aware oversupply pool: {cohort_id}",
                "frozen_calibration_profile": "contrast_1.25_unsharp_0.00",
                "candidate_images_are_final_decoded_jpegs": True,
            },
            "images": raw_images,
            "annotations": raw_annotations,
            "categories": base_coco["categories"],
        },
    )
    write_json(
        output_root / "generation_summary.json",
        {
            "cohort_id": cohort_id,
            "raw_candidate_count": len(output_candidates),
            "host_count": len(by_host),
            "raw_candidates_per_host": oversupply_count,
            "scientific_target_unique_per_host": TARGET_UNIQUE_PER_HOST,
            "scientific_variants": list(SCIENTIFIC_VARIANTS),
            "extra_sequence_variant_ranks": list(EXTRA_SEQUENCE_VARIANT_RANKS),
            "uid_unique": len(seen_uids) == len(output_candidates),
            "rng_seed_64_unique": len(used_rng_seeds) == len(output_candidates),
            "rng_collision_nonce_count": sum(
                int(row["rng_collision_nonce"] > 0) for row in output_candidates
            ),
            "candidate_manifest_raw_sha256": sha256(output_root / "candidate_manifest_raw.json"),
            "candidate_uid_manifest_sha256": sha256(output_root / "candidate_uid_manifest.csv"),
        },
    )
    return output_candidates


def select_unique_candidates(
    candidates: list[dict[str, Any]],
    *,
    budget_per_host: int,
    target_per_host: int = TARGET_UNIQUE_PER_HOST,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen validity/UID/final-hash/canonical-order rule."""
    by_host: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if int(row["host_generation_index"]) < budget_per_host:
            by_host[int(row["host_index"])].append(row)
    seen_uids: set[str] = set()
    seen_hashes: set[str] = set()
    retained: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    host_summaries: list[dict[str, Any]] = []

    for host in sorted(by_host):
        rows = sorted(by_host[host], key=lambda row: int(row["host_generation_index"]))
        retained_host = 0
        reason_counts: Counter[str] = Counter()
        host_uid_set: set[str] = set()
        host_seed_set: set[int] = set()
        host_hash_set: set[str] = set()
        for row in rows:
            uid = str(row["candidate_uid"])
            seed = int(row["rng_seed_64"])
            image_hash = str(row["final_image_sha256"])
            reason = ""
            keep = False
            if not bool(row.get("valid_candidate", False)):
                reason = "invalid_candidate"
            elif uid in seen_uids:
                reason = "duplicate_candidate_uid"
            elif image_hash in seen_hashes:
                reason = "duplicate_final_image_sha256"
            elif retained_host >= target_per_host:
                reason = "beyond_first_2048_unique_valid"
            else:
                keep = True
                retained_host += 1
                seen_uids.add(uid)
                seen_hashes.add(image_hash)
                retained.append(row)
            host_uid_set.add(uid)
            host_seed_set.add(seed)
            host_hash_set.add(image_hash)
            if reason:
                reason_counts[reason] += 1
            integrity_rows.append(
                {
                    "candidate_id": int(row["candidate_id"]),
                    "host_index": host,
                    "original_v47_host_id": int(row["original_v47_host_id"]),
                    "host_generation_index": int(row["host_generation_index"]),
                    "candidate_uid": uid,
                    "rng_seed_64": seed,
                    "final_image_sha256": image_hash,
                    "valid_candidate": bool(row.get("valid_candidate", False)),
                    "retained": keep,
                    "exclusion_reason": reason,
                }
            )
        host_summaries.append(
            {
                "host_index": host,
                "original_v47_host_id": int(rows[0]["original_v47_host_id"]),
                "raw_candidate_count": len(rows),
                "unique_uid_count": len(host_uid_set),
                "unique_rng_seed_count": len(host_seed_set),
                "unique_final_image_count": len(host_hash_set),
                "duplicate_final_image_rows": len(rows) - len(host_hash_set),
                "invalid_candidate_count": reason_counts["invalid_candidate"],
                "retained_unique_valid_count": retained_host,
                "pass": retained_host == target_per_host,
            }
        )
    summary = {
        "budget_per_host": budget_per_host,
        "target_unique_per_host": target_per_host,
        "host_count": len(host_summaries),
        "all_hosts_pass": bool(
            len(host_summaries) == 30 and all(row["pass"] for row in host_summaries)
        ),
        "retained_count": len(retained),
        "unique_retained_uid_count": len({str(row["candidate_uid"]) for row in retained}),
        "unique_retained_rng_seed_count": len({int(row["rng_seed_64"]) for row in retained}),
        "unique_retained_final_image_count": len(
            {str(row["final_image_sha256"]) for row in retained}
        ),
        "per_host": host_summaries,
    }
    return retained, integrity_rows, summary


def write_retained_pool(
    output_root: Path,
    retained: list[dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[int, Path]:
    """Persist the exact 2,048-unique-per-host scientific pool."""
    raw_coco = load_json(output_root / "annotations" / "instances_raw.json")
    keep_ids = {int(row["candidate_id"]) for row in retained}
    images = [row for row in raw_coco["images"] if int(row["id"]) in keep_ids]
    annotations = [
        row for row in raw_coco["annotations"] if int(row["image_id"]) in keep_ids
    ]
    write_json(output_root / "candidate_manifest.json", {"candidates": retained})
    write_csv(output_root / "candidate_integrity_manifest.csv", integrity_rows)
    write_json(
        output_root / "annotations" / "instances_train.json",
        {
            "info": {
                **raw_coco.get("info", {}),
                "retention_rule": (
                    "valid; unique full UID; unique final decoded JPEG SHA-256; "
                    "canonical generation order; first 2048 per host"
                ),
            },
            "images": images,
            "annotations": annotations,
            "categories": raw_coco["categories"],
        },
    )
    summary = copy.deepcopy(summary)
    summary.update(
        {
            "candidate_manifest_sha256": sha256(output_root / "candidate_manifest.json"),
            "candidate_integrity_manifest_sha256": sha256(
                output_root / "candidate_integrity_manifest.csv"
            ),
            "candidate_coco_sha256": sha256(
                output_root / "annotations" / "instances_train.json"
            ),
        }
    )
    write_json(output_root / "integrity_summary.json", summary)
    return {
        int(row["candidate_id"]): output_root
        / "images"
        / "train"
        / f"cand_{int(row['candidate_id']):07d}.jpg"
        for row in retained
    }


def load_raw_candidates(output_root: Path) -> list[dict[str, Any]]:
    return load_json(output_root / "candidate_manifest_raw.json")["candidates"]
