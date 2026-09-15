"""Candidate-level integrity and cross-cohort contamination validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from support_scale_radiometry_calibration_common import load_json


def load_candidate_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            payload = load_json(path)
            rows.extend(payload.get("candidates", []))
    return rows


def duplicated_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    output = []
    for value, members in sorted(grouped.items()):
        if len(members) > 1:
            output.append(
                {
                    key: value,
                    "multiplicity": len(members),
                    "candidate_ids": [int(row["candidate_id"]) for row in members],
                    "original_v47_host_ids": [
                        int(row["original_v47_host_id"]) for row in members
                    ],
                }
            )
    return output


def validate_candidate_integrity(
    *,
    raw_rows: list[dict[str, Any]],
    retained_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    selected_candidate_ids: Iterable[int] = (),
    fresh_host_hashes: Iterable[str] = (),
    prior_host_hashes: Iterable[str] = (),
    fresh_scenes: Iterable[str] = (),
    prior_scenes: Iterable[str] = (),
) -> dict[str, Any]:
    selected_ids = {int(value) for value in selected_candidate_ids}
    selected = [row for row in retained_rows if int(row["candidate_id"]) in selected_ids]
    prior_uids = {str(row["candidate_uid"]) for row in prior_rows if row.get("candidate_uid")}
    prior_hashes = {
        str(row.get("final_image_sha256", row.get("image_sha256"))) for row in prior_rows
    }
    prior_seeds = {
        int(row.get("rng_seed_64", row.get("appearance_seed")))
        for row in prior_rows
        if row.get("rng_seed_64", row.get("appearance_seed")) is not None
    }
    raw_uids = [str(row["candidate_uid"]) for row in raw_rows]
    raw_seeds = [int(row["rng_seed_64"]) for row in raw_rows]
    raw_hashes = [str(row["final_image_sha256"]) for row in raw_rows]
    retained_uids = {str(row["candidate_uid"]) for row in retained_rows}
    retained_seeds = {int(row["rng_seed_64"]) for row in retained_rows}
    retained_hashes = {str(row["final_image_sha256"]) for row in retained_rows}
    selected_uids = {str(row["candidate_uid"]) for row in selected}
    selected_hashes = {str(row["final_image_sha256"]) for row in selected}

    counts = Counter(int(row["original_v47_host_id"]) for row in retained_rows)
    raw_uid_groups = duplicated_groups(raw_rows, "candidate_uid")
    raw_hash_groups = duplicated_groups(raw_rows, "final_image_sha256")
    result = {
        "raw_candidate_count": len(raw_rows),
        "raw_unique_uid_count": len(set(raw_uids)),
        "raw_unique_rng_seed_64_count": len(set(raw_seeds)),
        "raw_unique_final_image_count": len(set(raw_hashes)),
        "raw_duplicate_uid_groups": raw_uid_groups,
        "raw_duplicate_final_image_groups": raw_hash_groups,
        "retained_candidate_count": len(retained_rows),
        "retained_per_host_counts": {str(key): value for key, value in sorted(counts.items())},
        "retained_unique_uid_count": len(retained_uids),
        "retained_unique_rng_seed_64_count": len(retained_seeds),
        "retained_unique_final_image_count": len(retained_hashes),
        "prior_uid_overlap_count": len(retained_uids & prior_uids),
        "prior_rng_overlap_count": len(retained_seeds & prior_seeds),
        "prior_final_image_overlap_count": len(retained_hashes & prior_hashes),
        "host_image_hash_overlap_count": len(
            set(map(str, fresh_host_hashes)) & set(map(str, prior_host_hashes))
        ),
        "source_scene_overlap_count": len(
            set(map(str, fresh_scenes)) & set(map(str, prior_scenes))
        ),
        "selected_candidate_count": len(selected),
        "selected_unique_uid_count": len(selected_uids),
        "selected_unique_final_image_count": len(selected_hashes),
        "selected_prior_uid_overlap_count": len(selected_uids & prior_uids),
        "selected_prior_final_image_overlap_count": len(selected_hashes & prior_hashes),
    }
    result["candidate_uid_gate"] = bool(
        len(set(raw_uids)) == len(raw_rows)
        and len(retained_uids) == len(retained_rows)
        and result["prior_uid_overlap_count"] == 0
    )
    result["candidate_uniqueness_gate"] = bool(
        len(retained_rows) == 30 * 2048
        and len(counts) == 30
        and set(counts.values()) == {2048}
        and len(retained_hashes) == len(retained_rows)
    )
    result["content_contamination_gate"] = bool(
        result["prior_final_image_overlap_count"] == 0
        and result["host_image_hash_overlap_count"] == 0
        and result["source_scene_overlap_count"] == 0
        and result["prior_uid_overlap_count"] == 0
    )
    result["rng_collision_gate"] = bool(
        len(set(raw_seeds)) == len(raw_rows) and result["prior_rng_overlap_count"] == 0
    )
    result["selected_intervention_integrity_gate"] = bool(
        len(selected) == 60
        and len(selected_uids) == 60
        and len(selected_hashes) == 60
        and result["selected_prior_uid_overlap_count"] == 0
        and result["selected_prior_final_image_overlap_count"] == 0
    )
    result["critical_content_contamination_count"] = (
        result["prior_final_image_overlap_count"]
        + result["host_image_hash_overlap_count"]
        + result["source_scene_overlap_count"]
        + result["prior_uid_overlap_count"]
    )
    result["technical_rng_collision_count"] = (
        len(raw_seeds) - len(set(raw_seeds)) + result["prior_rng_overlap_count"]
    )
    return result
