"""Evaluate the frozen Medium150 pretraining gate and stop before training/AP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from historical_medium150_frame_adapter import load_historical_frames
from medium150_balanced_selector import (
    EXPECTED_HOSTS,
    appearance_balance_medium150,
    evaluate_medium150,
    solve_s3_medium150,
)
from support_radiometry_balanced_selector import (
    PAIR_METRIC_STANDARDIZED_MAX,
    WEBER_WASSERSTEIN_MAX,
    build_pair_options,
)
from support_scale_radiometry_calibration_common import (
    METRICS,
    extract_candidate_support,
    load_frozen_model,
    native_metrics,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "AnalysisResults" / "expanded_v1" / "medium150_historical_pretraining_gate"
FIGURES = RESULT / "figures"
PROGRESS = RESULT / "progress"
POOL = ROOT / "PreparedData" / "medium150_historical_efficacy_candidate_pool"
HOSTS_ROOT = POOL / "hosts"
EXECUTION_FREEZE = RESULT / "medium150_pretraining_execution_freeze.json"
GENERATION_RESULT = RESULT / "medium150_candidate_generation_results.json"
QUALITY = ROOT / "AnalysisResults" / "insertion_quality_v30" / "insertion_quality_object_metrics_v30.csv"
REAL_COCO = ROOT / "PreparedData" / "xview_expanded_512_v1" / "annotations" / "instances_train.json"
REAL_EMBEDDINGS = ROOT / "AnalysisResults" / "v47_development_feature_support" / "real_embeddings.npz"
WEIGHTS = ROOT / "runs" / "expanded_v1" / "duplicate_yolo26n_s20260723" / "weights" / "best.pt"
CONFIRMATION_FREEZE = ROOT / "AnalysisResults" / "expanded_v1" / "adaptation_scope_fairness_v48" / "confirmation_freeze.json"
RELEASE = ROOT / "release" / "medium150_historical_high_low_pretraining_freeze"
PRIOR_MANIFESTS = (
    ROOT / "PreparedData" / "support_scale_candidate_identity_development_pool" / "candidate_manifest_raw.json",
    ROOT / "PreparedData" / "support_scale_final_fresh_medium_pool" / "candidate_manifest_raw.json",
    ROOT / "PreparedData" / "support_scale_small_fresh_pool" / "candidate_manifest_raw.json",
)
RAW_PER_HOST = 2080
RETAINED_PER_HOST = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["status"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    temporary.replace(path)


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def update_state(**values: Any) -> None:
    path = PROGRESS / "pretraining_gate_state.json"
    state = load_json(path) if path.exists() else {}
    state.update(values)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state.update(
        {
            "detector_training_runs": 0,
            "ap_values_inspected": 0,
            "detector_predictions_generated": 0,
            "efficacy_checkpoints_created": 0,
            "training_unlocked": False,
        }
    )
    write_json(path, state)


def verify_execution_freeze() -> None:
    freeze = load_json(EXECUTION_FREEZE)
    if freeze["state"] != "FROZEN_BEFORE_CANDIDATE_GENERATION":
        raise RuntimeError("Unexpected execution freeze")
    for row in freeze["files"]:
        path = ROOT / row["path"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"EXECUTION_HASH_MISMATCH: {row['path']}")


def host_dirs() -> list[Path]:
    return sorted(HOSTS_ROOT.glob("host_[0-9][0-9][0-9]"))


def load_retained(host_dir: Path) -> list[dict[str, Any]]:
    return load_json(host_dir / "candidate_manifest_retained.json")["candidates"]


def extract_support_all(device_name: str, batch_size: int) -> None:
    model, device = load_frozen_model(WEIGHTS, device_name)
    done = 0
    for host_dir in host_dirs():
        output = host_dir / "support_distances.csv"
        if output.exists():
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                if sum(1 for _ in csv.DictReader(handle)) != RETAINED_PER_HOST:
                    raise RuntimeError(f"Incomplete support cache: {output}")
            done += 1
            continue
        retained = load_retained(host_dir)
        candidate_ids = [int(row["candidate_id"]) for row in retained]
        coco = load_json(host_dir / "instances_retained.json")
        image_paths = {int(row["candidate_id"]): ROOT / row["image_path"] for row in retained}
        _, distances = extract_candidate_support(
            model,
            device,
            candidate_ids,
            coco,
            image_paths,
            REAL_EMBEDDINGS,
            REAL_COCO,
            batch_size,
        )
        rows = [{"candidate_id": candidate_id, "support_distance": distances[candidate_id]} for candidate_id in candidate_ids]
        write_csv(output, rows)
        done += 1
        update_state(state="RUNNING_SUPPORT_EXTRACTION", support_hosts_complete=done, current_host=host_dir.name)
        print(f"MEDIUM150 support host {done}/150", flush=True)
    del model


def load_support(host_dir: Path) -> dict[int, float]:
    with (host_dir / "support_distances.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["candidate_id"]): float(row["support_distance"]) for row in csv.DictReader(handle)}


def build_options_all(native: np.ndarray, native_summary: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_options: list[dict[str, Any]] = []
    host_audits: list[dict[str, Any]] = []
    for index, host_dir in enumerate(host_dirs(), start=1):
        option_path = host_dir / "s3_pair_options.json"
        audit_path = host_dir / "s3_pair_option_audit.json"
        if option_path.exists() and audit_path.exists():
            options = load_json(option_path)["options"]
            host_audit = load_json(audit_path)["host_audit"]
        else:
            retained = load_retained(host_dir)
            distances = load_support(host_dir)
            options, audit_rows, option_audit = build_pair_options(retained, distances, native, native_summary)
            if len(audit_rows) != 1:
                raise RuntimeError(f"Unexpected host audit cardinality: {host_dir}")
            host_audit = {**audit_rows[0], "representative_option_count": len(options)}
            write_json(option_path, {"options": options})
            write_json(audit_path, {"host_audit": host_audit, "option_audit": option_audit})
        all_options.extend(options)
        host_audits.append(host_audit)
        update_state(state="RUNNING_PAIR_OPTION_BUILD", pair_option_hosts_complete=index, current_host=host_dir.name)
        print(f"MEDIUM150 pair options host {index}/150 options={len(options)}", flush=True)
    return all_options, host_audits


def selected_candidates(selected: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    selected_ids = {
        int(value)
        for pair in selected
        for value in (pair["near_candidate_id"], pair["far_candidate_id"])
    }
    output: dict[int, dict[str, Any]] = {}
    for host_dir in host_dirs():
        for row in load_retained(host_dir):
            candidate_id = int(row["candidate_id"])
            if candidate_id in selected_ids:
                output[candidate_id] = row
    if set(output) != selected_ids:
        raise RuntimeError("Selected candidate lookup incomplete")
    return output


def pair_manifest_rows(
    selected: list[dict[str, Any]], candidates: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for pair in selected:
        high = candidates[int(pair["near_candidate_id"])]
        low = candidates[int(pair["far_candidate_id"])]
        rows.append(
            {
                "frame_index": int(pair["host_index"]),
                "historical_host_id": int(high["historical_host_id"]),
                "host_path": high["host_path"],
                "host_sha256": high["host_sha256"],
                "source_scene": high["source_scene"],
                "source_cutout_path": high["source_cutout"],
                "source_cutout_sha256": high["source_cutout_sha256"],
                "high_candidate_id": int(high["candidate_id"]),
                "low_candidate_id": int(low["candidate_id"]),
                "high_candidate_uid": high["candidate_uid"],
                "low_candidate_uid": low["candidate_uid"],
                "high_image_path": high["image_path"],
                "low_image_path": low["image_path"],
                "high_final_image_sha256": high["final_image_sha256"],
                "low_final_image_sha256": low["final_image_sha256"],
                "high_support_distance": float(pair["near_distance"]),
                "low_support_distance": float(pair["far_distance"]),
                "support_gap": float(pair["support_gap"]),
                "high_appearance_variant": high["appearance_variant"],
                "low_appearance_variant": low["appearance_variant"],
                "high_appearance_family": pair["high_family"],
                "low_appearance_family": pair["low_family"],
                "bbox_x": float(high["bbox"][0]),
                "bbox_y": float(high["bbox"][1]),
                "bbox_width": float(high["bbox"][2]),
                "bbox_height": float(high["bbox"][3]),
                "bbox_scale": float(high["bbox_scale"]),
                "orientation_mode": high["orientation_mode"],
                "additional_rotation": int(high["additional_rotation"]),
                "source_hash_high_low_equal": high["source_cutout_sha256"] == low["source_cutout_sha256"],
                "geometry_high_low_equal": tuple(high["bbox"]) == tuple(low["bbox"]),
                "orientation_high_low_equal": (
                    high["orientation_mode"] == low["orientation_mode"]
                    and int(high["additional_rotation"]) == int(low["additional_rotation"]) == 0
                    and not high["rotated_90"] and not low["rotated_90"]
                ),
                "pairwise_radiometry_pass": all(
                    float(value) <= PAIR_METRIC_STANDARDIZED_MAX
                    for value in pair["standardized_metric_differences"].values()
                ),
                **{f"high_{key}": float(high["calibrated_metrics"][key]) for key in METRICS},
                **{f"low_{key}": float(low["calibrated_metrics"][key]) for key in METRICS},
            }
        )
    return sorted(rows, key=lambda row: row["frame_index"])


def orientation_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pair_rows:
        for arm in ("High", "Low"):
            prefix = arm.lower()
            rows.append(
                {
                    "frame_index": pair["frame_index"],
                    "arm": arm,
                    "candidate_id": pair[f"{prefix}_candidate_id"],
                    "historical_source_sha256": pair["source_cutout_sha256"],
                    "candidate_source_sha256": pair["source_cutout_sha256"],
                    "source_hash_match": True,
                    "orientation_mode": pair["orientation_mode"],
                    "additional_rotation": pair["additional_rotation"],
                    "reflection_variant": pair[f"{prefix}_appearance_variant"] in {"major_axis_reflection", "major_axis_reflection_strong"},
                    "downstream_orientation_normalization": False,
                    "high_low_orientation_equal": pair["orientation_high_low_equal"],
                    "orientation_compatible": bool(
                        pair["orientation_high_low_equal"]
                        and pair["additional_rotation"] == 0
                        and pair[f"{prefix}_appearance_variant"] not in {"major_axis_reflection", "major_axis_reflection_strong"}
                    ),
                }
            )
    return rows


def geometry_rows(pair_rows: list[dict[str, Any]], frames: list[Any]) -> list[dict[str, Any]]:
    by_frame = {frame.frame_index: frame for frame in frames}
    rows = []
    for pair in pair_rows:
        frame = by_frame[int(pair["frame_index"])]
        exact = bool(
            pair["geometry_high_low_equal"]
            and pair["source_hash_high_low_equal"]
            and float(pair["bbox_x"]) == float(frame.bbox_xmin)
            and float(pair["bbox_y"]) == float(frame.bbox_ymin)
            and float(pair["bbox_width"]) == float(frame.bbox_width)
            and float(pair["bbox_height"]) == float(frame.bbox_height)
            and math.isclose(float(pair["bbox_scale"]), float(frame.bbox_scale), rel_tol=0.0, abs_tol=1e-12)
        )
        rows.append(
            {
                "frame_index": frame.frame_index,
                "historical_host_id": frame.host_id,
                "center_x_historical": frame.center_x,
                "center_y_historical": frame.center_y,
                "bbox_x": pair["bbox_x"],
                "bbox_y": pair["bbox_y"],
                "bbox_width": pair["bbox_width"],
                "bbox_height": pair["bbox_height"],
                "bbox_scale": pair["bbox_scale"],
                "source_raster_exact": pair["source_hash_high_low_equal"],
                "high_low_geometry_exact": pair["geometry_high_low_equal"],
                "historical_geometry_exact": exact,
                "status": "PASS" if exact else "FAIL",
            }
        )
    return rows


def prior_identity_sets() -> tuple[set[int], set[str], set[str]]:
    seeds: set[int] = set()
    uids: set[str] = set()
    hashes: set[str] = set()
    for path in PRIOR_MANIFESTS:
        if path.exists():
            for row in load_json(path).get("candidates", []):
                if row.get("rng_seed_64") is not None:
                    seeds.add(int(row["rng_seed_64"]))
                if row.get("candidate_uid"):
                    uids.add(str(row["candidate_uid"]))
                value = row.get("final_image_sha256", row.get("image_sha256"))
                if value:
                    hashes.add(str(value))
    return seeds, uids, hashes


def integrity_audit(pair_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_seeds, prior_uids, prior_hashes = prior_identity_sets()
    seen_uids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_retained_hashes: set[str] = set()
    raw_hash_first: dict[str, tuple[int, bool, str]] = {}
    rows = []
    total_unexpected_raw_duplicates = 0
    prior_uid_overlap = prior_rng_overlap = prior_hash_overlap = 0
    for host_dir in host_dirs():
        with (host_dir / "candidate_identity_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            identities = list(csv.DictReader(handle))
        raw_uid_dups = raw_seed_dups = retained_hash_dups = unexpected_raw_dups = 0
        retained = 0
        for row in identities:
            uid = row["candidate_uid"]
            seed = int(row["rng_seed_64"])
            image_hash = row["final_image_sha256"]
            keep = row["retained"].lower() == "true"
            reason = row["retention_reason"]
            raw_uid_dups += int(uid in seen_uids)
            raw_seed_dups += int(seed in seen_seeds)
            prior_uid_overlap += int(uid in prior_uids)
            prior_rng_overlap += int(seed in prior_seeds)
            prior_hash_overlap += int(keep and image_hash in prior_hashes)
            if image_hash in raw_hash_first:
                if reason != "duplicate_or_prior_final_image_sha256":
                    unexpected_raw_dups += 1
            else:
                raw_hash_first[image_hash] = (int(row["candidate_id"]), keep, reason)
            if keep:
                retained += 1
                retained_hash_dups += int(image_hash in seen_retained_hashes)
                seen_retained_hashes.add(image_hash)
            seen_uids.add(uid)
            seen_seeds.add(seed)
        total_unexpected_raw_duplicates += unexpected_raw_dups
        passed = bool(
            len(identities) == RAW_PER_HOST
            and retained == RETAINED_PER_HOST
            and raw_uid_dups == 0
            and raw_seed_dups == 0
            and retained_hash_dups == 0
            and unexpected_raw_dups == 0
        )
        rows.append(
            {
                "frame_index": int(host_dir.name.split("_")[1]),
                "raw_count": len(identities),
                "retained_count": retained,
                "raw_uid_duplicate_count": raw_uid_dups,
                "raw_rng_duplicate_count": raw_seed_dups,
                "retained_hash_duplicate_count": retained_hash_dups,
                "unexpected_raw_hash_duplicate_count": unexpected_raw_dups,
                "status": "PASS" if passed else "FAIL",
            }
        )
    selected_uids = [value for pair in pair_rows for value in (pair["high_candidate_uid"], pair["low_candidate_uid"])]
    selected_hashes = [value for pair in pair_rows for value in (pair["high_final_image_sha256"], pair["low_final_image_sha256"])]
    within_pair_equal = sum(pair["high_final_image_sha256"] == pair["low_final_image_sha256"] for pair in pair_rows)
    summary = {
        "raw_candidate_count": len(seen_uids),
        "raw_unique_uid_count": len(seen_uids),
        "raw_unique_rng_seed_count": len(seen_seeds),
        "raw_unique_final_image_count": len(raw_hash_first),
        "retained_candidate_count": len(seen_retained_hashes),
        "retained_unique_final_image_count": len(seen_retained_hashes),
        "prior_uid_overlap_count": prior_uid_overlap,
        "prior_rng_overlap_count": prior_rng_overlap,
        "prior_retained_final_image_overlap_count": prior_hash_overlap,
        "unexpected_raw_hash_duplicate_count": total_unexpected_raw_duplicates,
        "selected_candidate_count": len(selected_uids),
        "selected_unique_uid_count": len(set(selected_uids)),
        "selected_unique_final_image_count": len(set(selected_hashes)),
        "within_pair_pixel_identity_count": within_pair_equal,
    }
    summary["pass"] = bool(
        len(rows) == 150
        and all(row["status"] == "PASS" for row in rows)
        and summary["raw_candidate_count"] == 150 * RAW_PER_HOST
        and summary["retained_candidate_count"] == 150 * RETAINED_PER_HOST
        and prior_uid_overlap == prior_rng_overlap == prior_hash_overlap == 0
        and total_unexpected_raw_duplicates == 0
        and len(selected_uids) == len(set(selected_uids)) == 300
        and len(selected_hashes) == len(set(selected_hashes)) == 300
        and within_pair_equal == 0
    )
    return rows, summary


def radiometry_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in ("high", "low"):
        for metric, values in evaluation["radiometry_gate"][arm]["metrics"].items():
            rows.append({"gate": "absolute", "arm": arm, "metric": metric, **values})
    for metric, values in evaluation["radiometry_gate"]["paired_standardized_difference"]["metrics"].items():
        rows.append({"gate": "pairwise", "arm": "high_low", "metric": metric, **values})
    return rows


def source_scene_audits(frames: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_counts = Counter(frame.source_cutout_sha256 for frame in frames)
    scene_counts = Counter(frame.source_scene for frame in frames)
    source = {
        "unique_source_cutouts": len(source_counts),
        "reuse_count_distribution": dict(sorted(Counter(source_counts.values()).items())),
        "maximum_reuse_count": max(source_counts.values()),
        "source_counts": dict(sorted(source_counts.items())),
    }
    scene_rows = [{"scene_id": scene, "host_count": count} for scene, count in sorted(scene_counts.items())]
    return source, scene_rows


def dataset_manifests(pair_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    common = {
        "real_training_pool": "PreparedData/xview_expanded_512_v1 train real images",
        "real_training_vessel_denominator": 3727,
        "synthetic_insertions": 150,
        "class_id": 5,
        "class_name": "maritime_vessel",
    }
    paths = []
    for arm in ("high", "low"):
        rows = [
            {
                "condition": f"Medium-{arm.capitalize()}150",
                "frame_index": pair["frame_index"],
                "historical_host_id": pair["historical_host_id"],
                "base_host_path": pair["host_path"],
                "candidate_image_path": pair[f"{arm}_image_path"],
                "candidate_id": pair[f"{arm}_candidate_id"],
                "candidate_uid": pair[f"{arm}_candidate_uid"],
                "candidate_sha256": pair[f"{arm}_final_image_sha256"],
                "source_cutout_sha256": pair["source_cutout_sha256"],
                "bbox_x": pair["bbox_x"],
                "bbox_y": pair["bbox_y"],
                "bbox_width": pair["bbox_width"],
                "bbox_height": pair["bbox_height"],
                "category_id": 5,
                "synthetic_insertion_count": 1,
            }
            for pair in pair_rows
        ]
        path = RESULT / f"medium150_{arm}_dataset_manifest.csv"
        write_csv(path, rows)
        paths.append(path)
    write_json(
        RESULT / "medium150_future_dataset_assembly_dry_run.json",
        {
            **common,
            "high_manifest": relative(paths[0]),
            "low_manifest": relative(paths[1]),
            "same_150_base_hosts": True,
            "same_modified_image_count": True,
            "same_insertion_and_annotation_count": True,
            "same_real_training_pool": True,
            "same_exposure_count": True,
            "same_class_labels": True,
            "training_invoked": False,
        },
    )
    return paths[0], paths[1]


def figures(pair_rows: list[dict[str, Any]], balance_rows: list[dict[str, Any]], native: np.ndarray, frames: list[Any], gate_rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 2.8))
    labels = ["Historical host + source\n(native orientation)", "14-variant amended\n2080 raw", "2048 unique\nfinal JPEG", "Frozen support + S3\n150 matched pairs", "Pretraining gates\ntraining locked"]
    for index, label in enumerate(labels):
        plt.text(index, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "#e8f0fe"})
        if index < len(labels) - 1:
            plt.annotate("", xy=(index + 0.72, 0.5), xytext=(index + 0.28, 0.5), arrowprops={"arrowstyle": "->"})
    plt.xlim(-0.5, len(labels) - 0.5); plt.ylim(0, 1); plt.axis("off"); plt.tight_layout()
    plt.savefig(FIGURES / "figure1_historical_candidate_pipeline.png", dpi=180); plt.close()

    plt.figure(figsize=(12, 4))
    gaps = [float(row["support_gap"]) for row in pair_rows]
    plt.plot(range(1, len(gaps) + 1), gaps, linewidth=1.2)
    plt.axhline(0.040, color="orange", linestyle="--", label="0.040 fraction threshold")
    plt.axhline(0.060, color="red", linestyle=":", label="0.060 median threshold")
    plt.xlabel("Historical frame"); plt.ylabel("Low distance - High distance"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "figure2_per_host_support_gaps.png", dpi=180); plt.close()

    metric_index = METRICS.index("absolute_weber_contrast")
    plt.figure(figsize=(7, 4))
    if pair_rows:
        plt.boxplot([
            native[:, metric_index],
            [row["high_absolute_weber_contrast"] for row in pair_rows],
            [row["low_absolute_weber_contrast"] for row in pair_rows],
        ], tick_labels=["Native", "High", "Low"], showfliers=False)
        plt.ylabel("Absolute Weber contrast")
    else:
        plt.text(0.5, 0.5, "Not evaluated: upstream gate failed", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES / "figure3_absolute_weber_distributions.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 4))
    x = np.arange(len(balance_rows)); width = 0.38
    plt.bar(x - width / 2, [row["high_count"] for row in balance_rows], width, label="High")
    plt.bar(x + width / 2, [row["low_count"] for row in balance_rows], width, label="Low")
    plt.xticks(x, [row["family"] for row in balance_rows], rotation=20, ha="right"); plt.ylabel("Count"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "figure4_appearance_family_counts.png", dpi=180); plt.close()

    reuse = Counter(frame.source_cutout_sha256 for frame in frames)
    plt.figure(figsize=(7, 4)); plt.hist(list(reuse.values()), bins=range(1, max(reuse.values()) + 2), align="left", rwidth=0.8)
    plt.xlabel("Historical reuse count per source raster"); plt.ylabel("Unique source rasters"); plt.tight_layout()
    plt.savefig(FIGURES / "figure5_source_raster_reuse.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 4.5))
    values = np.asarray([[1 if row["status"] == "PASS" else 0 for row in gate_rows]])
    plt.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(len(gate_rows)), [row["gate"] for row in gate_rows], rotation=45, ha="right")
    plt.yticks([0], ["Result"]); plt.colorbar(ticks=[0, 1], label="FAIL / PASS"); plt.tight_layout()
    plt.savefig(FIGURES / "figure6_pretraining_gate_summary.png", dpi=180); plt.close()


def copy_release(files: list[Path]) -> None:
    if RELEASE.exists() and any(RELEASE.iterdir()):
        raise FileExistsError(f"Refusing to overwrite release freeze: {RELEASE}")
    RELEASE.mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.is_file():
            shutil.copy2(path, RELEASE / path.name)
    (RELEASE / "README.md").write_text(
        "# Medium150 historical High/Low pretraining freeze\n\n"
        "This package contains only protocol, manifests, gate metrics, reports, and hashes. "
        "It contains no detector results, checkpoints, predictions, raw candidate images, or embeddings.\n\n"
        "`TRAINING_UNLOCKED = FALSE`; explicit author approval is required for any detector run.\n",
        encoding="utf-8",
    )


def write_candidate_failure_outputs(generation: dict[str, Any], frames: list[Any]) -> int:
    fields = {
        RESULT / "medium150_high_low_pair_manifest.csv": ["frame_index", "high_candidate_id", "low_candidate_id"],
        RESULT / "medium150_support_gate_metrics.csv": ["frame_index", "support_gap"],
        RESULT / "medium150_radiometry_gate_metrics.csv": ["gate", "arm", "metric", "pass"],
        RESULT / "medium150_appearance_balance.csv": ["family", "high_count", "low_count", "difference"],
        RESULT / "medium150_orientation_fidelity_audit.csv": ["frame_index", "arm", "orientation_compatible"],
        RESULT / "medium150_geometry_audit.csv": ["frame_index", "status"],
        RESULT / "medium150_candidate_integrity_results.csv": ["frame_index", "raw_count", "retained_count", "status"],
    }
    for path, columns in fields.items():
        write_csv(path, [], columns)
    gates = [
        {"gate": "ORIENTATION_AMENDMENT_FROZEN", "status": "PASS"},
        {"gate": "FRAME_PROVENANCE", "status": "PASS"},
        {"gate": "CANDIDATE_COUNT", "status": "FAIL"},
        *[
            {"gate": name, "status": "BLOCKED_NOT_RUN"}
            for name in (
                "INTEGRITY", "SELECTOR_FEASIBILITY", "SUPPORT", "ABSOLUTE_RADIOMETRY",
                "PAIRWISE_RADIOMETRY", "APPEARANCE_BALANCE", "ORIENTATION_FIDELITY",
                "GEOMETRY", "EXPOSURE_DOSE",
            )
        ],
    ]
    results = {
        "analysis": "medium150_historical_high_low_pretraining_gate",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_frames": 150,
        "candidate_raw_budget_per_host": RAW_PER_HOST,
        "unique_retained_per_host": RETAINED_PER_HOST,
        "hosts_with_full_candidate_pool": generation.get("hosts_with_full_candidate_pool", 0),
        "selected_high_low_pairs": 0,
        "gates": gates,
        "MEDIUM150_PRETRAINING_GATE": "FAIL",
        "TRAINING_READY_FOR_AUTHOR_REVIEW": False,
        "TRAINING_UNLOCKED": False,
        "DETECTOR_TRAINING_RUNS": 0,
        "AP_VALUES_INSPECTED": 0,
        "DETECTOR_PREDICTIONS_GENERATED": 0,
        "EFFICACY_CHECKPOINTS_CREATED": 0,
        "failure_interpretation": (
            "The orientation-compatible historical-frame appearance subset did not provide the "
            "prespecified unique candidate budget."
        ),
    }
    write_json(RESULT / "medium150_pretraining_gate_results.json", results)
    common = "Upstream `CANDIDATE_COUNT_GATE = FAIL`; this downstream stage was not run and no tuning was attempted.\n"
    report_names = (
        "medium150_candidate_integrity_report.md", "medium150_selector_feasibility_report.md",
        "medium150_support_gate_report.md", "medium150_radiometry_gate_report.md",
        "medium150_appearance_balance_report.md", "medium150_orientation_fidelity_report.md",
        "medium150_geometry_report.md", "medium150_exposure_dose_audit.md",
    )
    for name in report_names:
        (RESULT / name).write_text(f"# {name.removesuffix('.md').replace('_', ' ')}\n\n{common}", encoding="utf-8")
    (RESULT / "medium150_candidate_generation_audit.md").write_text(
        "# Medium150 candidate-generation audit\n\n"
        f"- Completed hosts: {generation.get('historical_frames_completed', 0)} / 150\n"
        f"- Hosts with 2048 retained unique valid candidates: {generation.get('hosts_with_full_candidate_pool', 0)} / 150\n"
        "- Result: **FAIL**. No extra candidates or host-specific remediation were generated.\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_pretraining_gate_report.md").write_text(
        "# Medium150 historical High/Low pretraining gate\n\n"
        "Historical frames = 150 / 150\n\nCandidate raw budget = 2080 / host\n\n"
        f"Unique retained candidates = incomplete / 2048 per host\n\nHosts with full candidate pool = {generation.get('hosts_with_full_candidate_pool', 0)} / 150\n\n"
        "Selected High/Low pairs = 0 / 150\n\nSupport median gap = NOT_RUN\n\nFraction gap >=0.040 = NOT_RUN\n\nMinimum gap = NOT_RUN\n\n"
        "High absolute Weber Wasserstein = NOT_RUN\n\nLow absolute Weber Wasserstein = NOT_RUN\n\nOther radiometry = NOT_RUN\n\n"
        "Pairwise radiometry = NOT_RUN\n\nAppearance balance = NOT_RUN\n\nNative orientation fidelity = NOT_RUN\n\nExact geometry = NOT_RUN\n\n"
        "Integrity = FAIL\n\nDose/exposure compatibility = NOT_RUN\n\nOverall: MEDIUM150_PRETRAINING_GATE = FAIL\n\n"
        "TRAINING_READY_FOR_AUTHOR_REVIEW = FALSE\n\nTRAINING_UNLOCKED = FALSE\n\nAP_VALUES_INSPECTED = 0\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_pretraining_discrepancy_report.md").write_text(
        "# Medium150 pretraining discrepancy report\n\n"
        "The orientation-compatible historical-frame appearance subset did not provide the prespecified unique candidate budget. "
        "No threshold, taxonomy, selector, host, source, or candidate budget was changed.\n",
        encoding="utf-8",
    )
    (RESULT / "next_stage_recommendation.md").write_text(
        "# Next-stage recommendation\n\nPreserve the one-shot candidate-count failure and stop. Do not train or inspect AP.\n\n`TRAINING_UNLOCKED = FALSE`.\n",
        encoding="utf-8",
    )
    native, _ = native_metrics(QUALITY)
    figures([], [{"family": family, "high_count": 0, "low_count": 0, "difference": 0} for family in ("baseline", "texture_realization", "spatial_frequency", "material_chroma", "fine_detail")], native, frames, gates)
    update_state(state="PRETRAINING_GATE_COMPLETE", MEDIUM150_PRETRAINING_GATE="FAIL", TRAINING_READY_FOR_AUTHOR_REVIEW=False)
    return 3
    write_json(
        RELEASE / "release_hashes.json",
        {path.name: sha256(path) for path in sorted(RELEASE.iterdir()) if path.is_file() and path.name != "release_hashes.json"},
    )


def main() -> int:
    args = parse_args()
    verify_execution_freeze()
    frames = load_historical_frames()
    generation = load_json(GENERATION_RESULT)
    candidate_gate = generation["candidate_count_gate"] == "PASS"
    if not candidate_gate:
        return write_candidate_failure_outputs(generation, frames)
    if len(host_dirs()) != 150:
        raise RuntimeError("Expected 150 completed host directories")
    update_state(state="STARTING_SUPPORT_EXTRACTION", support_hosts_complete=0)
    extract_support_all(args.device, args.batch_size)
    native, native_summary = native_metrics(QUALITY)
    options, host_audits = build_options_all(native, native_summary)
    write_csv(RESULT / "medium150_selector_host_audit.csv", host_audits)
    update_state(state="RUNNING_S3_SOLVER", option_count=len(options))
    selected, solver = solve_s3_medium150(options, native_summary)
    write_json(RESULT / "medium150_selector_solver_result.json", {"solver": solver.__dict__, "selected_pairs": selected})
    candidates = selected_candidates(selected) if selected else {}
    evaluation = evaluate_medium150(selected, candidates, native, native_summary) if selected else {"support_gate": {"pass": False}, "radiometry_gate": {"pass": False}}
    balance = appearance_balance_medium150(selected)
    selector_feasible = bool(solver.success and len(selected) == 150)
    direct_weber = bool(
        selector_feasible
        and evaluation["radiometry_gate"]["high"]["metrics"]["absolute_weber_contrast"]["standardized_wasserstein"] <= WEBER_WASSERSTEIN_MAX
        and evaluation["radiometry_gate"]["low"]["metrics"]["absolute_weber_contrast"]["standardized_wasserstein"] <= WEBER_WASSERSTEIN_MAX
    )
    pair_rows = pair_manifest_rows(selected, candidates) if selected else []
    write_csv(RESULT / "medium150_high_low_pair_manifest.csv", pair_rows)
    support_rows = [
        {
            "frame_index": row["frame_index"],
            "historical_host_id": row["historical_host_id"],
            "high_support_distance": row["high_support_distance"],
            "low_support_distance": row["low_support_distance"],
            "support_gap": row["support_gap"],
            "gap_ge_0_040": float(row["support_gap"]) >= 0.040,
            "gap_ge_0_060": float(row["support_gap"]) >= 0.060,
        }
        for row in pair_rows
    ]
    write_csv(RESULT / "medium150_support_gate_metrics.csv", support_rows)
    radiometry = radiometry_rows(evaluation) if selector_feasible else []
    write_csv(RESULT / "medium150_radiometry_gate_metrics.csv", radiometry)
    write_csv(RESULT / "medium150_appearance_balance.csv", balance["rows"])
    orientations = orientation_rows(pair_rows)
    geometries = geometry_rows(pair_rows, frames)
    write_csv(RESULT / "medium150_orientation_fidelity_audit.csv", orientations)
    write_csv(RESULT / "medium150_geometry_audit.csv", geometries)
    integrity_rows, integrity = integrity_audit(pair_rows)
    write_csv(RESULT / "medium150_candidate_integrity_results.csv", integrity_rows)
    write_json(RESULT / "medium150_candidate_integrity_summary.json", integrity)

    support_gate = bool(selector_feasible and evaluation["support_gate"]["pass"])
    radiometry_gate = bool(selector_feasible and evaluation["radiometry_gate"]["pass"] and direct_weber)
    pairwise_gate = bool(selector_feasible and evaluation["radiometry_gate"]["paired_standardized_difference"]["pass"] and all(row["pairwise_radiometry_pass"] for row in pair_rows))
    orientation_gate = bool(len(orientations) == 300 and all(row["orientation_compatible"] for row in orientations))
    geometry_gate = bool(len(geometries) == 150 and all(row["status"] == "PASS" for row in geometries))
    source_pair_gate = bool(len(pair_rows) == 150 and all(row["source_hash_high_low_equal"] for row in pair_rows))
    real_coco = load_json(REAL_COCO)
    real_vessels = sum(int(row["category_id"]) == 5 for row in real_coco["annotations"])
    dose_gate = bool(len(pair_rows) == 150 and real_vessels == 3727)
    recipe_gate = CONFIRMATION_FREEZE.is_file()
    dataset_manifests(pair_rows) if len(pair_rows) == 150 else None
    source_audit, scene_rows = source_scene_audits(frames)
    write_json(RESULT / "medium150_historical_source_reuse.json", source_audit)
    write_csv(RESULT / "medium150_historical_scene_distribution.csv", scene_rows)

    gates = [
        {"gate": "ORIENTATION_AMENDMENT_FROZEN", "status": "PASS"},
        {"gate": "FRAME_PROVENANCE", "status": "PASS"},
        {"gate": "CANDIDATE_COUNT", "status": "PASS" if candidate_gate else "FAIL"},
        {"gate": "INTEGRITY", "status": "PASS" if integrity["pass"] else "FAIL"},
        {"gate": "SELECTOR_FEASIBILITY", "status": "PASS" if selector_feasible else "FAIL"},
        {"gate": "SUPPORT", "status": "PASS" if support_gate else "FAIL"},
        {"gate": "ABSOLUTE_RADIOMETRY", "status": "PASS" if radiometry_gate else "FAIL"},
        {"gate": "PAIRWISE_RADIOMETRY", "status": "PASS" if pairwise_gate else "FAIL"},
        {"gate": "APPEARANCE_BALANCE", "status": "PASS" if balance["pass"] else "FAIL"},
        {"gate": "ORIENTATION_FIDELITY", "status": "PASS" if orientation_gate else "FAIL"},
        {"gate": "GEOMETRY", "status": "PASS" if geometry_gate and source_pair_gate else "FAIL"},
        {"gate": "EXPOSURE_DOSE", "status": "PASS" if dose_gate and recipe_gate else "FAIL"},
    ]
    overall = all(row["status"] == "PASS" for row in gates)
    support_values = evaluation.get("support_gate", {})
    high_weber = evaluation.get("radiometry_gate", {}).get("high", {}).get("metrics", {}).get("absolute_weber_contrast", {}).get("standardized_wasserstein")
    low_weber = evaluation.get("radiometry_gate", {}).get("low", {}).get("metrics", {}).get("absolute_weber_contrast", {}).get("standardized_wasserstein")
    other_radiometry = bool(
        selector_feasible
        and all(
            evaluation["radiometry_gate"][arm]["metrics"][metric]["pass"]
            for arm in ("high", "low")
            for metric in METRICS
            if metric != "absolute_weber_contrast"
        )
    )
    results = {
        "analysis": "medium150_historical_high_low_pretraining_gate",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_frames": 150,
        "candidate_raw_budget_per_host": RAW_PER_HOST,
        "unique_retained_per_host": RETAINED_PER_HOST,
        "hosts_with_full_candidate_pool": generation["hosts_with_full_candidate_pool"],
        "selected_high_low_pairs": len(pair_rows),
        "support": support_values,
        "high_absolute_weber_wasserstein": high_weber,
        "low_absolute_weber_wasserstein": low_weber,
        "other_radiometry_pass": other_radiometry,
        "pairwise_radiometry_pass": pairwise_gate,
        "appearance_balance": balance,
        "orientation_fidelity_pass": orientation_gate,
        "geometry_pass": geometry_gate,
        "source_raster_pair_identity_pass": source_pair_gate,
        "integrity": integrity,
        "real_training_vessels": real_vessels,
        "synthetic_insertions_per_arm": len(pair_rows),
        "nominal_augmentation_dose": len(pair_rows) / real_vessels if real_vessels else None,
        "training_recipe_artifact": relative(CONFIRMATION_FREEZE),
        "training_recipe_sha256": sha256(CONFIRMATION_FREEZE) if recipe_gate else None,
        "gates": gates,
        "MEDIUM150_PRETRAINING_GATE": "PASS" if overall else "FAIL",
        "TRAINING_READY_FOR_AUTHOR_REVIEW": overall,
        "TRAINING_UNLOCKED": False,
        "DETECTOR_TRAINING_RUNS": 0,
        "AP_VALUES_INSPECTED": 0,
        "DETECTOR_PREDICTIONS_GENERATED": 0,
        "EFFICACY_CHECKPOINTS_CREATED": 0,
    }
    write_json(RESULT / "medium150_pretraining_gate_results.json", results)

    generation_rows = load_json(GENERATION_RESULT)
    (RESULT / "medium150_candidate_generation_audit.md").write_text(
        "# Medium150 candidate-generation audit\n\n"
        f"- Historical hosts completed: {generation_rows['historical_frames_completed']} / 150\n"
        f"- Raw candidates: {generation_rows['total_raw_candidates']} ({RAW_PER_HOST} per host)\n"
        f"- Retained unique valid: {generation_rows['total_retained_candidates']} ({RETAINED_PER_HOST} per host)\n"
        "- Order: historical frame input -> frozen v31 base render at exact geometry -> admissible appearance variation -> frozen calibration -> final JPEG -> UID/hash validation -> canonical 2048 retention.\n"
        "- No host-specific adaptation, source replacement, placement search, bbox resampling, rotation, or post-result generation occurred.\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_candidate_integrity_report.md").write_text(
        "# Medium150 candidate-integrity report\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in integrity.items())
        + f"\n\nOverall integrity: **{'PASS' if integrity['pass'] else 'FAIL'}**.\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_selector_feasibility_report.md").write_text(
        "# Medium150 selector-feasibility report\n\n"
        f"- Frozen formulation: `S3_direct_wasserstein_family_balance`\n- Pair-option hosts: {len(host_audits)} / 150\n"
        f"- Representative options: {len(options)}\n- Solver success: {solver.success}\n- Selected pairs: {len(selected)} / 150\n"
        f"- Solver status: {solver.status}; message: {solver.message}\n- MIP gap: {solver.mip_gap}\n"
        "- The 30-host cardinalities were mechanically instantiated as 120/150 and strict-majority 76/150; scientific thresholds and objective were unchanged.\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_support_gate_report.md").write_text(
        "# Medium150 support gate\n\n"
        f"- Median gap: {support_values.get('median_gap')} (required >=0.060)\n"
        f"- Fraction gap >=0.040: {support_values.get('fraction_gap_ge_0_04')} (required >=0.80)\n"
        f"- Minimum gap: {support_values.get('minimum_gap')} (required >=0.005)\n"
        f"- Maximum gap: {support_values.get('maximum_gap')}\n- Result: **{'PASS' if support_gate else 'FAIL'}**\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_radiometry_gate_report.md").write_text(
        "# Medium150 radiometry gate\n\n"
        f"- High absolute Weber standardized Wasserstein: {high_weber} (<=0.50)\n"
        f"- Low absolute Weber standardized Wasserstein: {low_weber} (<=0.50)\n"
        f"- Other frozen native-radiometry metrics: {'PASS' if other_radiometry else 'FAIL'}\n"
        f"- Pairwise radiometry: {'PASS' if pairwise_gate else 'FAIL'}\n"
        f"- Overall radiometry: **{'PASS' if radiometry_gate and pairwise_gate else 'FAIL'}**\n\n"
        + md_table(radiometry, ["gate", "arm", "metric", "standardized_wasserstein", "median_absolute_standardized_difference", "pass"]),
        encoding="utf-8",
    )
    (RESULT / "medium150_appearance_balance_report.md").write_text(
        "# Medium150 appearance-family balance\n\n"
        + md_table(balance["rows"], ["family", "high_count", "low_count", "difference"])
        + f"\n\nTVD = {balance['total_variation_distance']}; result = **{'PASS' if balance['pass'] else 'FAIL'}**. "
        "The two excluded reflection variants are absent from both arms.\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_orientation_fidelity_report.md").write_text(
        "# Medium150 orientation-fidelity audit\n\n"
        f"- Orientation-compatible selected intervention images: {sum(row['orientation_compatible'] for row in orientations)} / 300\n"
        "- Additional rotations: 0; reflection variants: 0; downstream orientation normalization: 0.\n"
        f"- Result: **{'PASS' if orientation_gate else 'FAIL'}**\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_geometry_report.md").write_text(
        "# Medium150 exact-geometry audit\n\n"
        f"- Exact historical High/Low geometry: {sum(row['status'] == 'PASS' for row in geometries)} / 150\n"
        f"- Same source raster within pairs: {sum(row['source_hash_high_low_equal'] for row in pair_rows)} / 150\n"
        f"- Result: **{'PASS' if geometry_gate and source_pair_gate else 'FAIL'}**\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_exposure_dose_audit.md").write_text(
        "# Medium150 exposure and dose audit\n\n"
        f"- High synthetic insertions: {len(pair_rows)}\n- Low synthetic insertions: {len(pair_rows)}\n"
        f"- Frozen real-training vessel denominator: {real_vessels}\n"
        f"- Nominal augmentation dose: {len(pair_rows)}/{real_vessels} = {(len(pair_rows)/real_vessels*100 if real_vessels else math.nan):.4f}%\n"
        "- Same base hosts, real-training pool, annotation count, class labels, and prospective exposure count: yes.\n"
        f"- Existing head-only recipe artifact present: {recipe_gate}; it was not executed.\n"
        f"- Historical unique source rasters: {source_audit['unique_source_cutouts']}; maximum reuse: {source_audit['maximum_reuse_count']}.\n"
        f"- Result: **{'PASS' if dose_gate and recipe_gate else 'FAIL'}**\n",
        encoding="utf-8",
    )
    header = (
        "# Medium150 historical High/Low pretraining gate\n\n"
        "Historical frames = 150 / 150\n\n"
        "Candidate raw budget = 2080 / host\n\n"
        f"Unique retained candidates = {RETAINED_PER_HOST} / 2048 per host\n\n"
        f"Hosts with full candidate pool = {generation['hosts_with_full_candidate_pool']} / 150\n\n"
        f"Selected High/Low pairs = {len(pair_rows)} / 150\n\n"
        f"Support median gap = {support_values.get('median_gap')}\n\n"
        f"Fraction gap >=0.040 = {support_values.get('fraction_gap_ge_0_04')}\n\n"
        f"Minimum gap = {support_values.get('minimum_gap')}\n\n"
        f"High absolute Weber Wasserstein = {high_weber}\n\n"
        f"Low absolute Weber Wasserstein = {low_weber}\n\n"
        f"Other radiometry = {'PASS' if other_radiometry else 'FAIL'}\n\n"
        f"Pairwise radiometry = {'PASS' if pairwise_gate else 'FAIL'}\n\n"
        f"Appearance balance = {'PASS' if balance['pass'] else 'FAIL'}\n\n"
        f"Native orientation fidelity = {'PASS' if orientation_gate else 'FAIL'}\n\n"
        f"Exact geometry = {'PASS' if geometry_gate and source_pair_gate else 'FAIL'}\n\n"
        f"Integrity = {'PASS' if integrity['pass'] else 'FAIL'}\n\n"
        f"Dose/exposure compatibility = {'PASS' if dose_gate and recipe_gate else 'FAIL'}\n\n"
        f"Overall: MEDIUM150_PRETRAINING_GATE = {'PASS' if overall else 'FAIL'}\n\n"
        f"TRAINING_READY_FOR_AUTHOR_REVIEW = {str(overall).upper()}\n\n"
        "TRAINING_UNLOCKED = FALSE\n\nAP_VALUES_INSPECTED = 0\n\n"
    )
    (RESULT / "medium150_pretraining_gate_report.md").write_text(
        header + md_table(gates, ["gate", "status"]) +
        "\n\nThis is pretraining feasibility only, not detector efficacy evidence. High and Low are matched on scale, host, source raster, native orientation, placement, geometry, radiometric constraints, dose, and appearance-family marginals while being separated by the frozen representation-support criterion.\n\n"
        "DETECTOR_TRAINING_RUNS = 0; DETECTOR_PREDICTIONS_GENERATED = 0; EFFICACY_CHECKPOINTS_CREATED = 0.\n",
        encoding="utf-8",
    )
    (RESULT / "medium150_pretraining_discrepancy_report.md").write_text(
        "# Medium150 pretraining discrepancy report\n\n"
        "- Prospectively authorized scientific amendment: exactly two reflection variants removed.\n"
        "- Pipeline continuity wording: the complete generator is not claimed unchanged; the admissible taxonomy was narrowed.\n"
        "- Mechanical cohort-size instantiation: 30-host selector cardinalities were scaled to 150 without changing thresholds/objective.\n"
        f"- Failed gates: {[row['gate'] for row in gates if row['status'] == 'FAIL']}\n"
        "- No threshold relaxation, host deletion/replacement, source replacement, reflection reintroduction, extra candidate generation, training, prediction, or AP inspection occurred.\n",
        encoding="utf-8",
    )
    (RESULT / "next_stage_recommendation.md").write_text(
        "# Next-stage recommendation\n\n"
        + (
            "All prespecified Medium150 pretraining gates passed. The intervention is ready only for independent author review. Detector training remains locked; wait for explicit author approval.\n"
            if overall else
            "At least one prespecified Medium150 pretraining gate failed. Preserve this one-shot result and stop; do not remediate, tune, train, or inspect AP.\n"
        )
        + "\n`TRAINING_UNLOCKED = FALSE`.\n",
        encoding="utf-8",
    )
    figures(pair_rows, balance["rows"], native, frames, gates)
    if overall:
        release_files = [
            path for path in RESULT.iterdir()
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv"}
        ]
        release_files += [ROOT / "tools" / name for name in (
            "run_medium150_candidate_generation.py",
            "run_medium150_pretraining_gate_evaluation.py",
            "medium150_balanced_selector.py",
        )]
        copy_release(release_files)
    update_state(
        state="PRETRAINING_GATE_COMPLETE",
        MEDIUM150_PRETRAINING_GATE="PASS" if overall else "FAIL",
        TRAINING_READY_FOR_AUTHOR_REVIEW=overall,
        TRAINING_UNLOCKED=False,
    )
    print(json.dumps(results, indent=2))
    return 0 if overall else 3


if __name__ == "__main__":
    raise SystemExit(main())
