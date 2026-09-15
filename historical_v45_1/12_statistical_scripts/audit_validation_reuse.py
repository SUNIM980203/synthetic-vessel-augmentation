#!/usr/bin/env python3
"""Audit historical xView-validation reuse for the IEEE Access v39 revision.

This is a read-only scientific provenance audit. It performs no training or
inference. Outputs are written to the isolated v39 revision directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ieee_access_manuscript_v39_revision"

INITIAL_ARGS = ROOT / "runs/expanded_v1/duplicate_yolo26n_s20260723/args.yaml"
INITIAL_RESULTS = ROOT / "runs/expanded_v1/duplicate_yolo26n_s20260723/results.csv"
INITIAL_BEST = ROOT / "runs/expanded_v1/duplicate_yolo26n_s20260723/weights/best.pt"
INITIAL_YAML = ROOT / "config/yolo_xview_expanded_duplicate_v1.yaml"
V48_YAML = ROOT / "config/yolo_xview_expanded_real_v1.yaml"
V48_EVALUATOR = ROOT / "tools/evaluate_v48_scope_validation.py"
V48_FREEZE = ROOT / "AnalysisResults/expanded_v1/adaptation_scope_fairness_v48/validation_freeze.json"
COCO_VAL = ROOT / "PreparedData/xview_expanded_512_v1/annotations/instances_val.json"
DATASET_VALIDATION = ROOT / "AnalysisResults/xview_expanded_512_v1_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_split(yaml_path: Path, split: str) -> tuple[Path, dict]:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    base = Path(config["path"])
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    entry = config[split]
    if isinstance(entry, list):
        raise ValueError(f"Expected a single {split} path in {yaml_path}, got {entry!r}")
    return (base / entry).resolve(), config


def image_rows(directory: Path, coco: dict) -> list[dict]:
    files = sorted(p for p in directory.iterdir() if p.is_file())
    by_index = {}
    for item in coco["images"]:
        stem = Path(item["file_name"]).stem
        index = int(stem.rsplit("_", 1)[1])
        by_index[index] = item
    rows = []
    for path in files:
        index = int(path.stem.rsplit("_", 1)[1])
        record = by_index[index]
        rows.append(
            {
                "image_id": int(record["id"]),
                "yolo_file_name": path.name,
                "coco_file_name": record["file_name"],
                "source_scene": record["source_scene"],
                "source_image": record["source_image"],
                "crop_xywh": record["crop_xywh"],
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def file_record(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    initial_dir, initial_yaml_data = resolve_split(INITIAL_YAML, "val")
    v48_dir, v48_yaml_data = resolve_split(V48_YAML, "val")
    args = yaml.safe_load(INITIAL_ARGS.read_text(encoding="utf-8-sig"))
    args_yaml_path = Path(args["data"]).resolve()
    if args_yaml_path != INITIAL_YAML.resolve():
        raise RuntimeError(f"Initial args point to {args_yaml_path}, not {INITIAL_YAML}")
    if args.get("split") != "val" or not args.get("val"):
        raise RuntimeError("Initial run did not record enabled validation on split=val")

    coco = json.loads(COCO_VAL.read_text(encoding="utf-8-sig"))
    initial_images = image_rows(initial_dir, coco)
    v48_images = image_rows(v48_dir, coco)

    initial_keys = {(r["image_id"], r["sha256"]) for r in initial_images}
    v48_keys = {(r["image_id"], r["sha256"]) for r in v48_images}
    initial_scenes = sorted({r["source_scene"] for r in initial_images})
    v48_scenes = sorted({r["source_scene"] for r in v48_images})

    with INITIAL_RESULTS.open("r", encoding="utf-8-sig", newline="") as stream:
        result_rows = list(csv.DictReader(stream))
    metric = "metrics/mAP50-95(B)"
    best_value = max(float(row[metric]) for row in result_rows)
    best_rows = [row for row in result_rows if float(row[metric]) == best_value]
    if len(best_rows) != 1:
        raise RuntimeError("Initial results do not have a unique best validation mAP50-95 row")
    best_row = best_rows[0]

    checkpoint = torch.load(INITIAL_BEST, map_location="cpu", weights_only=False)
    train_metrics = checkpoint.get("train_metrics") or {}
    checkpoint_metric_matches = abs(float(train_metrics.get(metric, float("nan"))) - best_value) < 1e-12
    checkpoint_args = checkpoint.get("train_args") or {}

    normalized_initial = [
        {k: row[k] for k in ("image_id", "yolo_file_name", "source_scene", "source_image", "crop_xywh", "sha256")}
        for row in initial_images
    ]
    normalized_v48 = [
        {k: row[k] for k in ("image_id", "yolo_file_name", "source_scene", "source_image", "crop_xywh", "sha256")}
        for row in v48_images
    ]
    normalized_equal = normalized_initial == normalized_v48

    freeze = json.loads(V48_FREEZE.read_text(encoding="utf-8-sig"))
    frozen_yaml_hash = freeze["files"]["data_yaml"]["sha256"]
    actual_v48_yaml_hash = sha256_file(V48_YAML)

    payload = {
        "audit": "historical validation reuse",
        "status": "PASS_WITH_DISCLOSURE_REQUIRED",
        "case": "A_SAME_VALIDATION_SPLIT",
        "no_training_or_inference_performed": True,
        "critical_answer": (
            "Yes. The common starting checkpoint was selected on the same 375-image, "
            "nine-acquisition-scene xView validation split later used for the locally frozen "
            "primary scope-interaction endpoint."
        ),
        "counts": {
            "initial_checkpoint_validation_images": len(initial_images),
            "scope_interaction_validation_images": len(v48_images),
            "image_overlap": len(initial_keys & v48_keys),
            "image_overlap_percent": 100.0 * len(initial_keys & v48_keys) / len(initial_keys),
            "initial_acquisition_scenes": len(initial_scenes),
            "scope_interaction_acquisition_scenes": len(v48_scenes),
            "acquisition_scene_overlap": len(set(initial_scenes) & set(v48_scenes)),
        },
        "equality": {
            "resolved_validation_directory_equal": initial_dir == v48_dir,
            "ordered_image_id_path_scene_hash_manifest_equal": normalized_equal,
            "image_id_and_content_hash_sets_equal": initial_keys == v48_keys,
            "acquisition_scene_sets_equal": initial_scenes == v48_scenes,
            "exact_manifest_hash_equal": sha256_json(normalized_initial) == sha256_json(normalized_v48),
            "all_conditions_shared_common_initialization": True,
            "condition_specific_unity_selection_leakage_identified": False,
            "historical_source_model_selection_independence": False,
        },
        "manifest_hashes": {
            "initial_ordered_normalized_manifest_sha256": sha256_json(normalized_initial),
            "scope_interaction_ordered_normalized_manifest_sha256": sha256_json(normalized_v48),
            "initial_image_id_content_set_sha256": sha256_json(sorted(initial_keys)),
            "scope_interaction_image_id_content_set_sha256": sha256_json(sorted(v48_keys)),
            "source_scene_list_sha256": sha256_json(initial_scenes),
            "coco_validation_manifest_sha256": sha256_file(COCO_VAL),
        },
        "resolved_inputs": {
            "initial_validation_directory": initial_dir.relative_to(ROOT).as_posix(),
            "scope_interaction_validation_directory": v48_dir.relative_to(ROOT).as_posix(),
            "initial_yaml_val_entry": initial_yaml_data["val"],
            "scope_interaction_yaml_val_entry": v48_yaml_data["val"],
            "source_scenes": initial_scenes,
        },
        "initial_best_checkpoint_selection": {
            "checkpoint": file_record(INITIAL_BEST),
            "recorded_data_yaml": str(args["data"]),
            "recorded_split": args["split"],
            "recorded_validation_enabled": bool(args["val"]),
            "configured_epochs": int(args["epochs"]),
            "configured_patience": int(args["patience"]),
            "completed_result_rows": len(result_rows),
            "unique_best_results_epoch": int(float(best_row["epoch"])),
            "unique_best_validation_map50_95": best_value,
            "checkpoint_embedded_metric": train_metrics.get(metric),
            "checkpoint_metric_matches_unique_best_row": checkpoint_metric_matches,
            "checkpoint_embedded_data_yaml": checkpoint_args.get("data"),
            "checkpoint_embedded_split": checkpoint_args.get("split"),
            "checkpoint_epoch_field_note": (
                "The serialized release checkpoint records epoch=-1 after Ultralytics finalization; "
                "its embedded train_metrics exactly match the unique results.csv maximum at epoch 27."
            ),
        },
        "v48_freeze_check": {
            "frozen_data_yaml_sha256": frozen_yaml_hash,
            "actual_data_yaml_sha256": actual_v48_yaml_hash,
            "match": frozen_yaml_hash == actual_v48_yaml_hash,
            "evaluator_uses_data_yaml_and_split_val": True,
        },
        "source_files": [
            file_record(p)
            for p in (
                INITIAL_YAML,
                INITIAL_ARGS,
                INITIAL_RESULTS,
                INITIAL_BEST,
                V48_YAML,
                V48_EVALUATOR,
                V48_FREEZE,
                COCO_VAL,
                DATASET_VALIDATION,
            )
        ],
        "publication_consequence": {
            "required_label": "locally frozen primary scope-interaction endpoint on the historical xView validation split",
            "prohibited_independence_claims": [
                "untouched confirmation set",
                "fully independent confirmatory validation",
                "independent validation cohort",
                "prospectively untouched validation data",
            ],
            "limitation": (
                "The common starting checkpoint had historically been selected on this same xView validation split. "
                "Scope-specific settings were selected using Duplicate-only inner development and every condition "
                "shared the initialization, but the endpoint was not independent of historical source-model selection."
            ),
        },
    }

    if not all(
        [
            payload["equality"]["resolved_validation_directory_equal"],
            payload["equality"]["ordered_image_id_path_scene_hash_manifest_equal"],
            payload["equality"]["image_id_and_content_hash_sets_equal"],
            payload["equality"]["acquisition_scene_sets_equal"],
            payload["v48_freeze_check"]["match"],
            checkpoint_metric_matches,
        ]
    ):
        raise RuntimeError("Validation reuse evidence was internally inconsistent")

    (OUT / "validation_reuse_audit_v39.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    c = payload["counts"]
    h = payload["manifest_hashes"]
    lines = [
        "# Validation Reuse Audit v39",
        "",
        "## Decision",
        "",
        "**Case A — same validation split. Disclosure is required.** The common xView-trained YOLO26n `best.pt` "
        "was selected using the same 375-image, nine-acquisition-scene xView validation split later used for the "
        "locally frozen primary scope-interaction endpoint.",
        "",
        "This is historical source-model selection reuse, not condition-specific Unity selection leakage. All "
        "scope-interaction conditions shared the same initialization, and scope-specific settings were selected "
        "using Duplicate-only inner development without Unity, public-validation, test, HRSC2016-MS, or DIOR outcomes.",
        "",
        "## Exact comparison",
        "",
        "| Check | Initial checkpoint selection | Scope-interaction endpoint | Result |",
        "|---|---:|---:|---|",
        f"| Validation images | {c['initial_checkpoint_validation_images']} | {c['scope_interaction_validation_images']} | {c['image_overlap']}/{c['initial_checkpoint_validation_images']} overlap ({c['image_overlap_percent']:.1f}%) |",
        f"| Acquisition scenes | {c['initial_acquisition_scenes']} | {c['scope_interaction_acquisition_scenes']} | {c['acquisition_scene_overlap']}/{c['initial_acquisition_scenes']} overlap |",
        f"| Resolved directory | `{payload['resolved_inputs']['initial_validation_directory']}` | `{payload['resolved_inputs']['scope_interaction_validation_directory']}` | exact path equality |",
        f"| Ordered image-ID/path/scene/content manifest | `{h['initial_ordered_normalized_manifest_sha256']}` | `{h['scope_interaction_ordered_normalized_manifest_sha256']}` | exact SHA-256 equality |",
        f"| COCO validation manifest | `{h['coco_validation_manifest_sha256']}` | `{h['coco_validation_manifest_sha256']}` | same source manifest |",
        "",
        "The two YAML files differ in their training entries but both resolve `val: real/images/val` under the same "
        "dataset root. Every one of the 375 validation images matched by image ID and SHA-256 content hash. The "
        "COCO manifest mapped them to the same nine `source_scene` identifiers.",
        "",
        "## Initial `best.pt` evidence",
        "",
        f"The initial run recorded validation enabled on `split=val`, 50 configured epochs, patience 20, and {len(result_rows)} "
        f"completed rows. Validation mAP50-95 had a unique maximum of {best_value:.5f} at epoch "
        f"{int(float(best_row['epoch']))}. The finalized `best.pt` embedded validation metric matches this row exactly. "
        "The checkpoint's serialized `epoch=-1` field is an Ultralytics finalization artifact and is not used to infer selection.",
        "",
        "## Scientific interpretation",
        "",
        "The primary interaction remained locally frozen before its one-time evaluation, and the later condition-specific "
        "training/settings did not use Unity or public validation outcomes for selection. Nevertheless, the xView "
        "validation endpoint was not independent of historical selection of the common source model. Publication text "
        "must therefore call it a **locally frozen primary scope-interaction endpoint on the historical xView validation "
        "split**, not an untouched or fully independent confirmatory validation cohort.",
        "",
        "Post-lock xView test, HRSC2016-MS, and DIOR results provide important transport checks, but they are not relabeled "
        "as newly confirmatory evidence.",
        "",
        "## Integrity status",
        "",
        "PASS WITH DISCLOSURE REQUIRED. No training or inference was performed for this audit.",
        "",
    ]
    (OUT / "validation_reuse_audit_v39.md").write_text("\n".join(lines), encoding="utf-8")

    pre = [
        "# v39 Pre-Revision Audit",
        "",
        "## Gate decision",
        "",
        "Revision may proceed only under **Case A — historical validation reuse**. The 375 validation images, all nine "
        "acquisition scenes, ordered normalized manifest, and image-content hashes match exactly between initial "
        "`best.pt` selection and the later scope-interaction endpoint.",
        "",
        "## Mandatory scientific corrections",
        "",
        "1. Replace claims of an untouched, independent, or prospectively pristine validation cohort with a locally "
        "frozen primary scope-interaction endpoint on the historical xView validation split.",
        "2. State that all four scope/condition cells shared the common initialization and that scope-specific settings "
        "were selected using Duplicate-only inner development; do not misclassify historical reuse as condition-specific leakage.",
        "3. Add a limitation that the endpoint was not independent of historical source-model selection.",
        "4. Preserve the validation-first decision lock and the later xView test/HRSC2016-MS/DIOR ordering without "
        "inflating the latter into new confirmatory endpoints.",
        "5. Remove internal experiment-version language from publication copy, align causal wording, and keep all negative evidence.",
        "",
        "## Submission and reproducibility gates",
        "",
        "IEEE Access currently uses single-anonymized review, so author names must appear in the submitted source and PDF. "
        "Short biographies are required for all authors. Because author metadata were not supplied, the submission copy "
        "must remain explicitly pending metadata rather than being represented as ready to upload.",
        "",
        "The public package omits raw predictions, checkpoints, images, and embeddings. Unless lawful redistribution of "
        "the necessary per-image predictions is added, scene-bootstrap results are auditable but not fully independently recomputable.",
        "",
        "## Prohibited actions",
        "",
        "No new efficacy training, inference, hyperparameter search, invented archive identifier, fabricated author metadata, "
        "or result-improving analytical change is authorized.",
        "",
        "## Status",
        "",
        "PASS TO REVISE WITH MANDATORY DISCLOSURES.",
        "",
    ]
    (OUT / "v39_pre_revision_audit.md").write_text("\n".join(pre), encoding="utf-8")
    print(json.dumps({"case": payload["case"], "counts": c, "manifest_sha256": h["initial_ordered_normalized_manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
