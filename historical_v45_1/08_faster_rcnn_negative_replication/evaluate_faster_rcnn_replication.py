#!/usr/bin/env python3
"""Evaluate the frozen Faster R-CNN replication on scene-disjoint xView validation."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from train_fasterrcnn_replication_v16 import YoloDetectionDataset, collate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/fasterrcnn_replication_v16.json"
RESULTS = ROOT / "AnalysisResults/expanded_v1/fasterrcnn_replication_v16"
RUNS = ROOT / "runs/fasterrcnn_replication_v16"
CLASS_NAMES = ["small_vehicle", "bus", "truck", "excavator", "maritime_vessel"]
T_CRITICAL = 2.131846786326649


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint: Path):
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, min_size=640, max_size=640,
        box_score_thresh=0.001, box_nms_thresh=0.5, box_detections_per_img=300,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, 6)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != 10:
        raise RuntimeError(f"Not an exact epoch-10 checkpoint: {checkpoint}")
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def build_ground_truth(dataset: YoloDetectionDataset) -> COCO:
    images, annotations, annotation_id = [], [], 1
    for image_id in range(len(dataset)):
        image, target = dataset[image_id]
        images.append({"id": image_id, "width": int(image.shape[2]), "height": int(image.shape[1]), "file_name": dataset.images[image_id].name})
        for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
            x1, y1, x2, y2 = map(float, box)
            width, height = x2 - x1, y2 - y1
            annotations.append({
                "id": annotation_id, "image_id": image_id, "category_id": int(label),
                "bbox": [x1, y1, width, height], "area": width * height, "iscrowd": 0,
            })
            annotation_id += 1
    coco = COCO()
    coco.dataset = {
        "images": images, "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)],
        "info": {"description": "scene-disjoint xView validation v16"}, "licenses": [],
    }
    coco.createIndex()
    return coco


def coco_metrics(coco_gt: COCO, predictions: list[dict]) -> tuple[dict, str]:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(coco_gt.getImgIds())
        evaluator.params.catIds = [1, 2, 3, 4, 5]
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"][:, :, :, 0, 2]
    per_class = {}
    for class_index, name in enumerate(CLASS_NAMES):
        values = precision[:, :, class_index]
        per_class[name] = float(values[values > -1].mean())
    return {
        "aggregate_map50_95": float(evaluator.stats[0]),
        "aggregate_ap50": float(evaluator.stats[1]),
        "aggregate_ap75": float(evaluator.stats[2]),
        "per_class_ap50_95": per_class,
        "vessel_ap50_95": per_class["maritime_vessel"],
    }, captured.getvalue()


def paired_summary(left: list[float], right: list[float]) -> dict:
    differences = [a - b for a, b in zip(left, right)]
    mean, sd = statistics.mean(differences), statistics.stdev(differences)
    return {
        "differences": differences, "mean_difference": mean, "sample_sd": sd,
        "lower_one_sided_95": mean - T_CRITICAL * sd / math.sqrt(len(differences)),
        "t_0.95_df4": T_CRITICAL,
    }


def main() -> int:
    if RESULTS.exists():
        raise FileExistsError(f"Refusing to overwrite {RESULTS}")
    study = json.loads(CONFIG.read_text(encoding="utf-8"))
    checkpoints = []
    for condition in study["conditions"]:
        for seed in study["seeds"]:
            path = RUNS / f"{condition}_s{seed}/last.pt"
            if not path.is_file():
                raise RuntimeError(f"Evaluation locked until checkpoint exists: {path}")
            checkpoints.append((condition, seed, path))
    validation_dir = ROOT / study["data_root"] / study["validation_images"]
    dataset = YoloDetectionDataset([validation_dir])
    if len(dataset) != 375:
        raise RuntimeError(f"Expected 375 validation images, got {len(dataset)}")
    coco_gt = build_ground_truth(dataset)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)
    RESULTS.mkdir(parents=True)
    device = torch.device("cuda:0")
    raw_results = []
    for ordinal, (condition, seed, checkpoint) in enumerate(checkpoints, start=1):
        print(f"V16 EVAL {ordinal}/15 {condition} seed={seed}", flush=True)
        model, checkpoint_payload = build_model(checkpoint)
        model.to(device).eval()
        predictions = []
        image_cursor = 0
        with torch.inference_mode():
            for images, _targets in loader:
                outputs = model([image.to(device, non_blocking=True) for image in images])
                for output in outputs:
                    boxes = output["boxes"].detach().cpu().numpy()
                    scores = output["scores"].detach().cpu().numpy()
                    labels = output["labels"].detach().cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = map(float, box)
                        predictions.append({
                            "image_id": image_cursor, "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
                        })
                    image_cursor += 1
        if image_cursor != len(dataset):
            raise RuntimeError("Inference image count mismatch")
        metrics, evaluator_log = coco_metrics(coco_gt, predictions)
        prediction_path = RESULTS / f"{condition}_s{seed}.predictions.json.gz"
        with gzip.open(prediction_path, "wt", encoding="utf-8", compresslevel=9) as stream:
            json.dump(predictions, stream, separators=(",", ":"))
        result = {
            "condition": condition, "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "initial_trainable_sha256": checkpoint_payload["initial_trainable_sha256"],
            "trainable_parameters": checkpoint_payload["trainable_parameters"],
            "frozen_parameters": checkpoint_payload["frozen_parameters"],
            "prediction_count": len(predictions),
            "raw_predictions": str(prediction_path), "raw_predictions_sha256": sha256_file(prediction_path),
            **metrics,
        }
        (RESULTS / f"{condition}_s{seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (RESULTS / f"{condition}_s{seed}.log").write_text(evaluator_log, encoding="utf-8")
        raw_results.append(result)
        del model
        torch.cuda.empty_cache()

    values = {
        condition: [next(item["vessel_ap50_95"] for item in raw_results if item["condition"] == condition and item["seed"] == seed) for seed in study["seeds"]]
        for condition in study["conditions"]
    }
    efficacy = paired_summary(values["unity_medium150"], values["duplicate"])
    efficacy_pass = efficacy["lower_one_sided_95"] > 0.0
    real = paired_summary(values["unity_medium150"], values["realcutout_medium150"])
    real_pass = efficacy_pass and real["lower_one_sided_95"] > -0.005
    initialization_audit = {
        str(seed): len({item["initial_trainable_sha256"] for item in raw_results if item["seed"] == seed}) == 1
        for seed in study["seeds"]
    }
    if not all(initialization_audit.values()):
        raise RuntimeError(f"Paired initialization audit failed: {initialization_audit}")
    summary = {
        "study": "independent Faster R-CNN replication v16",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(CONFIG),
        "initialization_hash_match_within_seed": initialization_audit,
        "vessel_ap50_95_values": values,
        "condition_means": {condition: statistics.mean(items) for condition, items in values.items()},
        "unity_efficacy": {**efficacy, "pass": efficacy_pass},
        "realcutout_noninferiority": {**real, "margin": -0.005, "inferentially_tested": efficacy_pass, "pass": real_pass},
        "external_evaluation_unlocked": efficacy_pass,
        "raw_results": raw_results,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Independent Faster R-CNN replication v16", "", "## Vessel AP50-95", "",
        "| Condition | Per-seed AP | Mean |", "|---|---|---:|",
    ]
    for condition, items in values.items():
        lines.append(f"| {condition} | {', '.join(f'{value:.6f}' for value in items)} | {statistics.mean(items):.6f} |")
    lines += [
        "", "## Frozen decisions", "",
        f"- Unity - Duplicate: mean {efficacy['mean_difference']:+.6f}, one-sided 95% LCB {efficacy['lower_one_sided_95']:+.6f}: **{'PASS' if efficacy_pass else 'FAIL'}**.",
        f"- Unity - RealCutout, margin -0.005: mean {real['mean_difference']:+.6f}, one-sided 95% LCB {real['lower_one_sided_95']:+.6f}: **{'PASS' if real_pass else 'FAIL/NOT TESTED'}**.",
        f"- External HRSC/DIOR evaluation: **{'UNLOCKED' if efficacy_pass else 'LOCKED'}**.", "",
        "The detector is a two-stage proposal-based Faster R-CNN with a frozen MobileNetV3/FPN backbone and trainable RPN/ROI heads. All paired initialization hashes matched.", "",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"condition_means": summary["condition_means"], "unity_efficacy": summary["unity_efficacy"], "external_evaluation_unlocked": efficacy_pass}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
