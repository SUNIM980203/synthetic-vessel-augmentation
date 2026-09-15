#!/usr/bin/env python3
"""Evaluate all accepted YOLO26s cells in the frozen domain order."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import time
import traceback
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops

import yolo26s_factorial_common as common


EVALUATOR_VERSION = "yolo26s_scope_lr_factorial_ap_evaluator_v1"
EVAL_RUNS = common.METRICS_DIR / "ultralytics_eval_runs"


class ExactJSONDetectionValidator(DetectionValidator):
    """Preserve native Python float serialization instead of Ultralytics rounding."""

    def pred_to_json(self, predn: dict, pbatch: dict) -> None:
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])
        box[:, :2] -= box[:, 2:] / 2
        for bbox, score, category in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            mapped_class = int(self.class_map[int(category)])
            self.jdict.append({
                "image_id": image_id,
                "file_name": path.name,
                "category_id": mapped_class,
                "class": mapped_class,
                "bbox": [float(value) for value in bbox],
                "score": float(score),
                "confidence": float(score),
            })


def load_ledger(jobs: list[dict]) -> dict:
    if common.EVALUATION_LEDGER.is_file():
        return common.load_json(common.EVALUATION_LEDGER)
    ledger = {
        "study": "fresh YOLO26s model-size scope x LR factorial evaluation",
        "created_utc": common.utc_now(),
        "updated_utc": common.utc_now(),
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_sha256": common.sha256_file(Path(__file__)),
        "domain_order": list(common.DOMAINS),
        "jobs": jobs,
    }
    common.atomic_json(common.EVALUATION_LEDGER, ledger)
    return ledger


def save_ledger(ledger: dict) -> None:
    ledger["updated_utc"] = common.utc_now()
    ledger["completed_jobs"] = sum(job["status"] == "COMPLETE" for job in ledger["jobs"])
    common.atomic_json(common.EVALUATION_LEDGER, ledger)


def result_paths(domain: str, cell: str) -> tuple[Path, Path, Path]:
    stem = f"{domain}__{cell}"
    return (
        common.METRICS_DIR / f"{stem}.metrics.json",
        common.PREDICTIONS_DIR / f"{stem}.predictions.json.gz",
        common.METRICS_DIR / f"{stem}.evaluation.log",
    )


def prediction_payload(raw_path: Path, metadata: dict) -> dict:
    predictions = json.loads(raw_path.read_text(encoding="utf-8"))
    required = {"image_id", "category_id", "class", "bbox", "score", "confidence"}
    for index, row in enumerate(predictions):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"prediction row {index} missing {sorted(missing)}")
    return {"metadata": metadata, "predictions": predictions}


def evaluate(job: dict, evaluator_sha: str) -> None:
    metric_path, prediction_path, log_path = result_paths(job["domain"], job["cell_name"])
    if metric_path.exists() or prediction_path.exists() or log_path.exists():
        raise FileExistsError(f"refusing to overwrite partial evaluation outputs for {job['domain']} {job['cell_name']}")
    checkpoint = Path(job["checkpoint"])
    expected_checkpoint_sha = job["checkpoint_sha256"]
    if common.sha256_file(checkpoint) != expected_checkpoint_sha:
        raise RuntimeError("accepted checkpoint hash changed before evaluation")
    eval_name = f"{job['domain']}__{job['cell_name']}"
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        metrics = YOLO(str(checkpoint)).val(
            validator=ExactJSONDetectionValidator,
            data=str(common.DOMAINS[job["domain"]].resolve()),
            split="test",
            imgsz=640,
            batch=8,
            device="0",
            workers=0,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=False,
            verbose=False,
            save_json=True,
            project=str(EVAL_RUNS.resolve()),
            name=eval_name,
            exist_ok=False,
        )
    indices = [int(value) for value in metrics.box.ap_class_index]
    if common.VESSEL_CLASS not in indices:
        raise RuntimeError(f"vessel class absent from evaluated ground truth: {eval_name}")
    vessel_position = indices.index(common.VESSEL_CLASS)
    all_ap = metrics.box.all_ap
    raw_prediction = Path(metrics.save_dir) / "predictions.json"
    if not raw_prediction.is_file():
        raise RuntimeError("Ultralytics prediction JSON was not created")
    metadata = {
        "schema": "yolo26s_scope_lr_factorial_prediction_archive_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_sha256": evaluator_sha,
        "created_utc": common.utc_now(),
        "domain": job["domain"],
        "seed": int(job["seed"]),
        "condition": job["condition"],
        "scope": job["scope"],
        "model_size": "YOLO26s",
        "learning_rate": float(job["learning_rate"]),
        "learning_rate_label": job["learning_rate_label"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha,
        "prediction_numeric_values_rounded": False,
        "prediction_fields": ["image_id", "class", "bbox", "confidence"],
    }
    wrapper = prediction_payload(raw_prediction, metadata)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_name(prediction_path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(wrapper, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(prediction_path)
    prediction_sha = common.sha256_file(prediction_path)
    payload = {
        **metadata,
        "data_yaml": str(common.DOMAINS[job["domain"]]),
        "data_yaml_sha256": common.sha256_file(common.DOMAINS[job["domain"]]),
        "aggregate": {
            "ap50_95": float(metrics.box.map),
            "ap50": float(metrics.box.map50),
            "ap75": float(metrics.box.map75),
        },
        "vessel": {
            "ap50_95": float(metrics.box.maps[common.VESSEL_CLASS]),
            "ap50": float(all_ap[vessel_position, 0]),
            "ap75": float(all_ap[vessel_position, 5]),
        },
        "ap_class_index": indices,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "duration_seconds": time.perf_counter() - started,
        "prediction_archive": str(prediction_path),
        "prediction_archive_sha256": prediction_sha,
        "prediction_count": len(wrapper["predictions"]),
    }
    common.write_json_exclusive(metric_path, payload)
    log_path.write_text(stdout.getvalue() + stderr.getvalue(), encoding="utf-8")
    raw_prediction.unlink()


def main() -> int:
    common.assert_protocol_locked()
    training = common.load_json(common.TRAINING_LEDGER)
    if training.get("status") != "PASS" or training.get("accepted_cells") != 80:
        raise RuntimeError("evaluation is locked until all 80 fresh cells are accepted")
    evaluator_sha = common.sha256_file(Path(__file__))
    frozen_manifest = common.load_json(common.ANALYSIS_MANIFEST)
    if frozen_manifest["roles"]["ap_evaluator"]["sha256"] != evaluator_sha:
        raise RuntimeError("AP evaluator differs from the pre-run analysis manifest")
    cells = []
    for item in sorted(training["jobs"], key=lambda row: int(row["ordinal"])):
        checkpoint = Path(item["accepted_checkpoint"])
        cells.append({
            "cell_name": item["cell_name"],
            "condition": item["condition"],
            "scope": item["scope"],
            "learning_rate": float(item["learning_rate"]),
            "learning_rate_label": common.LR_LABELS[float(item["learning_rate"])],
            "seed": int(item["seed"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": common.sha256_file(checkpoint),
        })
    jobs = [{**cell, "domain": domain, "status": "PENDING"} for domain in common.DOMAINS for cell in cells]
    ledger = load_ledger(jobs)
    for ordinal, job in enumerate(ledger["jobs"], 1):
        if job["status"] == "COMPLETE":
            metric_path, prediction_path, _ = result_paths(job["domain"], job["cell_name"])
            if not metric_path.is_file() or not prediction_path.is_file():
                raise RuntimeError(f"completed evaluation artifact missing: {job}")
            continue
        job["status"] = "RUNNING"
        job["started_utc"] = common.utc_now()
        save_ledger(ledger)
        print(f"YOLO26S FACTORIAL EVAL {ordinal}/240 {job['domain']} {job['cell_name']}", flush=True)
        try:
            evaluate(job, evaluator_sha)
            metric_path, prediction_path, _ = result_paths(job["domain"], job["cell_name"])
            job.update({
                "status": "COMPLETE",
                "completed_utc": common.utc_now(),
                "metrics": str(metric_path),
                "metrics_sha256": common.sha256_file(metric_path),
                "predictions": str(prediction_path),
                "predictions_sha256": common.sha256_file(prediction_path),
            })
        except Exception as error:
            job.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            save_ledger(ledger)
            raise
        save_ledger(ledger)
    ledger["status"] = "PASS" if ledger["completed_jobs"] == 240 else "FAIL"
    ledger["completed_utc"] = common.utc_now()
    save_ledger(ledger)
    print(f"YOLO26S FACTORIAL EVALUATION {ledger['status']}: {ledger['completed_jobs']}/240", flush=True)
    return 0 if ledger["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
