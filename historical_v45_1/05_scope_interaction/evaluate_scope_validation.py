#!/usr/bin/env python3
"""One-time frozen xView-validation evaluation for v48 scope fairness."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import scipy.stats
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "AnalysisResults/expanded_v1/adaptation_scope_fairness_v48"
LEDGER = STUDY / "confirmation/training_ledger.json"
CONFIRMATION_FREEZE = STUDY / "confirmation_freeze.json"
VALIDATION_FREEZE = STUDY / "validation_freeze.json"
OUTPUT = STUDY / "validation"
RUNS = ROOT / "runs/adaptation_scope_fairness_v48_validation"
DATA = ROOT / "config/yolo_xview_expanded_real_v1.yaml"
SEEDS = list(range(20260723, 20260733))
SCOPES = ("head_only", "full_network")
CONDITIONS = ("duplicate", "unity_medium150")
FAMILIES = ("aggregate", "vessel")
METRICS = ("ap50_95", "ap50", "ap75")
VESSEL_CLASS = 4
CLASS_NAMES = ["small_vehicle", "bus", "truck", "excavator", "maritime_vessel"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def checkpoint_for(job: dict) -> Path:
    if job["status"] == "reused_exact":
        base = ROOT / "runs/head_only_factorial_v15"
    else:
        base = ROOT / "runs/adaptation_scope_fairness_v48_confirmation"
    return base / job["run_name"] / "weights/last.pt"


def exact_epoch20(checkpoint: Path) -> bool:
    results = checkpoint.parents[1] / "results.csv"
    if not checkpoint.is_file() or not results.is_file():
        return False
    with results.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20:
        return False
    try:
        return int(float(rows[-1]["epoch"])) == 20
    except (KeyError, TypeError, ValueError):
        return False


def audit_freeze() -> tuple[list[dict], dict]:
    frozen = json.loads(VALIDATION_FREEZE.read_text(encoding="utf-8"))
    targets = {
        "evaluator": ROOT / "tools/evaluate_v48_scope_validation.py",
        "runner": ROOT / "tools/run_v48_scope_validation_task.ps1",
        "data_yaml": DATA,
        "training_ledger": LEDGER,
        "confirmation_freeze": CONFIRMATION_FREEZE,
    }
    mismatches = []
    for key, path in targets.items():
        expected = frozen["files"][key]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append({"key": key, "expected": expected, "actual": actual})
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    jobs = ledger["jobs"]
    expected_keys = {(scope, condition, seed) for scope in SCOPES for condition in CONDITIONS for seed in SEEDS}
    actual_keys = {(j["scope"], j["condition"], int(j["seed"])) for j in jobs}
    if ledger.get("status") != "PASS" or len(jobs) != 40 or actual_keys != expected_keys:
        raise RuntimeError("Training ledger is not the frozen complete 2x2x10 design")
    frozen_checkpoints = frozen["checkpoints"]
    for job in jobs:
        checkpoint = checkpoint_for(job)
        key = f"{job['scope']}|{job['condition']}|{job['seed']}"
        actual = sha256_file(checkpoint) if checkpoint.is_file() else None
        expected = frozen_checkpoints[key]
        ledger_hash = job["last_checkpoint_sha256"]
        if not exact_epoch20(checkpoint) or actual != expected or actual != ledger_hash:
            mismatches.append({"key": key, "expected": expected, "ledger": ledger_hash, "actual": actual})
    if mismatches:
        raise RuntimeError(f"Validation freeze audit failed: {mismatches}")
    return jobs, frozen


def sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    cumulative = sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2 * cumulative)


def summarize(values: list[float], seeds: list[int]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    sem = sd / math.sqrt(len(values))
    critical = float(scipy.stats.t.ppf(0.975, len(values) - 1))
    if sem:
        t_stat = mean / sem
        p_value = float(2 * scipy.stats.t.sf(abs(t_stat), len(values) - 1))
    else:
        t_stat = math.inf if mean else 0.0
        p_value = 0.0 if mean else 1.0
    return {
        "seeds": seeds,
        "values": values,
        "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
        "mean": mean,
        "sample_sd": sd,
        "two_sided_95_ci": [mean - critical * sem, mean + critical * sem],
        "paired_t_statistic": t_stat,
        "paired_t_two_sided_p": p_value,
        "cohen_dz": mean / sd if sd else None,
        "hedges_gz": (mean / sd) * (1 - 3 / (4 * len(values) - 5)) if sd else None,
        "positive_seeds": sum(value > 0 for value in values),
        "negative_seeds": sum(value < 0 for value in values),
        "zero_seeds": sum(value == 0 for value in values),
        "exact_two_sided_sign_p": sign_p(values),
    }


def evaluate_job(job: dict) -> dict:
    scope, condition, seed = job["scope"], job["condition"], int(job["seed"])
    checkpoint = checkpoint_for(job)
    cell = f"{scope}_{condition}_s{seed}"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = YOLO(str(checkpoint)).val(
            data=str(DATA), split="val", imgsz=640, batch=8, device="0", workers=0,
            conf=0.001, iou=0.7, max_det=300, plots=False, verbose=False,
            save_json=False, project=str(RUNS), name=cell, exist_ok=False,
        )
    indices = [int(value) for value in result.box.ap_class_index]
    if VESSEL_CLASS not in indices:
        raise RuntimeError(f"Vessel class missing from validation metrics for {cell}")
    vessel_position = indices.index(VESSEL_CLASS)
    all_ap = result.box.all_ap
    per_class = {}
    for position, class_index in enumerate(indices):
        per_class[CLASS_NAMES[class_index]] = {
            "ap50_95": float(result.box.maps[class_index]),
            "ap50": float(all_ap[position, 0]),
            "ap75": float(all_ap[position, 5]),
        }
    payload = {
        "scope": scope,
        "condition": condition,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "aggregate": {
            "ap50_95": float(result.box.map),
            "ap50": float(result.box.map50),
            "ap75": float(result.box.map75),
        },
        "vessel": {
            "ap50_95": float(result.box.maps[VESSEL_CLASS]),
            "ap50": float(all_ap[vessel_position, 0]),
            "ap75": float(all_ap[vessel_position, 5]),
        },
        "per_class": per_class,
        "ap_class_index": indices,
        "speed_ms_per_image": {key: float(value) for key, value in result.speed.items()},
        "prediction_artifacts_saved": False,
    }
    write_json_exclusive(OUTPUT / f"{cell}.json", payload)
    (OUTPUT / f"{cell}.log").write_text(stdout.getvalue() + stderr.getvalue(), encoding="utf-8")
    return payload


def analyze(rows: list[dict]) -> dict:
    lookup = {(row["scope"], row["condition"], int(row["seed"])): row for row in rows}
    output = {}
    for family in FAMILIES:
        family_output = {}
        for metric in METRICS:
            cells = {
                f"{scope}_{condition}": [lookup[(scope, condition, seed)][family][metric] for seed in SEEDS]
                for scope in SCOPES for condition in CONDITIONS
            }
            head = [u - d for u, d in zip(cells["head_only_unity_medium150"], cells["head_only_duplicate"])]
            full = [u - d for u, d in zip(cells["full_network_unity_medium150"], cells["full_network_duplicate"])]
            did = [h - f for h, f in zip(head, full)]
            family_output[metric] = {
                "cell_values": cells,
                "cell_means": {cell: statistics.mean(values) for cell, values in cells.items()},
                "unity_effect_head": summarize(head, SEEDS),
                "unity_effect_full": summarize(full, SEEDS),
                "head_minus_full_difference_in_differences": summarize(did, SEEDS),
            }
        output[family] = family_output
    return output


def classify(interval: list[float]) -> str:
    if interval[0] > 0:
        return "positive"
    if interval[1] < 0:
        return "negative"
    return "inconclusive"


def main() -> int:
    if OUTPUT.exists() or RUNS.exists():
        raise FileExistsError("Refusing to overwrite an existing v48 validation evaluation")
    jobs, frozen = audit_freeze()
    OUTPUT.mkdir(parents=True)
    started = {
        "study": "v48 adaptation-scope fairness xView validation",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha256_file(VALIDATION_FREEZE),
        "jobs": 40,
        "exactly_once_guard": True,
    }
    write_json_exclusive(OUTPUT / "evaluation_started.json", started)
    rows = []
    ordered = sorted(jobs, key=lambda j: (SCOPES.index(j["scope"]), CONDITIONS.index(j["condition"]), int(j["seed"])))
    for ordinal, job in enumerate(ordered, start=1):
        print(f"V48 VALIDATION {ordinal}/40 {job['scope']} {job['condition']} seed={job['seed']}", flush=True)
        rows.append(evaluate_job(job))
    analysis = analyze(rows)
    vessel_primary = analysis["vessel"]["ap50_95"]["head_minus_full_difference_in_differences"]
    aggregate_primary = analysis["aggregate"]["ap50_95"]["head_minus_full_difference_in_differences"]
    decision_lock = {
        "study": "v48 adaptation-scope fairness validation decision lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "validation_freeze_sha256": sha256_file(VALIDATION_FREEZE),
        "external_or_test_evaluation_opened": False,
        "primary_endpoint": "vessel AP50-95 head-minus-full difference-in-differences",
        "primary_decision": classify(vessel_primary["two_sided_95_ci"]),
        "primary": vessel_primary,
        "aggregate_ap50_95_decision": classify(aggregate_primary["two_sided_95_ci"]),
        "aggregate_ap50_95": aggregate_primary,
        "interpretation": "positive/negative only when the two-sided 95% CI excludes zero; otherwise inconclusive",
    }
    write_json_exclusive(OUTPUT / "decision_lock.json", decision_lock)
    nominal = {
        "images_per_epoch": 3000,
        "epochs": 20,
        "nominal_image_exposures_per_run": 60000,
        "batch_size": 16,
        "nominal_minibatches_per_epoch": 188,
        "nominal_minibatches_per_run": 3760,
        "optimizer": "AdamW",
        "nominal_gradient_accumulation_target": 4,
        "optimizer_step_note": "Exact AdamW step counts and extra partial-epoch exposure from crash/restart boundaries were not logged; 3760 is the auditable minibatch count, not an asserted optimizer.step count.",
        "condition_exposure_matched": True,
    }
    summary = {
        "study": "v48 adaptation-scope fairness xView validation",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "validation_freeze": frozen,
        "analysis": analysis,
        "decision_lock": decision_lock,
        "optimizer_update_and_exposure": nominal,
        "raw_results": rows,
        "prediction_artifacts_saved": False,
    }
    write_json_exclusive(OUTPUT / "summary.json", summary)
    lines = [
        "# v48 Adaptation-Scope Fairness — Frozen xView Validation",
        "",
        f"Primary vessel AP50-95 DiD: {vessel_primary['mean']:+.6f} "
        f"(two-sided 95% CI [{vessel_primary['two_sided_95_ci'][0]:+.6f}, {vessel_primary['two_sided_95_ci'][1]:+.6f}]); "
        f"decision **{decision_lock['primary_decision']}**.",
        f"Aggregate AP50-95 DiD: {aggregate_primary['mean']:+.6f} "
        f"(two-sided 95% CI [{aggregate_primary['two_sided_95_ci'][0]:+.6f}, {aggregate_primary['two_sided_95_ci'][1]:+.6f}]); "
        f"decision **{decision_lock['aggregate_ap50_95_decision']}**.",
        "",
        "The validation decision was locked before xView test or external-dataset evaluation. No prediction artifact was saved.",
        "Nominal exposure is 60,000 images and 3,760 minibatches per run; exact optimizer.step counts and crash-induced partial-epoch exposure were not logged and are not fabricated.",
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"primary": decision_lock["primary_decision"], "vessel": vessel_primary, "aggregate": aggregate_primary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
