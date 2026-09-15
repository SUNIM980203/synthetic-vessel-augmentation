#!/usr/bin/env python3
"""Post-lock xView test and external evaluation for v48 scope fairness."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

import evaluate_v48_scope_validation as base


ROOT = base.ROOT
STUDY = base.STUDY
FREEZE = STUDY / "external_evaluation_freeze.json"
DECISION_LOCK = STUDY / "validation/decision_lock.json"
OUTPUT = STUDY / "external_evaluation"
RUNS = ROOT / "runs/adaptation_scope_fairness_v48_external_evaluation"
DOMAINS = {
    "xview_test": ROOT / "config/yolo_xview_expanded_real_v1.yaml",
    "hrsc2016_ms": ROOT / "config/yolo_hrsc2016_ms_v13.yaml",
    "dior_public_mirror": ROOT / "config/yolo_dior_v14.yaml",
}


def write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def audit() -> tuple[list[dict], dict, dict]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    decision = json.loads(DECISION_LOCK.read_text(encoding="utf-8"))
    if decision.get("external_or_test_evaluation_opened") is not False:
        raise RuntimeError("Validation decision lock is missing or was not closed before external evaluation")
    files = {
        "evaluator": ROOT / "tools/evaluate_v48_scope_external.py",
        "runner": ROOT / "tools/run_v48_scope_external_task.ps1",
        "validation_decision_lock": DECISION_LOCK,
        "validation_freeze": STUDY / "validation_freeze.json",
        "training_ledger": base.LEDGER,
        "confirmation_freeze": base.CONFIRMATION_FREEZE,
        "xview_test_yaml": DOMAINS["xview_test"],
        "hrsc2016_ms_yaml": DOMAINS["hrsc2016_ms"],
        "dior_public_mirror_yaml": DOMAINS["dior_public_mirror"],
    }
    mismatches = []
    for key, path in files.items():
        actual = base.sha256_file(path)
        expected = frozen["files"][key]["sha256"]
        if actual != expected:
            mismatches.append({"key": key, "expected": expected, "actual": actual})
    jobs, validation_freeze = base.audit_freeze()
    if frozen["checkpoint_manifest_sha256"] != base.sha256_file(STUDY / "validation_freeze.json"):
        mismatches.append({"key": "checkpoint_manifest_sha256"})
    if mismatches:
        raise RuntimeError(f"External-evaluation freeze audit failed: {mismatches}")
    return jobs, frozen, decision


def evaluate(domain: str, job: dict) -> dict:
    scope, condition, seed = job["scope"], job["condition"], int(job["seed"])
    checkpoint = base.checkpoint_for(job)
    name = f"{domain}_{scope}_{condition}_s{seed}"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        metrics = YOLO(str(checkpoint)).val(
            data=str(DOMAINS[domain]), split="test", imgsz=640, batch=8,
            device="0", workers=0, conf=0.001, iou=0.7, max_det=300,
            plots=False, verbose=False, save_json=False, project=str(RUNS),
            name=name, exist_ok=False,
        )
    indices = [int(value) for value in metrics.box.ap_class_index]
    if base.VESSEL_CLASS not in indices:
        raise RuntimeError(f"Vessel class missing: {name}")
    position = indices.index(base.VESSEL_CLASS)
    all_ap = metrics.box.all_ap
    payload = {
        "domain": domain, "scope": scope, "condition": condition, "seed": seed,
        "checkpoint": str(checkpoint), "checkpoint_sha256": base.sha256_file(checkpoint),
        "aggregate": {
            "ap50_95": float(metrics.box.map), "ap50": float(metrics.box.map50),
            "ap75": float(metrics.box.map75),
        },
        "vessel": {
            "ap50_95": float(metrics.box.maps[base.VESSEL_CLASS]),
            "ap50": float(all_ap[position, 0]), "ap75": float(all_ap[position, 5]),
        },
        "ap_class_index": indices,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "prediction_artifacts_saved": False,
    }
    write_json_exclusive(OUTPUT / f"{name}.json", payload)
    (OUTPUT / f"{name}.log").write_text(stdout.getvalue() + stderr.getvalue(), encoding="utf-8")
    return payload


def build_report(summary: dict) -> str:
    labels = {"xview_test": "xView test", "hrsc2016_ms": "HRSC2016-MS", "dior_public_mirror": "DIOR public mirror"}
    lines = [
        "# v48 Post-Lock Evaluation / v48 판정 동결 후 평가", "",
        "The xView-validation decision was frozen before these datasets were opened.",
        "xView validation 판정을 동결한 뒤에만 아래 데이터셋을 평가했다.", "",
    ]
    for domain, label in labels.items():
        lines += [f"## {label}", "", "| Family | Metric | Head Unity effect | Full Unity effect | Head-minus-full DiD (95% CI) | Signs |", "|---|---|---:|---:|---:|---:|"]
        for family in base.FAMILIES:
            for metric in base.METRICS:
                block = summary["analysis"][domain][family][metric]
                did = block["head_minus_full_difference_in_differences"]
                low, high = did["two_sided_95_ci"]
                lines.append(
                    f"| {family} | {metric} | {block['unity_effect_head']['mean']:+.6f} | "
                    f"{block['unity_effect_full']['mean']:+.6f} | {did['mean']:+.6f} [{low:+.6f}, {high:+.6f}] | "
                    f"{did['positive_seeds']}+/{did['negative_seeds']}- |"
                )
        lines.append("")
    lines += [
        "No validation decision was revised using these results. No predictions or embeddings were retained.",
        "이 결과로 validation 판정을 수정하지 않았고 prediction·embedding은 보존하지 않았다.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    if OUTPUT.exists() or RUNS.exists():
        raise FileExistsError("Refusing to overwrite existing v48 external evaluation")
    jobs, frozen, decision = audit()
    OUTPUT.mkdir(parents=True)
    write_json_exclusive(OUTPUT / "evaluation_started.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": base.sha256_file(FREEZE),
        "validation_decision_lock_sha256": base.sha256_file(DECISION_LOCK),
        "domain_order": list(DOMAINS), "jobs": 120,
    })
    ordered = sorted(jobs, key=lambda j: (base.SCOPES.index(j["scope"]), base.CONDITIONS.index(j["condition"]), int(j["seed"])))
    rows = []
    ordinal = 0
    for domain in DOMAINS:
        for job in ordered:
            ordinal += 1
            print(f"V48 POST-LOCK {ordinal}/120 {domain} {job['scope']} {job['condition']} seed={job['seed']}", flush=True)
            rows.append(evaluate(domain, job))
    analysis = {domain: base.analyze([row for row in rows if row["domain"] == domain]) for domain in DOMAINS}
    summary = {
        "study": "v48 adaptation-scope fairness post-lock evaluation",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "freeze": frozen,
        "frozen_validation_decision": decision,
        "validation_decision_changed": False,
        "analysis": analysis,
        "raw_results": rows,
        "prediction_artifacts_saved": False,
    }
    write_json_exclusive(OUTPUT / "summary.json", summary)
    (OUTPUT / "report_en_ko.md").write_text(build_report(summary), encoding="utf-8")
    print("V48 POST-LOCK EVALUATION PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
