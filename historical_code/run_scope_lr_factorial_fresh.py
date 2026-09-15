#!/usr/bin/env python3
"""Run the locked 80-cell fresh scope x LR factorial without accepting resumes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

import scope_lr_factorial_common as common


MAX_ATTEMPTS_PER_CELL = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-cell", action="store_true", help="Run at most one pending cell.")
    return parser.parse_args()


def load_ledger(order: list[dict]) -> dict:
    if common.TRAINING_LEDGER.is_file():
        return common.load_json(common.TRAINING_LEDGER)
    jobs = []
    for row in order:
        jobs.append({
            **row,
            "cell_name": common.cell_name(row["condition"], row["scope"], row["learning_rate"], row["seed"]),
            "status": "PENDING",
            "attempts": [],
        })
    ledger = {
        "study": "fresh 2x2x2 condition x adaptation-scope x learning-rate factorial",
        "created_utc": common.utc_now(),
        "updated_utc": common.utc_now(),
        "protocol_lock_sha256": common.sha256_file(common.PROTOCOL_LOCK),
        "accepted_cells": 0,
        "invalid_accepted_cells": 0,
        "jobs": jobs,
    }
    common.atomic_json(common.TRAINING_LEDGER, ledger)
    return ledger


def save_ledger(ledger: dict) -> None:
    ledger["updated_utc"] = common.utc_now()
    ledger["accepted_cells"] = sum(job["status"] == "ACCEPTED" for job in ledger["jobs"])
    ledger["invalid_accepted_cells"] = sum(job["status"] == "INVALID_ACCEPTED" for job in ledger["jobs"])
    common.atomic_json(common.TRAINING_LEDGER, ledger)


def mark_abandoned_attempts(ledger: dict) -> None:
    changed = False
    for job in ledger["jobs"]:
        for attempt in job["attempts"]:
            if attempt["status"] == "RUNNING":
                run_dir = Path(attempt["run_dir"])
                attempt.update({
                    "status": "INVALID_TECHNICAL",
                    "reason": "process_or_system_interruption_before_runner_terminal_record",
                    "last_completed_epoch": common.result_rows(run_dir),
                    "invalidated_utc": common.utc_now(),
                })
                job["status"] = "PENDING"
                changed = True
    if changed:
        save_ledger(ledger)


def verify_args(run_dir: Path, job: dict, config: dict) -> list[str]:
    path = run_dir / "args.yaml"
    if not path.is_file():
        return ["args.yaml missing"]
    args = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "epochs": 20,
        "batch": 16,
        "imgsz": 640,
        "optimizer": "AdamW",
        "seed": int(job["seed"]),
        "deterministic": True,
        "resume": False,
        "lr0": float(job["learning_rate"]),
        "freeze": int(job["freeze"]),
        "workers": 0,
        "device": "0",
    }
    mismatches = []
    for key, wanted in expected.items():
        actual = args.get(key)
        if key in {"lr0"}:
            if abs(float(actual) - wanted) > 1e-12:
                mismatches.append(f"{key}: {actual!r} != {wanted!r}")
        elif key == "device":
            if str(actual) != wanted:
                mismatches.append(f"{key}: {actual!r} != {wanted!r}")
        elif actual != wanted:
            mismatches.append(f"{key}: {actual!r} != {wanted!r}")
    if str(Path(args.get("model", "")).resolve()) != str(common.INITIAL_CHECKPOINT.resolve()):
        mismatches.append("model path is not the fixed initial checkpoint")
    data_expected = str(common.TRAIN_DATA[job["condition"]].resolve())
    if str(Path(args.get("data", "")).resolve()) != data_expected:
        mismatches.append("data path mismatch")
    return mismatches


def training_kwargs(job: dict, attempt_parent: Path, attempt_name: str, frozen: dict) -> dict:
    recipe = frozen["training_recipe"]
    kwargs = dict(recipe)
    kwargs.update({
        "data": str(common.TRAIN_DATA[job["condition"]].resolve()),
        "seed": int(job["seed"]),
        "lr0": float(job["learning_rate"]),
        "freeze": int(job["freeze"]),
        "project": str(attempt_parent.resolve()),
        "name": attempt_name,
        "exist_ok": False,
        "device": "0",
        "resume": False,
    })
    return kwargs


def run_cell(job: dict, ledger: dict, frozen: dict) -> bool:
    invalid_count = sum(attempt["status"] == "INVALID_TECHNICAL" for attempt in job["attempts"])
    if invalid_count >= MAX_ATTEMPTS_PER_CELL:
        job["status"] = "SYSTEMATIC_STOP"
        job["stop_reason"] = f"{invalid_count} failed fresh attempts for the same cell"
        save_ledger(ledger)
        (common.AUDITS_DIR / "SYSTEMATIC_TECHNICAL_STOP.txt").write_text(
            f"{common.utc_now()}\n{job['cell_name']}\n{job['stop_reason']}\n", encoding="utf-8"
        )
        return False

    attempt_number = len(job["attempts"]) + 1
    attempt_name = f"attempt_{attempt_number:03d}"
    attempt_parent = common.FAILED_DIR / job["cell_name"]
    attempt_dir = attempt_parent / attempt_name
    if attempt_dir.exists():
        raise RuntimeError(f"Attempt path collision: {attempt_dir}")
    attempt = {
        "attempt_number": attempt_number,
        "status": "RUNNING",
        "started_utc": common.utc_now(),
        "run_dir": str(attempt_dir),
        "initial_checkpoint": str(common.INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": common.sha256_file(common.INITIAL_CHECKPOINT),
        "frozen_training_config_sha256": common.sha256_file(common.FROZEN_CONFIG),
        "resume": False,
    }
    job["attempts"].append(attempt)
    job["status"] = "RUNNING"
    save_ledger(ledger)
    print(f"FACTORIAL START {job['ordinal']}/80 {job['cell_name']} attempt={attempt_number}", flush=True)
    started = time.perf_counter()
    try:
        YOLO(str(common.INITIAL_CHECKPOINT)).train(**training_kwargs(job, attempt_parent, attempt_name, frozen))
        if not common.exact_complete(attempt_dir, 20):
            raise RuntimeError(f"exact uninterrupted epoch-20 artifacts missing: rows={common.result_rows(attempt_dir)}")
        mismatches = verify_args(attempt_dir, job, frozen)
        if mismatches:
            raise RuntimeError("accepted-run args mismatch: " + "; ".join(mismatches))
        accepted_dir = common.RUNS_DIR / job["cell_name"]
        if accepted_dir.exists():
            raise RuntimeError(f"refusing to overwrite accepted path: {accepted_dir}")
        checkpoint = attempt_dir / "weights/last.pt"
        attempt.update({
            "status": "ACCEPTED",
            "finished_utc": common.utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "last_completed_epoch": 20,
            "last_checkpoint_sha256": common.sha256_file(checkpoint),
        })
        accepted_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(attempt_dir), str(accepted_dir))
        attempt["run_dir"] = str(accepted_dir)
        job.update({
            "status": "ACCEPTED",
            "accepted_run_dir": str(accepted_dir),
            "accepted_checkpoint": str(accepted_dir / "weights/last.pt"),
            "accepted_checkpoint_sha256": common.sha256_file(accepted_dir / "weights/last.pt"),
            "accepted_utc": common.utc_now(),
            "uninterrupted": True,
            "resume": False,
        })
        save_ledger(ledger)
        print(f"FACTORIAL ACCEPTED {job['cell_name']} total={ledger['accepted_cells']}/80", flush=True)
        return True
    except Exception as error:
        attempt.update({
            "status": "INVALID_TECHNICAL",
            "finished_utc": common.utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "last_completed_epoch": common.result_rows(attempt_dir),
        })
        job["status"] = "PENDING"
        save_ledger(ledger)
        print(f"FACTORIAL INVALID_TECHNICAL {job['cell_name']}: {error}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    args = parse_args()
    common.assert_protocol_locked()
    if common.sha256_file(common.INITIAL_CHECKPOINT) != common.EXPECTED_INITIAL_SHA256:
        raise RuntimeError("fixed initial checkpoint SHA-256 mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("frozen CUDA device 0 is unavailable")
    order = common.load_execution_order()
    frozen = common.load_json(common.FROZEN_CONFIG)
    ledger = load_ledger(order)
    mark_abandoned_attempts(ledger)
    if len(ledger["jobs"]) != 80:
        raise RuntimeError("training ledger does not contain exactly 80 cells")

    accepted_now = 0
    for job in sorted(ledger["jobs"], key=lambda row: int(row["ordinal"])):
        if job["status"] == "ACCEPTED":
            if not common.exact_complete(Path(job["accepted_run_dir"]), 20):
                job["status"] = "INVALID_ACCEPTED"
                save_ledger(ledger)
                raise RuntimeError(f"previously accepted artifacts no longer pass: {job['cell_name']}")
            continue
        success = run_cell(job, ledger, frozen)
        if not success:
            invalid_count = sum(attempt["status"] == "INVALID_TECHNICAL" for attempt in job["attempts"])
            if invalid_count >= MAX_ATTEMPTS_PER_CELL:
                return 3
            return 2
        accepted_now += 1
        if args.one_cell and accepted_now >= 1:
            break

    save_ledger(ledger)
    if ledger["accepted_cells"] == 80 and ledger["invalid_accepted_cells"] == 0:
        ledger["status"] = "PASS"
        ledger["completed_utc"] = common.utc_now()
        save_ledger(ledger)
        print("FACTORIAL TRAINING PASS: ACCEPTED_CELLS=80 INVALID_ACCEPTED_CELLS=0", flush=True)
        return 0
    print(f"FACTORIAL TRAINING INCOMPLETE: {ledger['accepted_cells']}/80", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())

