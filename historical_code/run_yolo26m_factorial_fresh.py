#!/usr/bin/env python3
"""Run the locked 80-cell YOLO26m factorial under the low-load safety envelope."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import yaml
from ultralytics import YOLO

import yolo26m_factorial_common as common


CPU_THREADS = 4
CPU_AFFINITY_FRACTION = 0.5
BATCH_END_SLEEP_SECONDS = 0.25
GPU_POWER_LIMIT_WATTS = 250.0
EXPECTED_POWER_PLAN = "e892c4b4-2f3c-4274-9d0b-e6369d93bd5e"
TECHNICAL_MODE = "yolo26m_hyperv_off_250w_85pct_boostoff_v3_after_abnormal_reboot"
STOP_FLAG = common.AUDITS_DIR / "YOLO26M_FACTORIAL_TECHNICAL_STOP.txt"
COMPLETE_FLAG = common.AUDITS_DIR / "YOLO26M_FACTORIAL_TRAINING_COMPLETE.txt"
EXPECTED_ENVIRONMENT = {
    "python": "3.12.10", "torch": "2.11.0+cu128", "ultralytics": "8.4.50",
    "numpy": "2.4.4", "scipy": "1.17.1", "opencv": "4.13.0",
}


def versions() -> dict[str, str]:
    import cv2
    import numpy
    import scipy
    import ultralytics
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "numpy": numpy.__version__,
        "scipy": scipy.__version__, "opencv": cv2.__version__,
    }


def powershell_json(script: str) -> object | None:
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script], check=True,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=20,
    )
    payload = completed.stdout.strip()
    return json.loads(payload) if payload else None


def safety_state(queue_start_utc: str | None = None) -> dict:
    state = powershell_json(
        "$c=Get-CimInstance Win32_ComputerSystem;"
        "$d=Get-CimInstance -Namespace root\\Microsoft\\Windows\\DeviceGuard -ClassName Win32_DeviceGuard;"
        "[pscustomobject]@{HypervisorPresent=[bool]$c.HypervisorPresent;"
        "VBSStatus=[int]$d.VirtualizationBasedSecurityStatus;"
        "SecurityRunning=@($d.SecurityServicesRunning)}|ConvertTo-Json -Compress"
    )
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    state["GpuPowerLimitWatts"] = float(query.stdout.strip())
    plan = subprocess.run(["powercfg", "/GETACTIVESCHEME"], check=True, capture_output=True)
    match = re.search(rb"([0-9a-fA-F-]{36})", plan.stdout)
    state["ActivePowerPlanGuid"] = match.group(1).decode("ascii").lower() if match else None
    if state["HypervisorPresent"] or state["VBSStatus"] == 2 or 2 in state["SecurityRunning"]:
        raise RuntimeError(f"frozen Hyper-V/VBS-off safety state is not active: {state}")
    if abs(state["GpuPowerLimitWatts"] - GPU_POWER_LIMIT_WATTS) > 0.1:
        raise RuntimeError(f"GPU power-limit drift: {state}")
    if state["ActivePowerPlanGuid"] != EXPECTED_POWER_PLAN:
        raise RuntimeError(f"active power-plan drift: {state}")
    if queue_start_utc:
        event_query = "*[System[(EventID=41 or EventID=6008 or EventID=1001 or Provider[@Name='Microsoft-Windows-WHEA-Logger'])]]"
        completed = subprocess.run(
            ["wevtutil", "qe", "System", f"/q:{event_query}", "/rd:true", "/c:20", "/f:xml"],
            check=True, capture_output=True, timeout=10,
        )
        event_root = ET.fromstring("<Events>" + completed.stdout.decode("utf-8", errors="replace") + "</Events>")
        namespace = "{http://schemas.microsoft.com/win/2004/08/events/event}"
        queue_start = datetime.fromisoformat(queue_start_utc).astimezone(timezone.utc)
        events = []
        for event in event_root.findall(f"{namespace}Event"):
            system = event.find(f"{namespace}System")
            created = system.find(f"{namespace}TimeCreated")
            event_time = datetime.fromisoformat(created.attrib["SystemTime"].replace("Z", "+00:00"))
            if event_time < queue_start:
                continue
            provider = system.find(f"{namespace}Provider").attrib.get("Name")
            event_id = int(system.find(f"{namespace}EventID").text)
            events.append({"TimeCreatedUtc": event_time.isoformat(), "Id": event_id, "ProviderName": provider})
        if events:
            state["UnsafeSystemEvents"] = events
            raise RuntimeError(f"new abnormal reboot/WHEA event: {state['UnsafeSystemEvents']}")
        state["UnsafeSystemEvents"] = []
    return state


def configure_resource_envelope() -> dict:
    process = psutil.Process(os.getpid())
    logical = psutil.cpu_count(logical=True) or 1
    process.cpu_affinity(list(range(max(1, int(logical * CPU_AFFINITY_FRACTION)))))
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    return {
        "logical_cpu_count": logical, "cpu_affinity": process.cpu_affinity(),
        "priority": "BELOW_NORMAL_PRIORITY_CLASS", "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "batch_end_sleep_seconds": BATCH_END_SLEEP_SECONDS,
        "gpu_power_limit_watts": safety_state()["GpuPowerLimitWatts"],
    }


def stop(reason: str, job: dict | None = None) -> None:
    common.atomic_json(STOP_FLAG.with_suffix(".json"), {
        "recorded_utc": common.utc_now(), "reason": reason,
        "cell": job.get("cell_name") if job else None, "technical_mode": TECHNICAL_MODE,
    })
    STOP_FLAG.write_text(f"{common.utc_now()}\n{reason}\n", encoding="utf-8")


def load_ledger(order: list[dict]) -> dict:
    if common.TRAINING_LEDGER.is_file():
        return common.load_json(common.TRAINING_LEDGER)
    jobs = [{
        **row, "model_size": "YOLO26m",
        "cell_name": common.cell_name(row["condition"], row["scope"], row["learning_rate"], row["seed"]),
        "status": "PENDING", "attempts": [],
    } for row in order]
    ledger = {
        "study": "fresh YOLO26m 2x2x2 condition x adaptation-scope x learning-rate factorial",
        "created_utc": common.utc_now(), "updated_utc": common.utc_now(),
        "protocol_lock_sha256": common.sha256_file(common.PROTOCOL_LOCK),
        "accepted_cells": 0, "resumed_accepted_cells": 0, "invalid_accepted_cells": 0,
        "jobs": jobs,
    }
    common.atomic_json(common.TRAINING_LEDGER, ledger)
    return ledger


def save_ledger(ledger: dict) -> None:
    ledger["updated_utc"] = common.utc_now()
    ledger["accepted_cells"] = sum(j["status"] == "ACCEPTED" for j in ledger["jobs"])
    ledger["resumed_accepted_cells"] = sum(j.get("status") == "ACCEPTED" and j.get("resume") is not False for j in ledger["jobs"])
    ledger["invalid_accepted_cells"] = sum(j["status"] == "INVALID_ACCEPTED" for j in ledger["jobs"])
    common.atomic_json(common.TRAINING_LEDGER, ledger)


def verify_args(run_dir: Path, job: dict) -> list[str]:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return ["args.yaml missing"]
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = {
        "epochs": 20, "batch": 16, "imgsz": 640, "optimizer": "AdamW",
        "seed": int(job["seed"]), "deterministic": True, "resume": False,
        "lr0": float(job["learning_rate"]), "freeze": int(job["freeze"]),
        "workers": 0, "device": "0",
    }
    mismatches = []
    for key, wanted in expected.items():
        actual = args.get(key)
        if key == "lr0":
            if abs(float(actual) - wanted) > 1e-12:
                mismatches.append(f"{key}: {actual!r} != {wanted!r}")
        elif key == "device":
            if str(actual) != wanted:
                mismatches.append(f"{key}: {actual!r} != {wanted!r}")
        elif actual != wanted:
            mismatches.append(f"{key}: {actual!r} != {wanted!r}")
    if Path(args.get("model", "")).resolve() != common.INITIAL_CHECKPOINT.resolve():
        mismatches.append("model path is not the fixed YOLO26m xView checkpoint")
    if Path(args.get("data", "")).resolve() != common.TRAIN_DATA[job["condition"]].resolve():
        mismatches.append("data path mismatch")
    return mismatches


def training_kwargs(job: dict, parent: Path, name: str, frozen: dict) -> dict:
    kwargs = dict(frozen["training_recipe"])
    kwargs.update({
        "data": str(common.TRAIN_DATA[job["condition"]].resolve()),
        "seed": int(job["seed"]), "lr0": float(job["learning_rate"]),
        "freeze": int(job["freeze"]), "project": str(parent.resolve()), "name": name,
        "exist_ok": False, "device": "0", "resume": False,
    })
    return kwargs


def run_cell(job: dict, ledger: dict, frozen: dict, envelope: dict) -> bool:
    safety_state(ledger["queue_start_utc"])
    if any(a.get("status") == "RUNNING" for a in job["attempts"]):
        raise RuntimeError("unresolved interrupted attempt; automatic restart is prohibited")
    attempt_number = len(job["attempts"]) + 1
    attempt_name = f"attempt_{attempt_number:03d}"
    parent = common.FAILED_DIR / job["cell_name"]
    attempt_dir = parent / attempt_name
    if attempt_dir.exists():
        raise RuntimeError(f"attempt path collision: {attempt_dir}")
    attempt = {
        "attempt_number": attempt_number, "status": "RUNNING", "resume": False,
        "technical_mode": TECHNICAL_MODE, "resource_envelope": envelope,
        "started_utc": common.utc_now(), "run_dir": str(attempt_dir),
        "initial_checkpoint": str(common.INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": common.sha256_file(common.INITIAL_CHECKPOINT),
        "dataset_yaml": str(common.TRAIN_DATA[job["condition"]]),
        "dataset_yaml_sha256": common.sha256_file(common.TRAIN_DATA[job["condition"]]),
        "frozen_config_sha256": common.sha256_file(common.FROZEN_CONFIG),
    }
    job["attempts"].append(attempt)
    job["status"] = "RUNNING"
    save_ledger(ledger)
    print(f"YOLO26M START {job['ordinal']}/80 {job['cell_name']} attempt={attempt_number}", flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = YOLO(str(common.INITIAL_CHECKPOINT))
        model.add_callback("on_train_batch_end", lambda trainer: time.sleep(BATCH_END_SLEEP_SECONDS))
        model.train(**training_kwargs(job, parent, attempt_name, frozen))
        safety_state(ledger["queue_start_utc"])
        if not common.exact_complete(attempt_dir, 20):
            raise RuntimeError(f"expected 20 rows and last.pt; rows={common.result_rows(attempt_dir)}")
        mismatches = verify_args(attempt_dir, job)
        if mismatches:
            raise RuntimeError("accepted-run args mismatch: " + "; ".join(mismatches))
        accepted_dir = common.RUNS_DIR / job["cell_name"]
        if accepted_dir.exists():
            raise RuntimeError(f"refusing to overwrite accepted path: {accepted_dir}")
        checkpoint = attempt_dir / "weights" / "last.pt"
        duration = time.perf_counter() - started
        attempt.update({
            "status": "ACCEPTED", "finished_utc": common.utc_now(),
            "duration_seconds": duration, "average_epoch_seconds": duration / 20.0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "last_completed_epoch": 20, "last_checkpoint_sha256": common.sha256_file(checkpoint),
        })
        accepted_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(attempt_dir), str(accepted_dir))
        attempt["run_dir"] = str(accepted_dir)
        job.update({
            "status": "ACCEPTED", "accepted_run_dir": str(accepted_dir),
            "accepted_checkpoint": str(accepted_dir / "weights" / "last.pt"),
            "accepted_checkpoint_sha256": common.sha256_file(accepted_dir / "weights" / "last.pt"),
            "accepted_utc": common.utc_now(), "uninterrupted": True, "resume": False,
            "technical_mode": TECHNICAL_MODE,
        })
        save_ledger(ledger)
        print(f"YOLO26M ACCEPTED {job['cell_name']} total={ledger['accepted_cells']}/80", flush=True)
        return True
    except Exception as error:
        attempt.update({
            "status": "INVALID_TECHNICAL", "finished_utc": common.utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "last_completed_epoch": common.result_rows(attempt_dir),
        })
        job["status"] = "SYSTEMATIC_STOP"
        job["stop_reason"] = "fresh YOLO26m attempt failed; no automatic retry"
        save_ledger(ledger)
        stop(job["stop_reason"] + f": {error}", job)
        print(f"YOLO26M INVALID_TECHNICAL {job['cell_name']}: {error}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    common.assert_protocol_locked()
    if STOP_FLAG.exists():
        raise RuntimeError(f"technical stop marker exists: {STOP_FLAG}")
    if versions() != EXPECTED_ENVIRONMENT:
        raise RuntimeError(f"runtime mismatch: {versions()} != {EXPECTED_ENVIRONMENT}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("frozen single CUDA device is unavailable")
    if common.sha256_file(common.INITIAL_CHECKPOINT) != common.expected_initial_sha256():
        raise RuntimeError("fixed initial checkpoint hash mismatch")
    envelope = configure_resource_envelope()
    order = common.load_execution_order()
    if len(order) != 80:
        raise RuntimeError("execution order must contain exactly 80 cells")
    frozen = common.load_json(common.FROZEN_CONFIG)
    for path, expected in frozen["v48_document_hashes"].items():
        if common.sha256_file(common.ROOT / path) != expected:
            raise RuntimeError(f"v48 document hash drift: {path}")
    ledger = load_ledger(order)
    if "queue_start_utc" not in ledger:
        ledger["queue_start_utc"] = common.utc_now()
        save_ledger(ledger)
    if len(ledger["jobs"]) != 80:
        raise RuntimeError("training ledger must contain exactly 80 cells")
    abandoned = [(j, a) for j in ledger["jobs"] for a in j["attempts"] if a.get("status") == "RUNNING"]
    if abandoned:
        job, attempt = abandoned[0]
        attempt.update({"status": "INVALID_TECHNICAL", "reason": "process_or_system_interruption", "invalidated_utc": common.utc_now()})
        job["status"] = "SYSTEMATIC_STOP"
        save_ledger(ledger)
        stop("unresolved interrupted attempt; automatic restart prohibited", job)
        return 3
    print(json.dumps({"runtime": versions(), "safety": safety_state(ledger["queue_start_utc"]), "resource_envelope": envelope}, indent=2), flush=True)
    for job in sorted(ledger["jobs"], key=lambda row: int(row["ordinal"])):
        if job["status"] == "ACCEPTED":
            if not common.exact_complete(Path(job["accepted_run_dir"]), 20):
                job["status"] = "INVALID_ACCEPTED"
                save_ledger(ledger)
                stop("previously accepted artifacts no longer pass", job)
                return 3
            continue
        if job["status"] != "PENDING":
            stop(f"unexpected cell state: {job['status']}", job)
            return 3
        if not run_cell(job, ledger, frozen, envelope):
            return 3
    save_ledger(ledger)
    for path, expected in frozen["v48_document_hashes"].items():
        if common.sha256_file(common.ROOT / path) != expected:
            stop(f"v48 document hash drift after training: {path}")
            return 4
    if ledger["accepted_cells"] == 80 and ledger["resumed_accepted_cells"] == 0 and ledger["invalid_accepted_cells"] == 0:
        ledger["status"] = "PASS"
        ledger["completed_utc"] = common.utc_now()
        save_ledger(ledger)
        COMPLETE_FLAG.write_text(common.utc_now() + "\n", encoding="utf-8")
        print("YOLO26M FACTORIAL PASS: ACCEPTED=80 RESUMED=0 INVALID_ACCEPTED=0", flush=True)
        return 0
    stop("training ended without 80 accepted cells")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
