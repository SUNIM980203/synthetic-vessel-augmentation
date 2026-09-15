#!/usr/bin/env python3
"""Run the frozen YOLO26s queue under the separately locked OS-stability envelope."""

from __future__ import annotations

import json
from pathlib import Path

import run_yolo26s_factorial_fresh as frozen_runner
import run_yolo26s_os_stability_identity_pilot as stability
import yolo26s_factorial_common as common


ORIGINAL_CONFIGURE_RESOURCE_ENVELOPE = frozen_runner.configure_resource_envelope
TECHNICAL_MODE = "yolo26s_cpu_boost_off_85pct_gpu250_v1"
GPU_POWER_LIMIT_WATTS = 250.0
POWER_SCHEME_GUID = "e892c4b4-2f3c-4274-9d0b-e6369d93bd5e"
LOCK_FILE = common.AUDITS_DIR / "yolo26s_os_stable_queue_restart_lock_20260825.json"
PILOT_RESULT = common.AUDITS_DIR / "yolo26s_os_stability_identity_pilot_result_20260825.json"
RUNTIME_PREFLIGHT = common.AUDITS_DIR / "yolo26s_os_stable_queue_runtime_preflight_20260825.json"
STOP_FLAG = common.AUDITS_DIR / "YOLO26S_FACTORIAL_TECHNICAL_STOP.txt"


def verify_lock() -> dict:
    if not LOCK_FILE.is_file():
        raise RuntimeError(f"queue restart lock is missing: {LOCK_FILE}")
    lock = common.load_json(LOCK_FILE)
    if common.sha256_file(Path(__file__).resolve()) != lock["queue_runner_sha256"]:
        raise RuntimeError("OS-stable queue runner self-hash mismatch")
    for name, item in lock["files"].items():
        path = common.ROOT / item["path"]
        if not path.is_file() or common.sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"locked file mismatch for {name}: {path}")
    return lock


def verify_pilot(lock: dict) -> dict:
    pilot = common.load_json(PILOT_RESULT)
    gates = [
        pilot.get("status") == "PASS",
        pilot.get("non_accepting") is True,
        pilot.get("results_rows") == 20,
        pilot.get("uninterrupted") is True,
        pilot.get("resume") is False,
        pilot.get("scientific_identity", {}).get("scientific_state_equal") is True,
        pilot.get("system_event_markers_unchanged") is True,
        pilot.get("main_training_ledger_unchanged") is True,
    ]
    if not all(gates):
        raise RuntimeError("stability/identity pilot is not an exact PASS")
    if common.sha256_file(PILOT_RESULT) != lock["pilot_result_sha256"]:
        raise RuntimeError("pilot result hash mismatch")
    return pilot


def verify_ledger(lock: dict) -> dict:
    ledger = common.load_json(common.TRAINING_LEDGER)
    if common.sha256_file(common.TRAINING_LEDGER) != lock["prestart_training_ledger_sha256"]:
        raise RuntimeError("training ledger moved after queue restart lock")
    if ledger.get("status") != "AUTHORIZED_FRESH_RESTART_AFTER_OS_STABILITY_PILOT":
        raise RuntimeError("training ledger lacks the explicit fresh-restart authorization")
    if ledger.get("accepted_cells") != 8 or ledger.get("invalid_accepted_cells") != 0:
        raise RuntimeError("training ledger moved away from accepted 8/80")
    jobs = [job for job in ledger["jobs"] if int(job["ordinal"]) == 9]
    if len(jobs) != 1:
        raise RuntimeError("ordinal 9 is absent or duplicated")
    job = jobs[0]
    if job.get("status") != "PENDING" or len(job.get("attempts", [])) != 1:
        raise RuntimeError("ordinal 9 is not frozen for fresh attempt 2")
    prior = job["attempts"][0]
    if not all(
        [
            prior.get("status") == "INVALID_TECHNICAL",
            prior.get("resume") is False,
            prior.get("resume_permitted") is False,
            prior.get("scientific_acceptance") is False,
        ]
    ):
        raise RuntimeError("ordinal 9 attempt 1 invalidation changed")
    if any(
        attempt.get("status") == "RUNNING"
        for queued_job in ledger["jobs"]
        for attempt in queued_job.get("attempts", [])
    ):
        raise RuntimeError("unresolved RUNNING attempt exists")
    return ledger


def os_stable_safety_state() -> dict:
    state = stability.system_state()
    stability.assert_stability_profile(state)
    return state


def os_stable_resource_envelope() -> dict:
    envelope = ORIGINAL_CONFIGURE_RESOURCE_ENVELOPE()
    state = os_stable_safety_state()
    envelope.update(
        {
            "technical_mode": TECHNICAL_MODE,
            "power_scheme_guid": POWER_SCHEME_GUID,
            "cpu_minimum_state_percent": 5,
            "cpu_maximum_state_percent_all_efficiency_classes": 85,
            "cpu_boost_mode": "disabled",
            "gpu_power_limit_watts": state["GpuPowerLimitWatts"],
        }
    )
    return envelope


def verify_preconditions() -> dict:
    lock = verify_lock()
    pilot = verify_pilot(lock)
    ledger = verify_ledger(lock)
    common.assert_protocol_locked()
    if STOP_FLAG.exists():
        raise RuntimeError(f"active technical stop marker exists: {STOP_FLAG}")
    system = os_stable_safety_state()
    if stability.event_snapshot() != pilot["system_events_after"]:
        raise RuntimeError("system or WHEA event marker moved after the PASS pilot")
    return {
        "lock_sha256": common.sha256_file(LOCK_FILE),
        "pilot_result_sha256": common.sha256_file(PILOT_RESULT),
        "training_ledger_sha256": common.sha256_file(common.TRAINING_LEDGER),
        "accepted_cells": ledger["accepted_cells"],
        "next_ordinal": 9,
        "next_attempt": 2,
        "resume": False,
        "technical_mode": TECHNICAL_MODE,
        "system": system,
    }


def main() -> int:
    preflight = verify_preconditions()
    common.atomic_json(RUNTIME_PREFLIGHT, preflight)
    print(json.dumps({"status": "OS_STABLE_QUEUE_PREFLIGHT_PASS", **preflight}, indent=2), flush=True)
    frozen_runner.GPU_POWER_LIMIT_WATTS = GPU_POWER_LIMIT_WATTS
    frozen_runner.TECHNICAL_MODE = TECHNICAL_MODE
    frozen_runner.STOP_FLAG = STOP_FLAG
    frozen_runner.safety_state = os_stable_safety_state
    frozen_runner.configure_resource_envelope = os_stable_resource_envelope
    return frozen_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
