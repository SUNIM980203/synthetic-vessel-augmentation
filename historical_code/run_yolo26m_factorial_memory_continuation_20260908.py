"""One-shot user-authorized continuation after exact 20-epoch cache identity PASS.

Uses the unchanged frozen recipe/runner. Only unused CUDA cache is returned at
each batch boundary, as in the accepted-39 non-accepting identity replay.
No checkpoint recovery, automatic fresh retry, or postprocessing is launched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import psutil
import torch
import run_yolo26m_factorial_fresh as frozen
import run_yolo26m_factorial_360w_retry2 as safety_helpers
import yolo26m_factorial_common as common
from yolo26m_memory_cache_guard import MemoryCacheGuard

AUDIT = common.AUDITS_DIR / "yolo26m_memory_continuation_20260908"
AUTH = AUDIT / "continuation_authorization.json"
STOP = AUDIT / "YOLO26M_MEMORY_CONTINUATION_STOP.txt"
COMPLETE = AUDIT / "YOLO26M_MEMORY_CONTINUATION_COMPLETE.txt"
VERIFICATION = common.AUDITS_DIR / "yolo26m_memory_guard_full_identity_20260907"
PAUSED_LEDGER_SHA = "007a99d30e9684c051a75ae75fd5f9a8499379392cd8170ded792eb950769adb"
GUARD_SHA = "d0934863a978545121ecb0df89beb56b440bd3a4a3f0d408d2553ba4f329d252"
TECHNICAL_MODE = "yolo26m_360w_85pct_boostoff_cache_every_batch_identity_verified_20260908"
POLICY = {"reserve_fraction": 0.0, "min_unused_bytes": 0}
ORIGINAL_YOLO = frozen.YOLO
ORIGINAL_RUN_CELL = frozen.run_cell
IMMUTABLE_HASHES = {}
CURRENT_JOB = None
ACTIVE_GUARD = None
QUEUE_START = None
LAST_SAFETY = 0.0


def require_hashes(hashes):
    for path, expected in hashes.items():
        if common.sha256_file(Path(path)) != expected:
            raise RuntimeError(f"Protected hash drift: {path}")


def assert_no_other_python():
    # This queue uses the full single GPU; do not overlap any Python job.
    # Ignore our own process and executable-launch ancestors only.
    ignored = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid not in ignored and (process.info["name"] or "").lower().startswith("python"):
            raise RuntimeError(f"Another Python process is alive: {process.pid}")


def safety_state(since=None):
    global LAST_SAFETY
    state = safety_helpers.safety_state(since)
    for setting, expected in (("PROCTHROTTLEMAX", 85), ("PERFBOOSTMODE", 0)):
        raw = subprocess.run(["powercfg", "/QH", "SCHEME_CURRENT", "SUB_PROCESSOR", setting],
                             capture_output=True, check=True, timeout=10).stdout
        values = [int(value, 16) for value in re.findall(rb"0x([0-9a-fA-F]{8})", raw)][-2:]
        if values != [expected, expected]:
            raise RuntimeError(f"CPU power-profile drift: {setting} AC/DC={values}")
        state[setting] = values
    require_hashes(IMMUTABLE_HASHES)
    LAST_SAFETY = time.monotonic()
    return state


def audit_accepted(ledger):
    accepted = [j for j in ledger["jobs"] if j["status"] == "ACCEPTED"]
    for job in accepted:
        directory = Path(job["accepted_run_dir"])
        if not directory.resolve().is_relative_to(common.RUNS_DIR.resolve()):
            raise RuntimeError("Unsafe accepted path")
        if job.get("resume") is not False or job.get("uninterrupted") is not True:
            raise RuntimeError("Accepted provenance drift")
        if not common.exact_complete(directory, 20) or frozen.verify_args(directory, job):
            raise RuntimeError(f"Accepted artifact/args drift: ordinal {job['ordinal']}")
        digest = common.sha256_file(Path(job["accepted_checkpoint"]))
        attempts = [a for a in job["attempts"] if a["status"] == "ACCEPTED"]
        if len(attempts) != 1 or attempts[0].get("resume") is not False:
            raise RuntimeError("Accepted attempt provenance drift")
        if digest != job["accepted_checkpoint_sha256"] or digest != attempts[0]["last_checkpoint_sha256"]:
            raise RuntimeError("Accepted checkpoint SHA mismatch")
    return len(accepted)


def preflight():
    global IMMUTABLE_HASHES
    assert_no_other_python()
    if AUDIT.exists():
        raise RuntimeError("Continuation is one-shot: existing audit directory; do not relaunch")
    common.assert_protocol_locked()
    manifest = common.load_json(VERIFICATION / "manifest.json")
    result = common.load_json(VERIFICATION / "result.json")
    if (result.get("status") != "FULL_IDENTITY_PASS" or result.get("completed_epochs") != 20
            or not all(result.get(k) is True for k in ("exact_model_tensors", "exact_metrics_excluding_time",
                                                     "scientific_ledger_unchanged", "safety_pass"))
            or result.get("resume") is not False or result.get("accepted") is not False):
        raise RuntimeError("Full identity gate not PASS")
    if (VERIFICATION / "stderr.log").stat().st_size != 0 or (VERIFICATION / "STOP_REQUEST.json").exists():
        raise RuntimeError("Verification error/STOP artifact")
    require_hashes(manifest["protected_hashes"])
    require_hashes({common.TRAINING_LEDGER: PAUSED_LEDGER_SHA,
                   common.ROOT / "tools/yolo26m_memory_cache_guard.py": GUARD_SHA})
    candidate = VERIFICATION / "technical_runs/ordinal39_fresh_20epochs/weights/last.pt"
    require_hashes({candidate: result["candidate_checkpoint_sha256"]})
    ledger = common.load_json(common.TRAINING_LEDGER)
    if (ledger.get("status") != "USER_PAUSED_LOADING_BENCHMARK" or ledger.get("accepted_cells") != 42
            or ledger.get("resumed_accepted_cells") != 0 or ledger.get("invalid_accepted_cells") != 0):
        raise RuntimeError("Expected unchanged 42/80 user-pause boundary")
    order = common.load_execution_order()
    if len(ledger["jobs"]) != 80 or len(order) != 80:
        raise RuntimeError("Expected exact 80-job order")
    for job, row in zip(sorted(ledger["jobs"], key=lambda j: j["ordinal"]), order):
        if any(job[key] != row[key] for key in ("ordinal", "condition", "scope", "learning_rate", "seed", "freeze")):
            raise RuntimeError("Execution order / scientific condition drift")
        if job["cell_name"] != common.cell_name(job["condition"], job["scope"], job["learning_rate"], job["seed"]):
            raise RuntimeError("Cell name drift")
        expected_status = "ACCEPTED" if job["ordinal"] <= 42 else "USER_PAUSED" if job["ordinal"] == 43 else "PENDING"
        if job["status"] != expected_status or any(a["status"] == "RUNNING" for a in job["attempts"]):
            raise RuntimeError("Unexpected or unresolved job state")
    job = ledger["jobs"][42]
    if len(job["attempts"]) != 2:
        raise RuntimeError("Expected two preserved ordinal-43 attempts")
    latest = job["attempts"][-1]
    if (latest["status"] != "INVALID_TECHNICAL" or latest.get("resume") is not False
            or latest.get("last_completed_epoch") != 0
            or latest.get("reason") != "USER_REQUESTED_PAUSE_FOR_NONACCEPTING_LOADING_BENCHMARK"):
        raise RuntimeError("User-pause provenance drift")
    if audit_accepted(ledger) != 42:
        raise RuntimeError("42 accepted audit failed")
    if frozen.versions() != frozen.EXPECTED_ENVIRONMENT or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen runtime / CUDA mismatch")
    IMMUTABLE_HASHES = {p: h for p, h in manifest["protected_hashes"].items()
                        if Path(p).resolve() != common.TRAINING_LEDGER.resolve()}
    for path in (common.FROZEN_CONFIG, common.EXECUTION_ORDER, Path(__file__), Path(safety_helpers.__file__),
                 VERIFICATION / "manifest.json", VERIFICATION / "result.json"):
        IMMUTABLE_HASHES[str(path)] = common.sha256_file(path)
    frozen.GPU_POWER_LIMIT_WATTS = 360.0
    state = safety_state(manifest["started_utc"])
    return ledger, state


def epoch_start(trainer):
    trainer._oom_retries = 3
    trainer.nan_recovery_attempts = 3


def batch_start(trainer):
    if STOP.exists():
        raise RuntimeError("Technical stop requested; no retry")
    if trainer.batch_size != 16 or trainer.args.batch != 16 or trainer.args.resume is not False:
        raise RuntimeError("Batch/resume drift")


def batch_end(trainer):
    if not bool(torch.isfinite(trainer.loss).all()):
        raise RuntimeError("Non-finite loss; no checkpoint recovery")


def epoch_end(trainer):
    if time.monotonic() - LAST_SAFETY >= 600:
        safety_state(QUEUE_START)
    common.atomic_json(AUDIT / "progress.json", {
        "status": "RUNNING", "pid": os.getpid(), "updated_utc": common.utc_now(),
        "ordinal": CURRENT_JOB["ordinal"], "cell": CURRENT_JOB["cell_name"],
        "attempt": CURRENT_JOB["attempts"][-1]["attempt_number"],
        "completed_epochs": int(trainer.epoch) + 1, "resume": False,
        "cache_returns": len(ACTIVE_GUARD.records),
    })
    print(f"CACHE GUARD EPOCH {CURRENT_JOB['ordinal']}/80 {int(trainer.epoch)+1}/20", flush=True)


def guarded_yolo(*args, **kwargs):
    global ACTIVE_GUARD
    model = ORIGINAL_YOLO(*args, **kwargs)
    ACTIVE_GUARD = MemoryCacheGuard(**POLICY)
    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_start", batch_start)
    model.add_callback("on_train_batch_end", ACTIVE_GUARD)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_fit_epoch_end", epoch_end)
    # The original frozen run_cell adds its unchanged 0.25-second sleep next.
    return model


def run_cell(job, ledger, config, envelope):
    global CURRENT_JOB, ACTIVE_GUARD
    CURRENT_JOB, ACTIVE_GUARD = job, None
    expected_dir = common.FAILED_DIR / job["cell_name"] / f"attempt_{len(job['attempts'])+1:03d}"
    if not expected_dir.resolve().is_relative_to(common.FAILED_DIR.resolve()):
        raise RuntimeError("Unsafe attempt path")
    try:
        return ORIGINAL_RUN_CELL(job, ledger, config, envelope)
    finally:
        if ACTIVE_GUARD is not None:
            attempt = job["attempts"][-1]
            common.atomic_json(AUDIT / f"memory_ordinal{job['ordinal']:02d}_attempt{attempt['attempt_number']:03d}.json", {
                "ordinal": job["ordinal"], "attempt": attempt["attempt_number"], "policy": POLICY,
                "status": attempt["status"], "records": ACTIVE_GUARD.records,
            })
        CURRENT_JOB, ACTIVE_GUARD = None, None


def main(check_only=False):
    global QUEUE_START
    ledger, state = preflight()
    if check_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "accepted": 42, "next_ordinal": 43,
                          "next_attempt": 3, "resume": False, "safety": state}), flush=True)
        return 0
    AUDIT.mkdir(exist_ok=False)
    shutil.copy2(common.TRAINING_LEDGER, AUDIT / "ledger_before_continuation.json")
    QUEUE_START = common.utc_now()
    common.write_json_exclusive(AUTH, {
        "authorized_utc": QUEUE_START, "pid": os.getpid(), "user_authorization": "2026-09-08 user: 계속진행",
        "accepted_cells_preserved": 42, "target_ordinal": 43, "new_attempt": 3, "resume": False,
        "scientific_design_changed": False, "frozen_runner_modified": False, "protocol_relocked": False,
        "technical_mode": TECHNICAL_MODE, "cache_policy": POLICY,
        "identity_verification": str(VERIFICATION / "result.json"), "immutable_hashes": IMMUTABLE_HASHES,
        "ledger_before_sha256": PAUSED_LEDGER_SHA, "safety": state,
        "automatic_retry": False, "checkpoint_recovery": False, "postprocess_autolaunch": False,
        "stopping_rule": "Any safety/artifact/error stops without retry; preserve all failed attempts and prior STOP markers",
    })
    job = ledger["jobs"][42]
    job["status"] = "PENDING"
    job["fresh_restart_authorized"] = {"attempt_number": 3, "resume": False, "authorization": str(AUTH)}
    ledger.setdefault("previous_queue_starts", []).append({"queue_start_utc": ledger.get("queue_start_utc"),
        "ended_utc": QUEUE_START, "ended_reason": "USER_PAUSED_LOADING_BENCHMARK"})
    ledger.setdefault("technical_environment_transitions", []).append({"recorded_utc": QUEUE_START,
        "type": TECHNICAL_MODE, "scientific_design_changed": False, "accepted_cells_before_transition": 42,
        "audit": str(AUTH)})
    ledger.update(status="RUNNING_MEMORY_CACHE_CONTINUATION", queue_start_utc=QUEUE_START)
    frozen.save_ledger(ledger)
    frozen.STOP_FLAG, frozen.COMPLETE_FLAG = STOP, COMPLETE
    frozen.TECHNICAL_MODE = TECHNICAL_MODE
    frozen.safety_state, frozen.YOLO, frozen.run_cell = safety_state, guarded_yolo, run_cell
    print(json.dumps({"status": "STARTING", "pid": os.getpid(), "queue_start_utc": QUEUE_START,
                      "accepted": 42, "ordinal": 43, "attempt": 3, "resume": False}), flush=True)
    code = frozen.main()
    if code == 0:
        if audit_accepted(common.load_json(common.TRAINING_LEDGER)) != 80:
            raise RuntimeError("Final 80-accepted artifact audit failed")
    common.atomic_json(AUDIT / "completion.json", {"status": "PASS" if code == 0 else "STOP_NO_RETRY",
        "finished_utc": common.utc_now(), "returncode": code, "postprocess_started": False})
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="Read-only preflight; never starts training")
    args = parser.parse_args()
    try:
        code = main(args.preflight)
    except BaseException:
        if AUTH.exists() and not args.preflight:
            frozen.stop("Memory continuation exception; no retry: " + traceback.format_exc())
        raise
    raise SystemExit(code)
