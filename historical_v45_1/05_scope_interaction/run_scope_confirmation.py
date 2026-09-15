#!/usr/bin/env python3
"""Train the frozen v48 2x2 adaptation-scope confirmation cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "AnalysisResults/expanded_v1/adaptation_scope_fairness_v48/confirmation_freeze.json"
PROJECT = ROOT / "runs/adaptation_scope_fairness_v48_confirmation"
OUTPUT = ROOT / "AnalysisResults/expanded_v1/adaptation_scope_fairness_v48/confirmation"
LEDGER = OUTPUT / "training_ledger.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-incomplete", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_train_tree(base: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = total = 0
    for subfolder in ("images/train", "labels/train"):
        folder = base / subfolder
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            relative = path.relative_to(ROOT / "PreparedData/yolo_expanded_v1").as_posix()
            size = path.stat().st_size
            digest.update(f"{relative}\0{size}\0{sha256_file(path)}\n".encode())
            count += 1
            total += size
    return count, total, digest.hexdigest()


def result_rows(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def exact_complete(run_dir: Path, epochs: int) -> bool:
    return (run_dir / "weights/last.pt").is_file() and result_rows(run_dir) == epochs


def write_ledger(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = LEDGER.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEDGER)


def main() -> int:
    args = parse_args()
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "PASS":
        raise RuntimeError("v48 confirmation freeze is not PASS")
    initial = ROOT / frozen["initial_weights"]["path"]
    if sha256_file(initial) != frozen["initial_weights"]["sha256"]:
        raise RuntimeError("Frozen initialization SHA-256 mismatch")
    for item in frozen["frozen_files"]:
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Frozen file mismatch: {item['path']}")
    for condition, spec in frozen["conditions"].items():
        current = sha256_train_tree(ROOT / spec["train_tree"])
        expected = (spec["train_tree_file_count"], spec["train_tree_bytes"], spec["train_tree_sha256"])
        if current != expected:
            raise RuntimeError(f"Frozen training tree mismatch: {condition}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required")

    seeds = [int(value) for value in frozen["confirmation_seeds"]]
    selected = frozen["selected_scope_settings"]
    conditions = frozen["conditions"]
    reused = {(item["scope"], item["condition"], int(item["seed"])): item for item in frozen["reused_checkpoints"]}
    jobs: list[dict] = []
    for scope in ("head_only", "full_network"):
        setting = selected[scope]
        epochs = int(setting["epochs"])
        for condition in ("duplicate", "unity_medium150"):
            for seed in seeds:
                key = (scope, condition, seed)
                if key in reused:
                    item = reused[key]
                    checkpoint = ROOT / item["checkpoint"]
                    run_dir = checkpoint.parents[1]
                    if not exact_complete(run_dir, epochs) or sha256_file(checkpoint) != item["checkpoint_sha256"]:
                        raise RuntimeError(f"Frozen reused checkpoint mismatch: {checkpoint}")
                    status = "reused_exact"
                else:
                    name = f"{scope}_{condition}_lr{float(setting['learning_rate']):.0e}_e{epochs}_s{seed}".replace("-0", "-")
                    run_dir = PROJECT / name
                    checkpoint = run_dir / "weights/last.pt"
                    rows = result_rows(run_dir)
                    resumable = run_dir.exists() and 0 < rows < epochs and checkpoint.is_file()
                    if run_dir.exists() and not exact_complete(run_dir, epochs) and not (args.resume_incomplete and resumable):
                        raise RuntimeError(f"Refusing to overwrite incomplete run: {run_dir}")
                    status = "preexisting_complete" if exact_complete(run_dir, epochs) else "resume_pending" if resumable else "pending"
                jobs.append({
                    "scope": scope,
                    "condition": condition,
                    "seed": seed,
                    "data": str(ROOT / conditions[condition]["data"]),
                    "learning_rate": float(setting["learning_rate"]),
                    "freeze": int(setting["freeze"]),
                    "epochs": epochs,
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "status": status,
                    "result_rows": result_rows(run_dir),
                    "last_checkpoint": str(checkpoint) if checkpoint.is_file() else None,
                    "last_checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
                })

    ledger = {
        "study": "v48 adaptation-scope fairness confirmation",
        "started_utc": utc_now(),
        "finished_utc": None,
        "freeze_sha256": sha256_file(FREEZE),
        "software": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "selection_outcomes_used": ["duplicate_inner_validation"],
        "unity_outcomes_used_for_selection": False,
        "public_validation_test_external_used_for_selection": False,
        "jobs": jobs,
    }
    write_ledger(ledger)
    pending = [job for job in jobs if job["status"] in {"pending", "resume_pending"}]
    print(f"V48 CONFIRMATION AUDIT PASS: reused={len(jobs)-len(pending)} pending={len(pending)}", flush=True)
    for ordinal, job in enumerate(pending, 1):
        run_dir = Path(job["run_dir"])
        job["status"] = "running"
        job["started_utc"] = utc_now()
        started = time.perf_counter()
        write_ledger(ledger)
        print(f"V48 START {ordinal}/{len(pending)} {job['scope']} {job['condition']} seed={job['seed']}", flush=True)
        try:
            if job["result_rows"] > 0:
                job["resumed_from_epoch"] = job["result_rows"]
                job["resume_checkpoint_sha256"] = sha256_file(run_dir / "weights/last.pt")
                write_ledger(ledger)
                YOLO(str(run_dir / "weights/last.pt")).train(resume=True, device="0", workers=0)
            else:
                model = YOLO(str(initial))
                kwargs = dict(
                    data=job["data"], epochs=job["epochs"], patience=100, batch=16, imgsz=640,
                    save=True, save_period=1, cache=False, device="0", workers=0,
                    project=str(PROJECT), name=job["run_name"], exist_ok=False,
                    pretrained=True, optimizer="AdamW", verbose=True, seed=job["seed"],
                    deterministic=True, close_mosaic=10, mosaic=1.0, mixup=0.0,
                    copy_paste=0.0, amp=True, plots=False, val=True, split="val",
                    lr0=job["learning_rate"], lrf=0.1, warmup_epochs=0.5,
                )
                if job["scope"] == "head_only":
                    kwargs["freeze"] = job["freeze"]
                model.train(**kwargs)
            if not exact_complete(run_dir, job["epochs"]):
                raise RuntimeError(f"Exact final-epoch artifact missing: {run_dir}")
            checkpoint = run_dir / "weights/last.pt"
            job.update(status="complete", result_rows=job["epochs"], last_checkpoint=str(checkpoint), last_checkpoint_sha256=sha256_file(checkpoint))
        except Exception as error:
            job.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            job["duration_seconds"] = time.perf_counter() - started
            job["finished_utc"] = utc_now()
            write_ledger(ledger)

    accepted = {"complete", "preexisting_complete", "reused_exact"}
    ledger["finished_utc"] = utc_now()
    ledger["status"] = "PASS" if all(job["status"] in accepted for job in jobs) else "FAIL"
    write_ledger(ledger)
    print(f"V48 CONFIRMATION TRAINING {ledger['status']}: 40/40 exact final checkpoints", flush=True)
    return 0 if ledger["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
