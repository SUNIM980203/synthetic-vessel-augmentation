#!/usr/bin/env python3
"""Train the five new paired seeds for the frozen v46 four-condition expansion."""

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
CONFIG = ROOT / "config/seed_expansion_v46.json"
PROTOCOL = ROOT / "docs/seed_expansion_protocol_v46_en_ko.md"
PROJECT = ROOT / "runs/seed_expansion_v46"
OUTPUT = ROOT / "AnalysisResults/expanded_v1/seed_expansion_v46"
LEDGER = OUTPUT / "training_ledger.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="Resume one interrupted 1-19 epoch run from its saved last.pt; never deletes or overwrites it.",
    )
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
    results = run_dir / "results.csv"
    if not results.is_file():
        return 0
    with results.open("r", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def complete(run_dir: Path) -> bool:
    return (run_dir / "weights/last.pt").is_file() and result_rows(run_dir) == 20


def write_ledger(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = LEDGER.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(LEDGER)


def main() -> int:
    args = parse_args()
    study = json.loads(CONFIG.read_text(encoding="utf-8"))
    initial = ROOT / study["initial_weights"]
    if sha256_file(initial) != study["initial_weights_sha256"]:
        raise RuntimeError("Initial checkpoint SHA-256 mismatch")
    input_audit = {}
    for condition, spec in study["conditions"].items():
        data = ROOT / spec["data"]
        tree = ROOT / spec["train_tree"]
        if sha256_file(data) != spec["data_sha256"]:
            raise RuntimeError(f"Data YAML SHA-256 mismatch: {condition}")
        count, size, digest = sha256_train_tree(tree)
        if (count, size, digest) != (
            spec["train_tree_file_count"], spec["train_tree_bytes"], spec["train_tree_sha256"]
        ):
            raise RuntimeError(f"Frozen train tree mismatch: {condition}")
        input_audit[condition] = {"data": str(data), "tree": str(tree), "files": count, "bytes": size, "sha256": digest}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required")

    jobs = []
    for seed in study["new_seeds"]:
        for condition, spec in study["conditions"].items():
            name = f"{condition}_head20_adamw1e4_b16_s{seed}"
            run_dir = PROJECT / name
            rows = result_rows(run_dir)
            resumable = run_dir.exists() and 0 < rows < 20 and (run_dir / "weights/last.pt").is_file()
            if run_dir.exists() and not complete(run_dir) and not (args.resume_incomplete and resumable):
                raise RuntimeError(f"Refusing to overwrite incomplete run: {run_dir}")
            jobs.append({
                "condition": condition,
                "seed": seed,
                "data": str(ROOT / spec["data"]),
                "run_dir": str(run_dir),
                "name": name,
                "status": "preexisting_complete" if complete(run_dir) else "resume_pending" if resumable else "pending",
                "result_rows": rows,
            })
    ledger = {
        "study": study["study"],
        "started_utc": utc_now(),
        "finished_utc": None,
        "config_sha256": sha256_file(CONFIG),
        "protocol_sha256": sha256_file(PROTOCOL),
        "initial_weights_sha256": sha256_file(initial),
        "input_audit": input_audit,
        "software": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "fixed_recipe": study["fixed_recipe"],
        "jobs": jobs,
    }
    write_ledger(ledger)
    pending = [job for job in jobs if job["status"] in {"pending", "resume_pending"}]
    print(f"V46 INPUT AUDIT PASS: {len(jobs)-len(pending)}/{len(jobs)} complete; {len(pending)} pending", flush=True)
    for ordinal, job in enumerate(pending, 1):
        print(f"V46 START {ordinal}/{len(pending)} {job['condition']} seed={job['seed']} {utc_now()}", flush=True)
        job["status"] = "running"
        job["started_utc"] = utc_now()
        started = time.perf_counter()
        write_ledger(ledger)
        try:
            if job.get("result_rows", 0) > 0:
                job["resumed_from_epoch"] = job["result_rows"]
                job["resume_checkpoint_sha256"] = sha256_file(Path(job["run_dir"]) / "weights/last.pt")
                write_ledger(ledger)
                model = YOLO(str(Path(job["run_dir"]) / "weights/last.pt"))
                model.train(resume=True, device="0", workers=0)
            else:
                model = YOLO(str(initial))
                model.train(
                    data=job["data"], epochs=20, patience=100, batch=16, imgsz=640,
                    save=True, save_period=1, cache=False, device="0", workers=0,
                    project=str(PROJECT), name=job["name"], exist_ok=False,
                    pretrained=True, optimizer="AdamW", verbose=True, seed=job["seed"],
                    deterministic=True, close_mosaic=10, mosaic=0.5, mixup=0.0,
                    copy_paste=0.0, amp=True, freeze=23, plots=False, val=True,
                    split="val", lr0=0.0001, lrf=0.1, warmup_epochs=0.5,
                )
            run_dir = Path(job["run_dir"])
            if not complete(run_dir):
                raise RuntimeError(f"Exact epoch-20 artifact missing: {run_dir}")
            job["status"] = "complete"
            job["result_rows"] = 20
            job["last_checkpoint"] = str(run_dir / "weights/last.pt")
            job["last_checkpoint_sha256"] = sha256_file(run_dir / "weights/last.pt")
        except Exception as error:
            job["status"] = "failed"
            job["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            job["duration_seconds"] = time.perf_counter() - started
            job["finished_utc"] = utc_now()
            write_ledger(ledger)
    ledger["finished_utc"] = utc_now()
    ledger["status"] = "PASS" if all(job["status"] in {"complete", "preexisting_complete"} for job in jobs) else "FAIL"
    write_ledger(ledger)
    print(f"V46 TRAINING {ledger['status']}: {len(jobs)}/{len(jobs)} exact checkpoints", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
