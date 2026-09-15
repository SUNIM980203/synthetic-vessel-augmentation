#!/usr/bin/env python3
"""Shared, result-independent definitions for the fresh scope x LR factorial."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import scipy.stats


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "scope_lr_factorial_followup_v1"
PROTOCOL_DIR = STUDY / "00_protocol"
ENVIRONMENT_DIR = STUDY / "01_environment"
CONFIG_DIR = STUDY / "02_configs"
ORDER_DIR = STUDY / "03_execution_order"
RUNS_DIR = STUDY / "04_runs"
FAILED_DIR = STUDY / "05_failed_technical_runs"
PREDICTIONS_DIR = STUDY / "06_predictions"
METRICS_DIR = STUDY / "07_metrics"
STATISTICS_DIR = STUDY / "08_statistics"
SCENE_DIR = STUDY / "09_scene_sensitivity"
TABLES_DIR = STUDY / "10_tables"
FIGURES_DIR = STUDY / "11_figures"
AUDITS_DIR = STUDY / "12_audits"
REPORT_DIR = STUDY / "13_final_report"

SEEDS = list(range(20260723, 20260733))
CONDITIONS = ("duplicate", "unity_medium150")
SCOPES = ("head_only", "full_network")
LEARNING_RATES = (0.0001, 0.0002)
LR_LABELS = {0.0001: "1e-4", 0.0002: "2e-4"}
FREEZE = {"head_only": 23, "full_network": 0}
VESSEL_CLASS = 4
METRICS = ("ap50_95", "ap50", "ap75")
FAMILIES = ("vessel", "aggregate")
DOMAINS = {
    "xview_test": ROOT / "config/yolo_xview_expanded_real_v1.yaml",
    "hrsc2016_ms": ROOT / "config/yolo_hrsc2016_ms_v13.yaml",
    "dior_public_mirror": ROOT / "config/yolo_dior_v14.yaml",
}

INITIAL_CHECKPOINT = ROOT / "runs/expanded_v1/duplicate_yolo26n_s20260723/weights/best.pt"
EXPECTED_INITIAL_SHA256 = "7aab2bd4aebb6181df0350c80883e0c1b2615f363481649d579a23b0e4a7140f"
TRAIN_DATA = {
    "duplicate": ROOT / "config/yolo_xview_expanded_duplicate_v1.yaml",
    "unity_medium150": ROOT / "config/yolo_xview_expanded_vessel_medium_v10_150.yaml",
}

FROZEN_CONFIG = CONFIG_DIR / "scope_lr_factorial_frozen_training_config.json"
EXECUTION_ORDER = ORDER_DIR / "scope_lr_factorial_execution_order.csv"
PROTOCOL_JSON = PROTOCOL_DIR / "scope_lr_factorial_protocol_v1.json"
PROTOCOL_MD = PROTOCOL_DIR / "scope_lr_factorial_protocol_v1.md"
PROTOCOL_LOCK = PROTOCOL_DIR / "scope_lr_factorial_PROTOCOL_LOCK.txt"
ANALYSIS_MANIFEST = PROTOCOL_DIR / "scope_lr_factorial_analysis_manifest.json"
TRAINING_LEDGER = AUDITS_DIR / "scope_lr_factorial_training_ledger.json"
EVALUATION_LEDGER = AUDITS_DIR / "scope_lr_factorial_evaluation_ledger.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def result_rows(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def cell_name(condition: str, scope: str, learning_rate: float, seed: int) -> str:
    return f"{condition}__{scope}__lr{LR_LABELS[float(learning_rate)]}__s{int(seed)}"


def exact_complete(run_dir: Path, epochs: int = 20) -> bool:
    return result_rows(run_dir) == epochs and (run_dir / "weights/last.pt").is_file()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_execution_order() -> list[dict]:
    with EXECUTION_ORDER.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["ordinal"] = int(row["ordinal"])
        row["seed"] = int(row["seed"])
        row["learning_rate"] = float(row["learning_rate"])
        row["freeze"] = int(row["freeze"])
    return rows


def sign_test_two_sided(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0
    positive = sum(value > 0.0 for value in nonzero)
    tail = min(positive, len(nonzero) - positive)
    probability = sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2.0 * probability)


def student_summary(values: list[float]) -> dict:
    if len(values) < 2:
        raise ValueError("At least two paired values are required")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    sem = sd / math.sqrt(len(values))
    critical = float(scipy.stats.t.ppf(0.975, len(values) - 1))
    if sem == 0.0:
        t_statistic = math.inf if mean != 0.0 else 0.0
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        t_statistic = mean / sem
        p_value = float(2.0 * scipy.stats.t.sf(abs(t_statistic), len(values) - 1))
    return {
        "values": [float(value) for value in values],
        "n": len(values),
        "mean": float(mean),
        "sample_sd": float(sd),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "two_sided_95_ci": [float(mean - critical * sem), float(mean + critical * sem)],
        "student_t_df": len(values) - 1,
        "student_t_critical_0_975": critical,
        "one_sample_t_statistic": float(t_statistic),
        "one_sample_t_two_sided_p": p_value,
        "positive_count": sum(value > 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "zero_count": sum(value == 0.0 for value in values),
        "exact_two_sided_sign_p": sign_test_two_sided(values),
    }


def assert_protocol_locked() -> None:
    if not PROTOCOL_LOCK.is_file():
        raise RuntimeError("Protocol lock is missing")
    lock = PROTOCOL_LOCK.read_text(encoding="utf-8")
    for path in (PROTOCOL_JSON, PROTOCOL_MD, ANALYSIS_MANIFEST, FROZEN_CONFIG, EXECUTION_ORDER):
        digest = sha256_file(path)
        if digest not in lock:
            raise RuntimeError(f"Frozen file hash is not present in protocol lock: {path}")

