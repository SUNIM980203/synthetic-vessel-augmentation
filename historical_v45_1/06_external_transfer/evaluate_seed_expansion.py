#!/usr/bin/env python3
"""Evaluate the five new v46 seeds and assemble ten-seed paired analyses."""

from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import scipy.stats
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/seed_expansion_v46.json"
PROTOCOL = ROOT / "docs/seed_expansion_protocol_v46_en_ko.md"
OUTPUT = ROOT / "AnalysisResults/expanded_v1/seed_expansion_v46/evaluation"
RUNS = ROOT / "runs/seed_expansion_v46_evaluation"
V45 = ROOT / "AnalysisResults/expanded_v1/realonly_baseline_v45/evaluation/summary.json"
VESSEL_CLASS = 4
DOMAINS = {
    "xview_test": ROOT / "config/yolo_xview_expanded_real_v1.yaml",
    "hrsc2016_ms": ROOT / "config/yolo_hrsc2016_ms_v13.yaml",
    "dior_public_mirror": ROOT / "config/yolo_dior_v14.yaml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_rows(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def exact_epoch20(path: Path) -> bool:
    return path.is_file() and result_rows(path.parents[1]) == 20


def checkpoint(study: dict, condition: str, seed: int) -> Path:
    if seed in study["existing_seeds"]:
        patterns = {
            "realonly_standard": "runs/realonly_baseline_v45/realonly_head20_adamw1e4_b16_s{seed}/weights/last.pt",
            "duplicate": "runs/expanded_head_only_v11/duplicate_head20_adamw1e4_b16_s{seed}/weights/last.pt",
            "realcutout_medium150": "runs/expanded_real_vessel_control_v12/realvessel150_head20_adamw1e4_b16_s{seed}/weights/last.pt",
            "unity_medium150": "runs/expanded_head_only_v11/medium150_head20_adamw1e4_b16_s{seed}/weights/last.pt",
        }
        return ROOT / patterns[condition].format(seed=seed)
    return ROOT / f"runs/seed_expansion_v46/{condition}_head20_adamw1e4_b16_s{seed}/weights/last.pt"


def evaluate(domain: str, condition: str, seed: int, weights: Path) -> dict:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        metrics = YOLO(str(weights)).val(
            data=str(DOMAINS[domain]), split="test", imgsz=640, batch=8,
            device="0", workers=0, conf=0.001, iou=0.7, max_det=300,
            plots=False, verbose=False, save_json=True, project=str(RUNS),
            name=f"{domain}_{condition}_s{seed}", exist_ok=False,
        )
    indices = [int(value) for value in metrics.box.ap_class_index]
    if VESSEL_CLASS not in indices:
        raise RuntimeError(f"Vessel class missing: {domain} {condition} {seed}")
    position = indices.index(VESSEL_CLASS)
    all_ap = metrics.box.all_ap
    prediction = Path(metrics.save_dir) / "predictions.json"
    compressed = OUTPUT / f"{domain}_{condition}_s{seed}.predictions.json.gz"
    with prediction.open("rt", encoding="utf-8") as source, gzip.open(compressed, "wt", encoding="utf-8", compresslevel=9) as target:
        target.write(source.read())
    result = {
        "domain": domain, "condition": condition, "seed": seed,
        "checkpoint": str(weights), "checkpoint_sha256": sha256_file(weights),
        "aggregate": {"ap50_95": float(metrics.box.map), "ap50": float(metrics.box.map50), "ap75": float(metrics.box.map75)},
        "vessel": {
            "ap50_95": float(metrics.box.maps[VESSEL_CLASS]),
            "ap50": float(all_ap[position, 0]), "ap75": float(all_ap[position, 5]),
            "precision": float(metrics.box.p[position]), "recall": float(metrics.box.r[position]),
        },
        "raw_predictions": str(compressed), "raw_predictions_sha256": sha256_file(compressed),
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }
    (OUTPUT / f"{domain}_{condition}_s{seed}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / f"{domain}_{condition}_s{seed}.log").write_text(stdout.getvalue() + stderr.getvalue(), encoding="utf-8")
    return result


def sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    cumulative = sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2 * cumulative)


def paired(left: list[float], right: list[float]) -> dict:
    differences = [a - b for a, b in zip(left, right)]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    sem = sd / math.sqrt(len(differences))
    critical = float(scipy.stats.t.ppf(0.975, len(differences) - 1))
    t_stat = mean / sem if sem else math.inf
    return {
        "differences": differences,
        "mean_difference": mean,
        "sample_sd": sd,
        "two_sided_95_ci": [mean - critical * sem, mean + critical * sem],
        "paired_t_statistic": t_stat,
        "paired_t_two_sided_p": float(2 * scipy.stats.t.sf(abs(t_stat), len(differences) - 1)) if sem else 0.0,
        "cohen_dz": mean / sd if sd else None,
        "hedges_gz": (mean / sd) * (1 - 3 / (4 * len(differences) - 5)) if sd else None,
        "positive_seeds": sum(value > 0 for value in differences),
        "negative_seeds": sum(value < 0 for value in differences),
        "exact_two_sided_sign_p": sign_p(differences),
    }


def analyze(rows: list[dict], seeds: list[int]) -> dict:
    conditions = ["realonly_standard", "duplicate", "realcutout_medium150", "unity_medium150"]
    contrasts = [
        ("unity_minus_duplicate", "unity_medium150", "duplicate"),
        ("unity_minus_realcutout", "unity_medium150", "realcutout_medium150"),
        ("unity_minus_realonly", "unity_medium150", "realonly_standard"),
        ("duplicate_minus_realonly", "duplicate", "realonly_standard"),
        ("realcutout_minus_realonly", "realcutout_medium150", "realonly_standard"),
    ]
    output = {}
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["domain"] == domain]
        cell_values = {}
        means = {}
        for condition in conditions:
            ordered = [next(row for row in domain_rows if row["condition"] == condition and row["seed"] == seed) for seed in seeds]
            cell_values[condition] = {metric: [float(row["vessel"][metric]) for row in ordered] for metric in ("ap50_95", "ap50", "ap75", "precision", "recall")}
            means[condition] = {metric: statistics.mean(values) for metric, values in cell_values[condition].items()}
        contrast_results = {}
        for name, left, right in contrasts:
            contrast_results[name] = {metric: paired(cell_values[left][metric], cell_values[right][metric]) for metric in ("ap50_95", "ap50", "ap75")}
        output[domain] = {"condition_values": cell_values, "condition_means": means, "contrasts": contrast_results}
    return output


def build_report(summary: dict) -> str:
    labels = {"xview_test": "xView test", "hrsc2016_ms": "HRSC2016-MS", "dior_public_mirror": "DIOR public mirror"}
    lines = [
        "# Ten-Seed Head-Only Robustness Expansion v46 / 10개 시드 강건성 확장 v46", "",
        "Primary reporting uses paired two-sided 95% confidence intervals across ten seeds. Frozen v13-v15 decisions are unchanged.", "",
    ]
    for domain, label in labels.items():
        block = summary["analysis"][domain]
        lines += [f"## {label}", "", "| Condition | Mean AP50-95 | Mean AP50 | Mean AP75 |", "|---|---:|---:|---:|"]
        for condition, values in block["condition_means"].items():
            lines.append(f"| {condition} | {values['ap50_95']:.6f} | {values['ap50']:.6f} | {values['ap75']:.6f} |")
        lines += ["", "| Contrast | AP50-95 difference | Two-sided 95% CI | p | dz | Positive seeds | Sign p |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, metrics in block["contrasts"].items():
            value = metrics["ap50_95"]; low, high = value["two_sided_95_ci"]
            lines.append(f"| {name} | {value['mean_difference']:+.6f} | [{low:+.6f}, {high:+.6f}] | {value['paired_t_two_sided_p']:.6g} | {value['cohen_dz']:+.3f} | {value['positive_seeds']}/10 | {value['exact_two_sided_sign_p']:.6f} |")
        lines.append("")
    lines += [
        "## Interpretation boundary / 해석 범위", "",
        "- RealOnly is not exposure matched to the 3,000-image conditions.",
        "- Unity-Duplicate and Unity-RealCutout are the exposure-matched content comparisons.",
        "- Feature support remains a candidate-selection signal, not an established independent cause or mediator.",
        "- HRSC and DIOR are fixed external evaluations, not independent preregistrations.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    study = json.loads(CONFIG.read_text(encoding="utf-8"))
    existing = json.loads(V45.read_text(encoding="utf-8"))
    seeds = [int(value) for value in study["all_seeds"]]
    new_seeds = [int(value) for value in study["new_seeds"]]
    conditions = list(study["conditions"])
    checkpoints = {(condition, seed): checkpoint(study, condition, seed) for condition in conditions for seed in seeds}
    missing = [str(path) for path in checkpoints.values() if not exact_epoch20(path)]
    if missing:
        raise RuntimeError(f"Evaluation locked; incomplete checkpoints: {missing}")
    OUTPUT.mkdir(parents=True)
    rows = [row for row in existing["raw_results"] if int(row["seed"]) in study["existing_seeds"]]
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 frozen v45 rows, found {len(rows)}")
    jobs = [(domain, condition, seed) for domain in DOMAINS for seed in new_seeds for condition in conditions]
    for ordinal, (domain, condition, seed) in enumerate(jobs, 1):
        print(f"V46 EVAL {ordinal}/{len(jobs)} {domain} {condition} seed={seed}", flush=True)
        rows.append(evaluate(domain, condition, seed, checkpoints[(condition, seed)]))
    summary = {
        "study": study["study"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(CONFIG), "protocol_sha256": sha256_file(PROTOCOL),
        "software": {"python": platform.python_version(), "ultralytics": ultralytics.__version__, "torch": torch.__version__, "scipy": scipy.__version__},
        "seeds": seeds, "historical_v13_v15_decisions_unchanged": True,
        "exposure_audit": existing["exposure_audit"],
        "analysis": analyze(rows, seeds), "raw_results": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "report_en_ko.md").write_text(build_report(summary), encoding="utf-8")
    print("V46 EVALUATION PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
