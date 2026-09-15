#!/usr/bin/env python3
"""Frozen YOLO26s factorial contrasts, intervals, and exact sign tests."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yolo26s_factorial_common as common


OUTPUT_JSON = common.STATISTICS_DIR / "yolo26s_factorial_statistics.json"
PER_SEED_CSV = common.STATISTICS_DIR / "yolo26s_factorial_per_seed_contrasts.csv"
RAW_CSV = common.METRICS_DIR / "yolo26s_factorial_metrics_long.csv"
ABSOLUTE_CSV = common.TABLES_DIR / "yolo26s_factorial_absolute_AP.csv"


def load_rows() -> list[dict]:
    evaluation = common.load_json(common.EVALUATION_LEDGER)
    if evaluation.get("status") != "PASS" or evaluation.get("completed_jobs") != 240:
        raise RuntimeError("factorial analysis is locked until all 240 evaluations pass")
    rows = []
    for job in evaluation["jobs"]:
        metric = common.load_json(Path(job["metrics"]))
        for family in common.FAMILIES:
            for metric_name in common.METRICS:
                rows.append({
                    "model_size": "YOLO26s",
                    "domain": job["domain"],
                    "condition": job["condition"],
                    "scope": job["scope"],
                    "learning_rate": float(job["learning_rate"]),
                    "learning_rate_label": job["learning_rate_label"],
                    "seed": int(job["seed"]),
                    "family": family,
                    "metric": metric_name,
                    "value": float(metric[family][metric_name]),
                    "checkpoint_sha256": job["checkpoint_sha256"],
                    "metrics_sha256": job["metrics_sha256"],
                })
    if len(rows) != 240 * 2 * 3:
        raise RuntimeError(f"unexpected long metric row count: {len(rows)}")
    return rows


def write_csv_exclusive(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_absolute(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["domain"], row["scope"], row["learning_rate"], row["learning_rate_label"], row["condition"], row["family"], row["metric"])
        grouped[key].append(row["value"])
    output = []
    for key, values in sorted(grouped.items()):
        domain, scope, lr, lr_label, condition, family, metric = key
        if len(values) != 10:
            raise RuntimeError(f"absolute cell does not contain 10 seeds: {key}")
        output.append({
            "model_size": "YOLO26s",
            "dataset": domain,
            "scope": scope,
            "learning_rate": lr,
            "learning_rate_label": lr_label,
            "condition": condition,
            "metric_family": family,
            "metric": metric,
            "n_seeds": len(values),
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        })
    return output


def analyze_block(rows: list[dict], domain: str, family: str, metric: str) -> tuple[dict, list[dict]]:
    selected = [row for row in rows if row["domain"] == domain and row["family"] == family and row["metric"] == metric]
    lookup = {
        (row["condition"], row["scope"], float(row["learning_rate"]), int(row["seed"])): float(row["value"])
        for row in selected
    }
    expected = len(common.CONDITIONS) * len(common.SCOPES) * len(common.LEARNING_RATES) * len(common.SEEDS)
    if len(lookup) != expected:
        raise RuntimeError(f"factorial block is not complete: {domain} {family} {metric}: {len(lookup)}/{expected}")
    per_seed = []
    fixed = {lr: [] for lr in common.LEARNING_RATES}
    marginal, three_way, lrmod_head, lrmod_full = [], [], [], []
    scope_effects = {(scope, lr): [] for scope in common.SCOPES for lr in common.LEARNING_RATES}
    for seed in common.SEEDS:
        delta = {}
        did = {}
        for lr in common.LEARNING_RATES:
            for scope in common.SCOPES:
                value = lookup[("unity_medium150", scope, lr, seed)] - lookup[("duplicate", scope, lr, seed)]
                delta[(scope, lr)] = value
                scope_effects[(scope, lr)].append(value)
            did[lr] = delta[("head_only", lr)] - delta[("full_network", lr)]
            fixed[lr].append(did[lr])
        marginal_value = 0.5 * (did[0.0001] + did[0.0002])
        three_value = did[0.0002] - did[0.0001]
        head_mod = delta[("head_only", 0.0002)] - delta[("head_only", 0.0001)]
        full_mod = delta[("full_network", 0.0002)] - delta[("full_network", 0.0001)]
        marginal.append(marginal_value)
        three_way.append(three_value)
        lrmod_head.append(head_mod)
        lrmod_full.append(full_mod)
        per_seed.append({
            "model_size": "YOLO26s",
            "dataset": domain,
            "metric_family": family,
            "metric": metric,
            "seed": seed,
            "delta_head_1e-4": delta[("head_only", 0.0001)],
            "delta_full_1e-4": delta[("full_network", 0.0001)],
            "did_1e-4": did[0.0001],
            "delta_head_2e-4": delta[("head_only", 0.0002)],
            "delta_full_2e-4": delta[("full_network", 0.0002)],
            "did_2e-4": did[0.0002],
            "marginal_did": marginal_value,
            "three_way": three_value,
            "lrmod_head": head_mod,
            "lrmod_full": full_mod,
        })
    result = {
        "unity_minus_duplicate": {
            scope: {common.LR_LABELS[lr]: common.student_summary(scope_effects[(scope, lr)]) for lr in common.LEARNING_RATES}
            for scope in common.SCOPES
        },
        "fixed_lr_did": {common.LR_LABELS[lr]: common.student_summary(fixed[lr]) for lr in common.LEARNING_RATES},
        "marginal_did": common.student_summary(marginal),
        "three_way": common.student_summary(three_way),
        "within_scope_lr_moderation": {
            "head_only": common.student_summary(lrmod_head),
            "full_network": common.student_summary(lrmod_full),
        },
    }
    return result, per_seed


def main() -> int:
    common.assert_protocol_locked()
    for path in (OUTPUT_JSON, PER_SEED_CSV, RAW_CSV, ABSOLUTE_CSV):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen analysis output: {path}")
    manifest = common.load_json(common.ANALYSIS_MANIFEST)
    if manifest["roles"]["factorial_contrast_script"]["sha256"] != common.sha256_file(Path(__file__)):
        raise RuntimeError("factorial analysis script differs from pre-run manifest")
    rows = load_rows()
    write_csv_exclusive(RAW_CSV, list(rows[0]), rows)
    absolute = summarize_absolute(rows)
    write_csv_exclusive(ABSOLUTE_CSV, list(absolute[0]), absolute)
    analysis, per_seed_rows = {}, []
    for domain in common.DOMAINS:
        analysis[domain] = {}
        for family in common.FAMILIES:
            analysis[domain][family] = {}
            for metric in common.METRICS:
                block, per_seed = analyze_block(rows, domain, family, metric)
                analysis[domain][family][metric] = block
                per_seed_rows.extend(per_seed)
    write_csv_exclusive(PER_SEED_CSV, list(per_seed_rows[0]), per_seed_rows)
    primary = analysis["xview_test"]["vessel"]["ap50_95"]["marginal_did"]
    payload = {
        "study": "fresh YOLO26s condition x scope x learning-rate factorial",
        "completed_utc": common.utc_now(),
        "primary_dataset": "xview_test",
        "primary_metric": "vessel_ap50_95",
        "primary_estimand": "mean across seeds of 0.5*(DiD_1e-4 + DiD_2e-4)",
        "primary_decision_rule": "two-sided Student-t 95% CI excludes zero on the positive side",
        "primary_positive_interaction_supported": primary["two_sided_95_ci"][0] > 0.0,
        "analysis": analysis,
        "source_metrics_long_csv": str(RAW_CSV),
        "source_metrics_long_csv_sha256": common.sha256_file(RAW_CSV),
        "per_seed_contrasts_csv": str(PER_SEED_CSV),
        "per_seed_contrasts_csv_sha256": common.sha256_file(PER_SEED_CSV),
        "absolute_ap_csv": str(ABSOLUTE_CSV),
        "absolute_ap_csv_sha256": common.sha256_file(ABSOLUTE_CSV),
    }
    common.write_json_exclusive(OUTPUT_JSON, payload)
    print(json.dumps({"primary": primary, "decision": payload["primary_positive_interaction_supported"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
