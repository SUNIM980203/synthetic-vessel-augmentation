#!/usr/bin/env python3
"""Frozen descriptive n/s/m comparison for the YOLO26m third-size follow-up."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yolo26m_factorial_common as common


N_STATS = common.ROOT / "scope_lr_factorial_followup_v1/08_statistics/scope_lr_factorial_statistics.json"
S_STATS = common.ROOT / "yolo26s_scope_lr_factorial_replication_v1/12_statistics/yolo26s_factorial_statistics.json"
M_STATS = common.STATISTICS_DIR / "yolo26m_factorial_statistics.json"
N_PER_SEED = common.ROOT / "scope_lr_factorial_followup_v1/08_statistics/scope_lr_factorial_per_seed_contrasts.csv"
S_PER_SEED = common.ROOT / "yolo26s_scope_lr_factorial_replication_v1/12_statistics/yolo26s_factorial_per_seed_contrasts.csv"
M_PER_SEED = common.STATISTICS_DIR / "yolo26m_factorial_per_seed_contrasts.csv"

SUMMARY_CSV = common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_scope_interaction_summary.csv"
PER_SEED_CSV = common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_per_seed_marginal_did.csv"
OUTPUT_JSON = common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_model_size_heterogeneity.json"
OUTPUT_MD = common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_model_size_heterogeneity.md"


def read_primary(path: Path) -> dict:
    payload = common.load_json(path)
    return payload["analysis"]["xview_test"]["vessel"]["ap50_95"]["marginal_did"]


def read_seed_values(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        row for row in rows
        if row["dataset"] == "xview_test"
        and row["metric_family"] == "vessel"
        and row["metric"] == "ap50_95"
    ]
    values = {int(row["seed"]): float(row["marginal_did"]) for row in selected}
    if sorted(values) != common.SEEDS:
        raise RuntimeError(f"paired seed set mismatch in {path}: {sorted(values)}")
    return values


def write_csv_exclusive(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def includes_zero(summary: dict) -> bool:
    low, high = map(float, summary["two_sided_95_ci"])
    return low <= 0.0 <= high


def direction(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def excludes_zero_in_observed_direction(summary: dict) -> bool:
    low, high = map(float, summary["two_sided_95_ci"])
    mean = float(summary["mean"])
    return (mean > 0.0 and low > 0.0) or (mean < 0.0 and high < 0.0)


def classify(models: dict[str, dict], oriented: dict[str, dict]) -> tuple[str, str]:
    # Frozen precedence: compatibility first, then agreement with n/s, then null/inconclusive, then residual.
    cross = (oriented["n_minus_s"], oriented["n_minus_m"], oriented["s_minus_m"])
    if all(includes_zero(item) for item in cross):
        return "CASE M5", "All three paired cross-model 95% CIs include zero; nominal directional differences should not be overstated."

    n_dir = direction(float(models["YOLO26n"]["mean"]))
    s_dir = direction(float(models["YOLO26s"]["mean"]))
    m_dir = direction(float(models["YOLO26m"]["mean"]))
    m_minus_s = oriented["m_minus_s"]
    m_minus_n = oriented["m_minus_n"]
    if m_dir != 0 and m_dir == n_dir and m_dir != s_dir and excludes_zero_in_observed_direction(m_minus_s):
        return "CASE M1", "YOLO26m agrees directionally with YOLO26n and differs from YOLO26s; the tested sizes show non-monotonic/configuration heterogeneity."
    if m_dir != 0 and m_dir == s_dir and m_dir != n_dir and excludes_zero_in_observed_direction(m_minus_n):
        return "CASE M2", "YOLO26m agrees directionally with YOLO26s and differs from YOLO26n; the nano result may mark a small-model boundary."
    if includes_zero(models["YOLO26m"]):
        return "CASE M3", "The YOLO26m interaction is near zero or statistically inconclusive under the frozen seed-level interval."
    return "CASE M4", "YOLO26m exhibits a remaining qualitatively different pattern under the frozen hierarchy."


def fmt(summary: dict) -> str:
    low, high = summary["two_sided_95_ci"]
    return f"{summary['mean']:+.6f} (95% CI [{low:+.6f}, {high:+.6f}]; {summary['positive_count']}/{summary['n']} positive)"


def main() -> int:
    common.assert_protocol_locked()
    for path in (SUMMARY_CSV, PER_SEED_CSV, OUTPUT_JSON, OUTPUT_MD):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen three-model output: {path}")
    manifest = common.load_json(common.ANALYSIS_MANIFEST)
    if manifest["roles"]["model_size_comparison"]["sha256"] != common.sha256_file(Path(__file__)):
        raise RuntimeError("three-model comparison script differs from the pre-run manifest")

    source_stats = {"YOLO26n": N_STATS, "YOLO26s": S_STATS, "YOLO26m": M_STATS}
    source_seed = {"YOLO26n": N_PER_SEED, "YOLO26s": S_PER_SEED, "YOLO26m": M_PER_SEED}
    models = {name: read_primary(path) for name, path in source_stats.items()}
    seed_values = {name: read_seed_values(path) for name, path in source_seed.items()}
    for name in models:
        expected = [float(models[name]["values"][index]) for index in range(len(common.SEEDS))]
        observed = [seed_values[name][seed] for seed in common.SEEDS]
        if expected != observed:
            raise RuntimeError(f"statistics/per-seed Marginal DiD mismatch for {name}")

    per_seed_rows = []
    differences = {key: [] for key in ("n_minus_s", "n_minus_m", "s_minus_m", "m_minus_s", "m_minus_n")}
    for seed in common.SEEDS:
        n_value = seed_values["YOLO26n"][seed]
        s_value = seed_values["YOLO26s"][seed]
        m_value = seed_values["YOLO26m"][seed]
        row = {
            "seed": seed,
            "yolo26n_marginal_did": n_value,
            "yolo26s_marginal_did": s_value,
            "yolo26m_marginal_did": m_value,
            "n_minus_s": n_value - s_value,
            "n_minus_m": n_value - m_value,
            "s_minus_m": s_value - m_value,
        }
        per_seed_rows.append(row)
        for key in ("n_minus_s", "n_minus_m", "s_minus_m"):
            differences[key].append(float(row[key]))
        differences["m_minus_s"].append(m_value - s_value)
        differences["m_minus_n"].append(m_value - n_value)
    oriented = {key: common.student_summary(values) for key, values in differences.items()}
    classification, interpretation = classify(models, oriented)

    summary_rows = []
    for name in ("YOLO26n", "YOLO26s", "YOLO26m"):
        value = models[name]
        summary_rows.append({
            "model_size": name,
            "mean_marginal_did": value["mean"],
            "sample_sd": value["sample_sd"],
            "median": value["median"],
            "minimum": value["minimum"],
            "maximum": value["maximum"],
            "ci95_low": value["two_sided_95_ci"][0],
            "ci95_high": value["two_sided_95_ci"][1],
            "positive_seeds": value["positive_count"],
            "n_seeds": value["n"],
            "source_statistics": str(source_stats[name]),
            "source_statistics_sha256": common.sha256_file(source_stats[name]),
        })
    write_csv_exclusive(SUMMARY_CSV, summary_rows)
    write_csv_exclusive(PER_SEED_CSV, per_seed_rows)

    payload = {
        "study": "prospectively frozen within-study YOLO26m third-model-size follow-up",
        "completed_utc": common.utc_now(),
        "primary_dataset": "xview_test",
        "primary_metric": "vessel_ap50_95",
        "models": models,
        "paired_seed_differences": {
            "n_minus_s": oriented["n_minus_s"],
            "n_minus_m": oriented["n_minus_m"],
            "s_minus_m": oriented["s_minus_m"],
        },
        "auxiliary_orientations_for_classification": {
            "m_minus_s": oriented["m_minus_s"],
            "m_minus_n": oriented["m_minus_n"],
        },
        "classification_precedence": ["M5", "M1", "M2", "M3", "M4"],
        "interpretation_class": classification,
        "interpretation": interpretation,
        "causal_boundary": "Paired differences are descriptive cross-model comparisons; model size is not randomized or interpreted causally, and no monotonic trend is fitted.",
        "sources": {
            name: {
                "statistics": str(source_stats[name]),
                "statistics_sha256": common.sha256_file(source_stats[name]),
                "per_seed": str(source_seed[name]),
                "per_seed_sha256": common.sha256_file(source_seed[name]),
            }
            for name in source_stats
        },
    }
    common.write_json_exclusive(OUTPUT_JSON, payload)
    OUTPUT_MD.write_text(
        "# YOLO26 n/s/m scope-interaction heterogeneity\n\n"
        "This is a prospectively frozen within-study three-model-size comparison on the previously used scene-disjoint xView test set. It is descriptive, not a causal model-size experiment.\n\n"
        f"- YOLO26n: {fmt(models['YOLO26n'])}.\n"
        f"- YOLO26s: {fmt(models['YOLO26s'])}.\n"
        f"- YOLO26m: {fmt(models['YOLO26m'])}.\n"
        f"- n − s: {fmt(oriented['n_minus_s'])}.\n"
        f"- n − m: {fmt(oriented['n_minus_m'])}.\n"
        f"- s − m: {fmt(oriented['s_minus_m'])}.\n\n"
        f"Frozen interpretation: **{classification}**. {interpretation}\n\n"
        "No pooled overall effect or monotonic size trend was estimated.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "classification": classification}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
