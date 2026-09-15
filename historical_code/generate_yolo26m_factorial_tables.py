#!/usr/bin/env python3
"""Generate source-backed YOLO26m and n/s/m factorial tables."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import yolo26m_factorial_common as common


M_ABSOLUTE = common.TABLES_DIR / "yolo26m_factorial_absolute_AP.csv"
N_ABSOLUTE = common.ROOT / "scope_lr_factorial_followup_v1/10_tables/scope_lr_factorial_absolute_AP.csv"
S_ABSOLUTE = common.ROOT / "yolo26s_scope_lr_factorial_replication_v1/15_tables/yolo26s_factorial_absolute_AP.csv"
COMBINED_ABSOLUTE = common.TABLES_DIR / "yolo26_n_s_m_absolute_AP.csv"
TABLE_NAMES = {
    "A": common.TABLES_DIR / "table_A_yolo26m_absolute_factorial_AP.csv",
    "B": common.TABLES_DIR / "table_B_yolo26m_unity_minus_duplicate_effects.csv",
    "C": common.TABLES_DIR / "table_C_yolo26m_fixed_lr_scope_did.csv",
    "D": common.TABLES_DIR / "table_D_yolo26m_marginal_did_and_three_way.csv",
    "E": common.TABLES_DIR / "table_E_yolo26m_scene_bootstrap_and_loso.csv",
    "F": common.TABLES_DIR / "table_F_yolo26_n_s_m_marginal_did.csv",
    "G": common.TABLES_DIR / "table_G_yolo26_n_s_m_paired_differences.csv",
}
TABLES_MD = common.TABLES_DIR / "yolo26m_scope_lr_factorial_tables.md"
MODEL_MD = common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_interpretation.md"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_absolute(path: Path, model_size: str) -> list[dict]:
    rows = read_csv(path)
    for row in rows:
        row["model_size"] = model_size
    fields = ["model_size", "dataset", "scope", "learning_rate", "learning_rate_label", "condition", "metric_family", "metric", "n_seeds", "mean", "sample_sd"]
    normalized = [{key: row[key] for key in fields} for row in rows]
    if len(normalized) != 144:
        raise RuntimeError(f"expected 144 absolute-AP summaries for {model_size}, found {len(normalized)}")
    return normalized


def main() -> int:
    common.assert_protocol_locked()
    outputs = [*TABLE_NAMES.values(), COMBINED_ABSOLUTE, TABLES_MD, MODEL_MD]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite table or interpretation output")
    manifest = common.load_json(common.ANALYSIS_MANIFEST)
    if manifest["roles"]["table_generator"]["sha256"] != common.sha256_file(Path(__file__)):
        raise RuntimeError("table generator differs from the pre-run manifest")

    stats = common.load_json(common.STATISTICS_DIR / "yolo26m_factorial_statistics.json")
    scene = common.load_json(common.SCENE_DIR / "yolo26m_factorial_scene_sensitivity.json")
    comparison = common.load_json(common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_model_size_heterogeneity.json")

    shutil.copyfile(M_ABSOLUTE, TABLE_NAMES["A"])
    combined = normalize_absolute(N_ABSOLUTE, "YOLO26n") + normalize_absolute(S_ABSOLUTE, "YOLO26s") + normalize_absolute(M_ABSOLUTE, "YOLO26m")
    write_csv(COMBINED_ABSOLUTE, combined)

    effects, fixed, marginal = [], [], []
    for domain, domain_block in stats["analysis"].items():
        for family, family_block in domain_block.items():
            for metric, block in family_block.items():
                for scope, scope_block in block["unity_minus_duplicate"].items():
                    for lr_label, summary in scope_block.items():
                        effects.append({
                            "dataset": domain, "metric_family": family, "metric": metric,
                            "scope": scope, "learning_rate_label": lr_label,
                            "mean_effect": summary["mean"], "sample_sd": summary["sample_sd"],
                            "ci95_low": summary["two_sided_95_ci"][0], "ci95_high": summary["two_sided_95_ci"][1],
                            "positive_seeds": summary["positive_count"], "n_seeds": summary["n"],
                            "t_p_two_sided": summary["one_sample_t_two_sided_p"],
                            "sign_p_two_sided": summary["exact_two_sided_sign_p"],
                        })
                for lr_label, summary in block["fixed_lr_did"].items():
                    fixed.append({
                        "dataset": domain, "metric_family": family, "metric": metric,
                        "learning_rate_label": lr_label, "mean_did": summary["mean"],
                        "sample_sd": summary["sample_sd"], "ci95_low": summary["two_sided_95_ci"][0],
                        "ci95_high": summary["two_sided_95_ci"][1], "positive_seeds": summary["positive_count"],
                        "n_seeds": summary["n"], "t_p_two_sided": summary["one_sample_t_two_sided_p"],
                        "sign_p_two_sided": summary["exact_two_sided_sign_p"],
                    })
                for label, summary in (
                    ("marginal_did", block["marginal_did"]), ("three_way", block["three_way"]),
                    ("lrmod_head", block["within_scope_lr_moderation"]["head_only"]),
                    ("lrmod_full", block["within_scope_lr_moderation"]["full_network"]),
                ):
                    marginal.append({
                        "dataset": domain, "metric_family": family, "metric": metric, "contrast": label,
                        "mean": summary["mean"], "sample_sd": summary["sample_sd"], "median": summary["median"],
                        "minimum": summary["minimum"], "maximum": summary["maximum"],
                        "ci95_low": summary["two_sided_95_ci"][0], "ci95_high": summary["two_sided_95_ci"][1],
                        "positive_seeds": summary["positive_count"], "n_seeds": summary["n"],
                        "t_p_two_sided": summary["one_sample_t_two_sided_p"],
                        "sign_p_two_sided": summary["exact_two_sided_sign_p"],
                    })
    write_csv(TABLE_NAMES["B"], effects)
    write_csv(TABLE_NAMES["C"], fixed)
    write_csv(TABLE_NAMES["D"], marginal)

    scene_rows = [
        {"analysis": "full_sample", "estimate": scene["full_sample"]["marginal_did"], "lower": "", "upper": "", "n": 1, "positive": "", "zero_target_replicates": scene["zero_target_replicates"]},
        {"analysis": "scene_bootstrap_including_zero_target", "estimate": scene["scene_bootstrap_including_zero_target"]["mean"], "lower": scene["scene_bootstrap_including_zero_target"]["percentile_95_interval"][0], "upper": scene["scene_bootstrap_including_zero_target"]["percentile_95_interval"][1], "n": scene["scene_bootstrap_including_zero_target"]["n"], "positive": scene["scene_bootstrap_including_zero_target"]["positive_fraction"], "zero_target_replicates": scene["zero_target_replicates"]},
        {"analysis": "leave_one_scene_out_range", "estimate": scene["full_sample"]["marginal_did"], "lower": scene["leave_one_scene_out"]["minimum"], "upper": scene["leave_one_scene_out"]["maximum"], "n": scene["leave_one_scene_out"]["n"], "positive": scene["leave_one_scene_out"]["positive_count"], "zero_target_replicates": scene["zero_target_replicates"]},
    ]
    write_csv(TABLE_NAMES["E"], scene_rows)

    summary_rows = []
    for model_size in ("YOLO26n", "YOLO26s", "YOLO26m"):
        value = comparison["models"][model_size]
        summary_rows.append({
            "model_size": model_size, "mean_marginal_did": value["mean"], "sample_sd": value["sample_sd"],
            "ci95_low": value["two_sided_95_ci"][0], "ci95_high": value["two_sided_95_ci"][1],
            "positive_seeds": value["positive_count"], "n_seeds": value["n"],
        })
    write_csv(TABLE_NAMES["F"], summary_rows)
    pair_rows = []
    for label, value in comparison["paired_seed_differences"].items():
        pair_rows.append({
            "contrast": label, "mean_difference": value["mean"], "sample_sd": value["sample_sd"],
            "ci95_low": value["two_sided_95_ci"][0], "ci95_high": value["two_sided_95_ci"][1],
            "positive_seeds": value["positive_count"], "n_seeds": value["n"],
        })
    write_csv(TABLE_NAMES["G"], pair_rows)

    primary = stats["analysis"]["xview_test"]["vessel"]["ap50_95"]["marginal_did"]
    lines = [
        "# YOLO26m scope × learning-rate factorial tables", "",
        "All tables are generated directly from retained metric, prediction, and authoritative n/s result artifacts.", "",
        "## Designated YOLO26m endpoint", "",
        f"xView-test vessel AP50–95 Marginal DiD: {primary['mean']:+.6f} "
        f"(two-sided Student-t 95% CI [{primary['two_sided_95_ci'][0]:+.6f}, {primary['two_sided_95_ci'][1]:+.6f}]); "
        f"positive seeds {primary['positive_count']}/10; t-test p={primary['one_sample_t_two_sided_p']:.6g}; "
        f"exact sign-test p={primary['exact_two_sided_sign_p']:.6g}.", "",
        f"Three-model frozen interpretation: **{comparison['interpretation_class']}**.", "",
        f"Combined absolute table: `{COMBINED_ABSOLUTE.name}`.", "",
    ]
    lines.extend(f"- Table {label}: `{path.name}`" for label, path in TABLE_NAMES.items())
    TABLES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    MODEL_MD.write_text(
        "# YOLO26 n/s/m interpretation\n\n"
        "The three model sizes are compared descriptively using identical seed labels; model size is not randomized and no monotonic trend is fitted.\n\n"
        f"Frozen class: **{comparison['interpretation_class']}**. {comparison['interpretation']}\n\n"
        "The result is bounded to the three tested YOLO26 configurations, retained data, two fixed LRs, and the 20-epoch recipe.\n",
        encoding="utf-8",
    )
    print("YOLO26M FACTORIAL TABLES PASS: n/s/m combined outputs complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
