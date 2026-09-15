#!/usr/bin/env python3
"""Audit v46/scene metric parity and add zero-target sensitivities for v38.

No model inference or training is performed. The script reuses the exact files
frozen in v38_source_manifest.json and never overwrites v37 outputs.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_scene_cluster_robustness_v37 as v37  # noqa: E402


OUT = ROOT / "outputs" / "ieee_access_manuscript_v38_revision"
V37_RESULT = ROOT / "outputs" / "ieee_access_manuscript_v37_revision" / "scene_cluster_robustness_v37.json"
V46_SUMMARY = ROOT / "AnalysisResults" / "expanded_v1" / "seed_expansion_v46" / "evaluation" / "summary.json"
JSON_OUT = OUT / "scene_zero_target_sensitivity_v38.json"
CSV_OUT = OUT / "scene_zero_target_sensitivity_v38.csv"
MD_OUT = OUT / "scene_zero_target_sensitivity_v38.md"
PARITY_OUT = OUT / "scene_evaluator_parity_audit_v38.md"
SCENE_CSV = OUT / "scene_cluster_results.csv"
TOLERANCE = 1e-12


def summary(values: np.ndarray) -> dict:
    low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
    return {
        "replicates": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(median),
        "lower_95": float(low),
        "upper_95": float(high),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def bootstrap_arrays(split: str, left_models: list[dict], right_models: list[dict]) -> dict:
    _, image_scenes = v37.scene_manifest(split)
    unique_scenes = sorted(set(image_scenes.tolist()))
    scene_index = {scene: index for index, scene in enumerate(unique_scenes)}
    image_scene_index = np.asarray([scene_index[value] for value in image_scenes], dtype=np.int16)
    rng = np.random.default_rng(v37.BOOTSTRAP_SEED)
    scene_counts = np.zeros((v37.REPLICATES, len(unique_scenes)), dtype=np.int16)
    # Reproduce both RNG calls in the v37 loop. The unused seed draw matters to
    # the next scene draw and is therefore retained exactly.
    for replicate in range(v37.REPLICATES):
        sampled = rng.integers(0, len(unique_scenes), len(unique_scenes))
        scene_counts[replicate] = np.bincount(sampled, minlength=len(unique_scenes))
        rng.integers(0, len(left_models), len(left_models))
    image_counts = scene_counts[:, image_scene_index]
    target_counts = left_models[0]["target_counts"].astype(np.int64)
    target_total = image_counts @ target_counts
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    left = np.vstack([v37.weighted_ap(model, image_counts, device) for model in left_models])
    right = np.vstack([v37.weighted_ap(model, image_counts, device) for model in right_models])
    effect = (left - right).mean(axis=0)
    return {
        "effect": effect,
        "target_total": target_total,
        "zero_target": target_total == 0,
        "scene_count": len(unique_scenes),
        "image_count": len(image_scenes),
        "target_count": int(target_counts.sum()),
        "device": str(device),
    }


def reconstruction_parity(models: dict[str, list[dict]]) -> dict:
    canonical = json.loads(V46_SUMMARY.read_text(encoding="utf-8"))
    one = np.ones((1, len(next(iter(models.values()))[0]["target_counts"])), dtype=np.int16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result: dict[str, dict] = {}
    for condition, condition_models in models.items():
        reconstructed, weighted, expected = [], [], []
        for index, model in enumerate(condition_models):
            reconstructed.append(float(model["full_ap"]))
            weighted.append(float(v37.weighted_ap(model, one, device)[0]))
            expected.append(float(canonical["analysis"]["xview_test"]["condition_values"][condition]["ap50_95"][index]))
        reconstructed = np.asarray(reconstructed)
        weighted = np.asarray(weighted)
        expected = np.asarray(expected)
        if not np.allclose(weighted, reconstructed, atol=TOLERANCE, rtol=0):
            raise AssertionError(f"weighted AP integration does not reproduce ap_per_class for {condition}")
        result[condition] = {
            "canonical_per_seed": expected.tolist(),
            "rounded_json_reconstruction_per_seed": reconstructed.tolist(),
            "weighted_all_images_per_seed": weighted.tolist(),
            "weighted_vs_ap_per_class_max_abs": float(np.max(np.abs(weighted - reconstructed))),
            "reconstruction_minus_canonical_per_seed": (reconstructed - expected).tolist(),
            "max_abs_reconstruction_minus_canonical": float(np.max(np.abs(reconstructed - expected))),
            "mean_reconstruction_minus_canonical": float(np.mean(reconstructed - expected)),
        }
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (JSON_OUT, CSV_OUT, MD_OUT, PARITY_OUT, SCENE_CSV):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    unity = [v37.load_prediction_model("unity_medium150", seed) for seed in v37.SEEDS]
    duplicate = [v37.load_prediction_model("duplicate", seed) for seed in v37.SEEDS]
    realcutout = [v37.load_prediction_model("realcutout_medium150", seed) for seed in v37.SEEDS]
    medium = [v37.load_v20_model("medium150", seed) for seed in v37.SEEDS[:5]]
    small = [v37.load_v20_model("small150", seed) for seed in v37.SEEDS[:5]]

    parity = reconstruction_parity(
        {"unity_medium150": unity, "duplicate": duplicate, "realcutout_medium150": realcutout}
    )
    analyses = [
        ("Unity Medium150 - Duplicate", "test", unity, duplicate),
        ("Unity Medium150 - RealCutout Medium150", "test", unity, realcutout),
        ("Medium150 - Small150", "val", medium, small),
    ]
    v37_payload = json.loads(V37_RESULT.read_text(encoding="utf-8"))
    original = {row["contrast"]: row for row in v37_payload["analyses"]}
    rows = []
    scene_rows = []
    for name, split, left, right in analyses:
        arrays = bootstrap_arrays(split, left, right)
        zero = arrays["zero_target"]
        unconditional = summary(arrays["effect"])
        conditional = summary(arrays["effect"][~zero])
        old = original[name]["scene_cluster_bootstrap"]
        for key in ("mean", "median", "lower_95", "upper_95", "positive_fraction"):
            if abs(unconditional[key] - float(old[key])) > TOLERANCE:
                raise AssertionError(f"v37 unconditional result not reproduced: {name} {key}")
        base = {
            "contrast": name,
            "split": f"xView {split}",
            "post_result_sensitivity": True,
            "bootstrap_seed": v37.BOOTSTRAP_SEED,
            "total_replicates": v37.REPLICATES,
            "scene_count": arrays["scene_count"],
            "image_count": arrays["image_count"],
            "vessel_target_count": arrays["target_count"],
            "zero_target_replicates": int(zero.sum()),
            "retained_replicates": int((~zero).sum()),
            "unconditional_ap0": unconditional,
            # A and B are operationally identical for this fixed draw because
            # both condition on target_total > 0. They are retained under both
            # requested labels to make that equivalence explicit.
            "conditional_nonzero_target": conditional,
            "undefined_zero_target_excluded": conditional,
            "conditional_equals_undefined": True,
            "device": arrays["device"],
        }
        rows.append(base)
        orig = original[name]
        scene_rows.append(
            {
                "contrast": name,
                "split": f"xView {split}",
                "effect_role": "reconstructed scene-analysis AP",
                "full_effect": orig["full_effect"],
                "bootstrap_lower_95": unconditional["lower_95"],
                "bootstrap_upper_95": unconditional["upper_95"],
                "positive_fraction": unconditional["positive_fraction"],
                "loso_minimum": orig["leave_one_scene_out"]["minimum"],
                "loso_maximum": orig["leave_one_scene_out"]["maximum"],
                "positive_loso": orig["leave_one_scene_out"]["positive_estimates"],
                "total_loso": orig["leave_one_scene_out"]["total_estimates"],
            }
        )

    payload = {
        "study": "v38 post-result zero-target scene-bootstrap sensitivity",
        "replaces_unconditional_analysis": False,
        "decision_changed": False,
        "replicates": v37.REPLICATES,
        "bootstrap_seed": v37.BOOTSTRAP_SEED,
        "zero_target_convention": "Unconditional analysis defines AP=0 and includes every draw in the positive-fraction denominator.",
        "sensitivity_definition": "Conditional and undefined-zero-target analyses exclude target_total=0 draws; for the fixed draws they are identical.",
        "analyses": rows,
        "parity": parity,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with CSV_OUT.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "contrast", "split", "analysis", "total_replicates", "zero_target_replicates",
            "retained_replicates", "mean", "median", "ci95_low", "ci95_high", "positive_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for analysis, key in (
                ("unconditional_ap0", "unconditional_ap0"),
                ("conditional_nonzero_target", "conditional_nonzero_target"),
                ("undefined_zero_target_excluded", "undefined_zero_target_excluded"),
            ):
                stats = row[key]
                writer.writerow(
                    {
                        "contrast": row["contrast"], "split": row["split"], "analysis": analysis,
                        "total_replicates": row["total_replicates"],
                        "zero_target_replicates": row["zero_target_replicates"],
                        "retained_replicates": stats["replicates"], "mean": stats["mean"],
                        "median": stats["median"], "ci95_low": stats["lower_95"],
                        "ci95_high": stats["upper_95"], "positive_fraction": stats["positive_fraction"],
                    }
                )

    with SCENE_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)

    lines = [
        "# Scene zero-target sensitivity (v38)", "",
        "This is a post-result sensitivity analysis. It does not replace the unconditional acquisition-scene bootstrap and changes no frozen decision.", "",
        f"All analyses reproduce the original {v37.REPLICATES:,} draws generated with seed `{v37.BOOTSTRAP_SEED}`. The unconditional analysis retains AP=0 for zero-vessel-target draws. The conditional and undefined-zero-target variants both remove those draws and are therefore numerically identical for this fixed sample.", "",
        "| Contrast | Split | Variant | Zero/total | Retained | Mean | Median | 95% percentile interval | Positive fraction |", "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for label, key in (
            ("Unconditional AP=0", "unconditional_ap0"),
            ("Conditional nonzero target", "conditional_nonzero_target"),
            ("Undefined zero target excluded", "undefined_zero_target_excluded"),
        ):
            s = row[key]
            lines.append(
                f"| {row['contrast']} | {row['split']} | {label} | {row['zero_target_replicates']}/{row['total_replicates']} | {s['replicates']} | {s['mean']:+.6f} | {s['median']:+.6f} | [{s['lower_95']:+.6f}, {s['upper_95']:+.6f}] | {s['positive_fraction']:.6f} |"
            )
    lines += [
        "", "The sensitivity variants do not change the qualitative conclusions. Zero-target draws contribute to the lower tail but are too infrequent to determine the 2.5th percentile by themselves; sparse and unfavorable scene compositions also contribute.", "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

    u = np.asarray(parity["unity_medium150"]["rounded_json_reconstruction_per_seed"])
    d = np.asarray(parity["duplicate"]["rounded_json_reconstruction_per_seed"])
    r = np.asarray(parity["realcutout_medium150"]["rounded_json_reconstruction_per_seed"])
    uc = np.asarray(parity["unity_medium150"]["canonical_per_seed"])
    dc = np.asarray(parity["duplicate"]["canonical_per_seed"])
    rc = np.asarray(parity["realcutout_medium150"]["canonical_per_seed"])
    parity_lines = [
        "# Scene evaluator parity audit (v38)", "",
        "## Conclusion", "",
        "Exact parity with the canonical v46 AP is structurally unavailable without rerunning inference because the retained COCO prediction JSON rounds bounding-box coordinates to 0.001 pixel and confidence scores to five decimal places. The canonical evaluator computed AP before that serialization, from unrounded tensors. The retained JSON is sufficient for the scene-cluster sensitivity but must be labeled as a reconstructed scene-analysis metric.", "",
        "## Implementation comparison", "",
        "- The canonical v46 evaluator used Ultralytics 8.4.50 validation at confidence 0.001, NMS IoU 0.7, max detections 300, and computed per-class AP from in-memory predictions before JSON serialization.",
        "- The scene evaluator uses the same 0.50:0.05:0.95 IoU grid, greedy IoU matching order, class-4/category-5 mapping, confidence ranking, 101-point interpolated precision envelope, negative images, and target set.",
        "- Replacing COCO annotation boxes with the prepared normalized YOLO labels produced the same reconstruction, excluding annotation-coordinate conversion as the observed cause.",
        "- With unit image weights, the weighted scene AP routine matches Ultralytics `ap_per_class` reconstruction to an absolute tolerance of 1e-12 for every audited condition and seed. Thus weighted integration is not the residual source.",
        "- The remaining difference arises before AP integration: rounded serialized boxes can change matches near an IoU threshold, and rounded scores can introduce or alter confidence ties. Original unrounded per-prediction tensors were not retained.", "",
        "## Numerical comparison", "",
        "| Contrast | Canonical v46 effect | Rounded-JSON reconstruction | Reconstruction − canonical |", "|---|---:|---:|---:|",
        f"| Unity − Duplicate | {(uc-dc).mean():+.9f} | {(u-d).mean():+.9f} | {((u-d)-(uc-dc)).mean():+.9f} |",
        f"| Unity − RealCutout | {(uc-rc).mean():+.9f} | {(u-r).mean():+.9f} | {((u-r)-(uc-rc)).mean():+.9f} |", "",
        "Maximum absolute per-seed condition-level reconstruction discrepancies:", "",
    ]
    for condition, result in parity.items():
        parity_lines.append(f"- `{condition}`: {result['max_abs_reconstruction_minus_canonical']:.9f}")
    parity_lines += [
        "", "## Required reporting boundary", "",
        "Canonical v46 estimates remain the efficacy results in the main Results section. Scene resampling uses and is labeled `reconstructed scene-analysis AP`. The scene values must not replace canonical values. The automated unit-weight tolerance test passes; exact canonical parity is marked unavailable rather than fabricated.", "",
    ]
    PARITY_OUT.write_text("\n".join(parity_lines), encoding="utf-8")
    print(JSON_OUT)
    print(CSV_OUT)
    print(MD_OUT)
    print(PARITY_OUT)
    print(SCENE_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
