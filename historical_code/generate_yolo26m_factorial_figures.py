#!/usr/bin/env python3
"""Generate the five prespecified source-backed YOLO26m/n/s/m figures."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yolo26m_factorial_common as common


FIGURES = {
    "M1": common.FIGURES_DIR / "figure_M1_yolo26m_unity_effects",
    "M2": common.FIGURES_DIR / "figure_M2_yolo26m_did",
    "M3": common.FIGURES_DIR / "figure_M3_yolo26_n_s_m_marginal_did",
    "M4": common.FIGURES_DIR / "figure_M4_yolo26_n_s_m_per_seed",
    "M5": common.FIGURES_DIR / "figure_M5_yolo26_n_s_m_scene_sensitivity",
}
FIGURE_MANIFEST = common.FIGURES_DIR / "yolo26m_factorial_figure_manifest.json"
N_SCENE = common.ROOT / "scope_lr_factorial_followup_v1/09_scene_sensitivity/scope_lr_factorial_scene_sensitivity.json"
S_SCENE = common.ROOT / "yolo26s_scope_lr_factorial_replication_v1/13_scene_sensitivity/yolo26s_factorial_scene_sensitivity.json"
M_SCENE = common.SCENE_DIR / "yolo26m_factorial_scene_sensitivity.json"


PLOT_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
KIND = HERE.name.split("_")[1]
with (HERE / "figure_source.csv").open("r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

titles = {
    "M1": "YOLO26m Unity − Duplicate effects by scope and LR",
    "M2": "YOLO26m fixed-LR and Marginal DiD",
    "M3": "YOLO26 model-size comparison of Marginal DiD",
    "M4": "Per-seed Marginal DiD across YOLO26 n/s/m",
    "M5": "xView scene sensitivity across YOLO26 n/s/m",
}
fig, ax = plt.subplots(figsize=(8.2, 4.8))
limits = []
if KIND in {"M1", "M2", "M3"}:
    label_key = {"M1": "label", "M2": "contrast", "M3": "model_size"}[KIND]
    labels = [row[label_key] for row in rows]
    means = np.asarray([float(row["mean"]) for row in rows])
    lows = np.asarray([float(row["ci95_low"]) for row in rows])
    highs = np.asarray([float(row["ci95_high"]) for row in rows])
    y = np.arange(len(rows))
    ax.errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="o", color="#2457A7", capsize=4, lw=1.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    limits.extend(lows.tolist() + highs.tolist())
    ax.set_xlabel("AP50–95 contrast")
elif KIND == "M4":
    models = ["YOLO26n", "YOLO26s", "YOLO26m"]
    keys = ["yolo26n_marginal_did", "yolo26s_marginal_did", "yolo26m_marginal_did"]
    x = np.arange(3)
    for row in rows:
        values = np.asarray([float(row[key]) for key in keys])
        ax.plot(x, values, marker="o", alpha=0.55, lw=1.0, label=str(row["seed"]))
        limits.extend(values.tolist())
    ax.set_xticks(x, models)
    ax.set_ylabel("Marginal DiD (AP50–95)")
    ax.legend(title="Seed", fontsize=7, ncol=2, loc="best")
elif KIND == "M5":
    labels = [row["model_size"] for row in rows]
    y = np.arange(len(rows))
    boot = np.asarray([float(row["bootstrap_mean"]) for row in rows])
    low = np.asarray([float(row["bootstrap_ci_low"]) for row in rows])
    high = np.asarray([float(row["bootstrap_ci_high"]) for row in rows])
    full = np.asarray([float(row["full_sample"]) for row in rows])
    loso_low = np.asarray([float(row["loso_minimum"]) for row in rows])
    loso_high = np.asarray([float(row["loso_maximum"]) for row in rows])
    ax.hlines(y, loso_low, loso_high, color="#B8B8B8", lw=6, label="LOSO range")
    ax.errorbar(boot, y, xerr=np.vstack([boot - low, high - boot]), fmt="o", color="#2457A7", capsize=4, label="Scene bootstrap mean and 95% interval")
    ax.scatter(full, y, marker="x", s=65, color="#B13B2E", label="Full sample")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Marginal DiD (AP50–95)")
    ax.legend(fontsize=8, loc="best")
    limits.extend(low.tolist() + high.tolist() + full.tolist() + loso_low.tolist() + loso_high.tolist())

ax.axvline(0.0, color="black", lw=1.0, ls="--") if KIND in {"M1", "M2", "M3", "M5"} else ax.axhline(0.0, color="black", lw=1.0, ls="--")
max_abs = max(abs(value) for value in limits) if limits else 1.0
max_abs = max_abs * 1.12 if max_abs > 0 else 1.0
if KIND in {"M1", "M2", "M3", "M5"}:
    ax.set_xlim(-max_abs, max_abs)
else:
    ax.set_ylim(-max_abs, max_abs)
ax.set_title(titles[KIND])
ax.grid(axis="x" if KIND != "M4" else "y", alpha=0.25)
fig.tight_layout()
fig.savefig(HERE / "figure.pdf", bbox_inches="tight")
fig.savefig(HERE / "figure.png", dpi=300, bbox_inches="tight")
plt.close(fig)
'''


def write_source(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "figure_source.csv"
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def scene_row(model_size: str, path: Path) -> dict:
    scene = common.load_json(path)
    bootstrap = scene["scene_bootstrap_including_zero_target"]
    loso = scene["leave_one_scene_out"]
    return {
        "model_size": model_size,
        "full_sample": scene["full_sample"]["marginal_did"],
        "bootstrap_mean": bootstrap["mean"],
        "bootstrap_ci_low": bootstrap["percentile_95_interval"][0],
        "bootstrap_ci_high": bootstrap["percentile_95_interval"][1],
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "zero_target_replicates": scene["zero_target_replicates"],
        "loso_minimum": loso["minimum"],
        "loso_maximum": loso["maximum"],
        "loso_positive_count": loso["positive_count"],
        "loso_n": loso["n"],
        "source_scene_json": str(path),
        "source_scene_sha256": common.sha256_file(path),
    }


def main() -> int:
    common.assert_protocol_locked()
    if FIGURE_MANIFEST.exists() or any(path.exists() for path in FIGURES.values()):
        raise FileExistsError("refusing to overwrite figure outputs")
    manifest = common.load_json(common.ANALYSIS_MANIFEST)
    if manifest["roles"]["figure_generator"]["sha256"] != common.sha256_file(Path(__file__)):
        raise RuntimeError("figure generator differs from the pre-run manifest")
    stats = common.load_json(common.STATISTICS_DIR / "yolo26m_factorial_statistics.json")
    comparison = common.load_json(common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_model_size_heterogeneity.json")
    block = stats["analysis"]["xview_test"]["vessel"]["ap50_95"]

    m1 = []
    for scope, scope_label in (("head_only", "Head-only"), ("full_network", "Full-network")):
        for lr in ("1e-4", "2e-4"):
            value = block["unity_minus_duplicate"][scope][lr]
            m1.append({"label": f"{scope_label} @ {lr}", "mean": value["mean"], "ci95_low": value["two_sided_95_ci"][0], "ci95_high": value["two_sided_95_ci"][1]})
    m2 = []
    for label, value in (("DiD @ 1e-4", block["fixed_lr_did"]["1e-4"]), ("DiD @ 2e-4", block["fixed_lr_did"]["2e-4"]), ("Marginal DiD", block["marginal_did"])):
        m2.append({"contrast": label, "mean": value["mean"], "ci95_low": value["two_sided_95_ci"][0], "ci95_high": value["two_sided_95_ci"][1]})
    m3 = []
    for model_size in ("YOLO26n", "YOLO26s", "YOLO26m"):
        value = comparison["models"][model_size]
        m3.append({"model_size": model_size, "mean": value["mean"], "ci95_low": value["two_sided_95_ci"][0], "ci95_high": value["two_sided_95_ci"][1]})
    with (common.MODEL_COMPARISON_DIR / "yolo26_n_s_m_per_seed_marginal_did.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        m4 = list(csv.DictReader(stream))
    m5 = [scene_row("YOLO26n", N_SCENE), scene_row("YOLO26s", S_SCENE), scene_row("YOLO26m", M_SCENE)]
    sources = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5}

    artifacts = {}
    for label, directory in FIGURES.items():
        source_path = write_source(directory, sources[label])
        script = directory / "figure_script.py"
        script.write_text(PLOT_SCRIPT, encoding="utf-8")
        subprocess.run([sys.executable, str(script)], check=True, cwd=common.ROOT)
        expected = (source_path, script, directory / "figure.pdf", directory / "figure.png")
        for artifact in expected:
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"missing figure artifact: {artifact}")
        artifacts[label] = {
            "directory": str(directory),
            "csv": str(source_path), "csv_sha256": common.sha256_file(source_path),
            "python": str(script), "python_sha256": common.sha256_file(script),
            "pdf": str(directory / "figure.pdf"), "pdf_sha256": common.sha256_file(directory / "figure.pdf"),
            "png": str(directory / "figure.png"), "png_sha256": common.sha256_file(directory / "figure.png"),
        }
    common.write_json_exclusive(FIGURE_MANIFEST, {"completed_utc": common.utc_now(), "status": "PASS", "figures": artifacts})
    print("YOLO26M FIGURES PASS: M1-M5 CSV/Python/PDF/PNG", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
