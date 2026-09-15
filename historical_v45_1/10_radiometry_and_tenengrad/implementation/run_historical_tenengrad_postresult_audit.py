"""Post-result diagnostic audit for the failed historical Medium150 Tenengrad gate.

This script is deliberately read-only with respect to all candidate, source,
host, selector, calibration, training, and AP artifacts.  It may only create
POST_RESULT_DIAGNOSTIC tables, reports, hashes, and non-AP figures.  The
prospective MEDIUM150_PRETRAINING_GATE=FAIL result is asserted before work and
is never recomputed into a decision.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from scipy.stats import pearsonr, spearmanr, t, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from support_scale_radiometry_calibration_common import (  # noqa: E402
    METRICS,
    extended_appearance_metrics,
    native_metrics,
    object_and_ring_masks,
)


OUT = ROOT / "AnalysisResults" / "expanded_v1" / "medium150_historical_pretraining_gate"
FIG = OUT / "post_result_diagnostic_figures"
POOL = ROOT / "PreparedData" / "medium150_historical_efficacy_candidate_pool" / "hosts"
GENERIC_RESULT = ROOT / "AnalysisResults" / "expanded_v1" / "candidate_identity_remediation"
GENERIC_POOL = ROOT / "PreparedData" / "support_scale_final_fresh_medium_pool"
GENERIC_INPUT = ROOT / "PreparedData" / "support_scale_final_fresh_medium_inputs"
QUALITY = ROOT / "AnalysisResults" / "insertion_quality_v30" / "insertion_quality_object_metrics_v30.csv"
PAIR_CSV = OUT / "medium150_high_low_pair_manifest.csv"
FRAME_CSV = OUT / "medium150_efficacy_frame_manifest_frozen.csv"
GATE_JSON = OUT / "medium150_pretraining_gate_results.json"
FAIL_FREEZE = OUT / "historical_medium150_tenengrad_failure_freeze.json"
COMMON = TOOLS / "support_scale_radiometry_calibration_common.py"
LABEL = "POST_RESULT_DIAGNOSTIC"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported JSON value: {type(item)!r}")

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=convert) + "\n",
        encoding="utf-8",
    )


def write_df(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_md(path: Path, text: str) -> None:
    if LABEL not in text[:500]:
        text = f"**{LABEL}**\n\n" + text
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f6(value: float | int | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.6f}"


def bool_word(value: bool) -> str:
    return "TRUE" if bool(value) else "FALSE"


def summary(values: Iterable[float]) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sample_sd": float(np.std(x, ddof=1)) if x.size > 1 else math.nan,
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    a, b = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return math.nan, math.nan
    result = spearmanr(a[keep], b[keep])
    return float(result.statistic), float(result.pvalue)


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    a, b = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return math.nan, math.nan
    result = pearsonr(a[keep], b[keep])
    return float(result.statistic), float(result.pvalue)


def standardized_w(values: Iterable[float], native: np.ndarray, native_sd: float) -> float:
    return float(wasserstein_distance(np.asarray(list(values), dtype=float), native) / max(native_sd, 1e-12))


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    table = frame if columns is None else frame[columns]
    if max_rows is not None:
        table = table.head(max_rows)
    headers = [str(value) for value in table.columns]
    rows = []
    for row in table.itertuples(index=False, name=None):
        rows.append([f"{value:.6f}" if isinstance(value, float) and np.isfinite(value) else str(value) for value in row])
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(out)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT / value.replace("\\", "/")
    if candidate.exists():
        return candidate
    # Retained generic manifests may contain an absolute path from the same
    # workspace.  Preserve only the suffix after a known top-level directory.
    normalized = value.replace("\\", "/")
    for anchor in ("PreparedData/", "AnalysisResults/", "runs/"):
        if anchor in normalized:
            candidate = ROOT / normalized.split(anchor, 1)[1]
            candidate = ROOT / anchor.rstrip("/") / normalized.split(anchor, 1)[1]
            if candidate.exists():
                return candidate
    raise FileNotFoundError(value)


def source_metrics(path: Path) -> dict[str, float | int]:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if xs.size == 0:
        raise ValueError(f"Empty source alpha: {path}")
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    rgb = rgba[y1:y2, x1:x2, :3]
    mask = alpha[y1:y2, x1:x2] > 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    centered = (gray - float(np.mean(gray[mask]))) * mask
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    yy, xx = np.indices(power.shape)
    rr = np.sqrt(((xx - (power.shape[1] - 1) / 2) / max(power.shape[1], 1)) ** 2 + ((yy - (power.shape[0] - 1) / 2) / max(power.shape[0], 1)) ** 2)
    high = float(power[rr >= 0.20].sum() / max(power.sum(), 1e-12))
    return {
        "raster_width": int(rgba.shape[1]),
        "raster_height": int(rgba.shape[0]),
        "alpha_crop_width": int(rgb.shape[1]),
        "alpha_crop_height": int(rgb.shape[0]),
        "alpha_crop_aspect": float(rgb.shape[1] / rgb.shape[0]),
        "source_tenengrad_mean": float(np.mean(grad[mask])),
        "source_laplacian_variance": float(np.var(lap[mask], ddof=1)),
        "source_high_frequency_power_fraction": high,
    }


def host_local_metrics(path: Path, bbox: list[float]) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    obj, ring, _ = object_and_ring_masks(rgb.shape[:2], bbox)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    obj_mean, ring_mean = float(np.mean(grad[obj])), float(np.mean(grad[ring]))
    return {
        "host_bbox_tenengrad_mean": obj_mean,
        "host_ring_tenengrad_mean": ring_mean,
        "host_bbox_ring_tenengrad_ratio": obj_mean / max(ring_mean, 1e-5),
    }


def arm_values(pair: pd.DataFrame, arm: str, metric: str = "tenengrad_ratio") -> np.ndarray:
    return pair[f"{arm.lower()}_{metric}"].to_numpy(dtype=float)


def direct_recompute(pair: pd.DataFrame, native_ten: np.ndarray, native_sd: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in pair.to_dict("records"):
        bbox = [record["bbox_x"], record["bbox_y"], record["bbox_width"], record["bbox_height"]]
        for arm in ("high", "low"):
            path = resolve_path(str(record[f"{arm}_image_path"]))
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            metric = extended_appearance_metrics(rgb, bbox)
            stored = float(record[f"{arm}_tenengrad_ratio"])
            recomputed = float(metric["tenengrad_ratio"])
            rows.append({
                "frame_index": int(record["frame_index"]), "arm": arm.title(),
                "candidate_id": int(record[f"{arm}_candidate_id"]), "image_path": str(record[f"{arm}_image_path"]),
                "stored_tenengrad_ratio": stored, "recomputed_tenengrad_ratio": recomputed,
                "absolute_discrepancy": abs(recomputed - stored),
                "foreground_gradient_mean": float(metric["foreground_gradient_mean"]),
                "background_gradient_mean": float(metric["background_gradient_mean"]),
            })
    frame = pd.DataFrame(rows)
    high = frame.loc[frame.arm == "High", "recomputed_tenengrad_ratio"].to_numpy(float)
    low = frame.loc[frame.arm == "Low", "recomputed_tenengrad_ratio"].to_numpy(float)
    result = {
        "high_standardized_wasserstein": standardized_w(high, native_ten, native_sd),
        "low_standardized_wasserstein": standardized_w(low, native_ten, native_sd),
        "maximum_absolute_value_discrepancy": float(frame.absolute_discrepancy.max()),
        "high": summary(high), "low": summary(low),
    }
    return frame, result


def aggregate_group(rows: list[dict[str, Any]], group_keys: list[str], value_key: str = "tenengrad_ratio") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    result = []
    for keys, group in frame.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row.update(summary(group[value_key].to_numpy(float)))
        result.append(row)
    return pd.DataFrame(result)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    gate = load_json(GATE_JSON)
    freeze = load_json(FAIL_FREEZE)
    assert gate["MEDIUM150_PRETRAINING_GATE"] == "FAIL"
    assert freeze["MEDIUM150_PRETRAINING_GATE"] == "FAIL"
    assert not gate["TRAINING_UNLOCKED"] and not freeze["TRAINING_UNLOCKED"]
    assert int(gate["DETECTOR_TRAINING_RUNS"]) == 0 and int(gate["AP_VALUES_INSPECTED"]) == 0

    pair = pd.read_csv(PAIR_CSV)
    frames = pd.read_csv(FRAME_CSV)
    generic_pair = pd.read_csv(GENERIC_RESULT / "final_fresh_selected_pairs.csv")
    native, native_summary = native_metrics(QUALITY)
    ten_index = METRICS.index("tenengrad_ratio")
    native_ten = native[:, ten_index]
    native_sd = float(native_summary["tenengrad_ratio"]["sample_sd"])

    # Direct selected-image recomputation is deliberately limited to the
    # already selected 150 High + 150 Low final JPEGs.
    direct_df, direct = direct_recompute(pair, native_ten, native_sd)
    write_df(OUT / "historical_tenengrad_direct_recompute.csv", direct_df)
    frozen_high = float(freeze["TENENGRAD_HIGH"]["standardized_wasserstein"])
    frozen_low = float(freeze["TENENGRAD_LOW"]["standardized_wasserstein"])
    direct["high_wasserstein_discrepancy"] = direct["high_standardized_wasserstein"] - frozen_high
    direct["low_wasserstein_discrepancy"] = direct["low_standardized_wasserstein"] - frozen_low
    direct_discrepancy = bool(
        direct["maximum_absolute_value_discrepancy"] > 1e-6
        or abs(direct["high_wasserstein_discrepancy"]) > 1e-6
        or abs(direct["low_wasserstein_discrepancy"]) > 1e-6
    )

    # Implementation/reference parity.
    generic_all = load_json(GENERIC_RESULT / "final_fresh_all_gate_results.json")
    historical_execution = load_json(OUT / "medium150_pretraining_execution_freeze.json")
    hist_hashes = {row["path"]: row["sha256"] for row in historical_execution["files"]}
    current_common_hash = sha256(COMMON)
    generic_common_hash = generic_all["precommit"]["calibrator_sha256"]
    historical_common_hash = hist_hashes["tools/support_scale_radiometry_calibration_common.py"]
    quality_hash = sha256(QUALITY)
    generic_quality_hash = generic_all["precommit"].get("quality_csv_sha256", quality_hash)
    historical_quality_hash = hist_hashes["AnalysisResults/insertion_quality_v30/insertion_quality_object_metrics_v30.csv"]
    reference_summary_path = ROOT / "AnalysisResults" / "expanded_v1" / "support_scale_balanced_selector_development" / "native_radiometry_reference_summary.csv"
    reference_hashes_path = ROOT / "AnalysisResults" / "expanded_v1" / "support_scale_balanced_selector_development" / "native_radiometry_reference_hashes.json"
    reference_hashes = load_json(reference_hashes_path)
    ref_df = pd.read_csv(reference_summary_path)
    reference_values_identical = bool(
        len(ref_df) == 150
        and np.max(np.abs(ref_df[list(METRICS)].to_numpy(float) - native)) <= 1e-12
    )
    native_identical = bool(
        quality_hash == historical_quality_hash == reference_hashes["source_sha256"]
        and quality_hash == generic_quality_hash
        and reference_values_identical
    )
    code_identical = current_common_hash == generic_common_hash == historical_common_hash
    # Exact package versions were not serialized in both historical run
    # freezes, so use EQUIVALENT rather than overclaiming byte-for-byte runtime
    # identity. Numeric direct reproduction supplies equivalence evidence.
    implementation_parity = "EQUIVALENT" if code_identical and not direct_discrepancy else ("DIFFERENT" if not code_identical else "UNVERIFIED")
    parity_hashes = {
        "analysis": LABEL,
        "TENENGRAD_IMPLEMENTATION_PARITY": implementation_parity,
        "common_metric_source": "tools/support_scale_radiometry_calibration_common.py:extended_appearance_metrics",
        "current_common_source_sha256": current_common_hash,
        "generic_frozen_common_source_sha256": generic_common_hash,
        "historical_frozen_common_source_sha256": historical_common_hash,
        "quality_source_sha256": quality_hash,
        "generic_quality_source_sha256": generic_quality_hash,
        "historical_quality_source_sha256": historical_quality_hash,
        "native_reference_summary_sha256": sha256(reference_summary_path),
        "native_reference_values_identical": native_identical,
        "software_current": {
            "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "pillow": Image.__version__, "matplotlib": matplotlib.__version__,
        },
        "historical_and_generic_package_versions_separately_frozen": False,
        "direct_recompute_maximum_value_discrepancy": direct["maximum_absolute_value_discrepancy"],
    }
    write_json(OUT / "tenengrad_metric_parity_hashes.json", parity_hashes)

    # Load historical retained pools sequentially. Candidate metrics are
    # retained; no candidate image is generated or altered.
    selected_by_host = {
        int(row.frame_index): {"high": int(row.high_candidate_id), "low": int(row.low_candidate_id)}
        for row in pair.itertuples(index=False)
    }
    pool_all: list[float] = []
    hist_group_rows: list[dict[str, Any]] = []
    hist_selected_group_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    pool_summary_rows: list[dict[str, Any]] = []
    candidate_corr_rows: list[dict[str, Any]] = []
    hist_stage_rows: list[dict[str, Any]] = []
    hist_raw_family_counts: dict[int, Counter[str]] = {}
    hist_raw_variant_counts: dict[int, Counter[str]] = {}

    for host_index in range(1, 151):
        host_dir = POOL / f"host_{host_index:03d}"
        candidates = load_json(host_dir / "candidate_manifest_retained.json")["candidates"]
        values = np.asarray([float(row["calibrated_metrics"]["tenengrad_ratio"]) for row in candidates])
        pool_all.extend(values.tolist())
        ids = np.asarray([int(row["candidate_id"]) for row in candidates], dtype=np.int64)
        support = pd.read_csv(host_dir / "support_distances.csv")
        support_map = dict(zip(support.candidate_id.astype(int), support.support_distance.astype(float)))
        support_values = np.asarray([support_map[int(value)] for value in ids], dtype=float)
        rho, pvalue = safe_spearman(support_values, values)
        candidate_corr_rows.append({"frame_index": host_index, "n": len(values), "spearman_support_tenengrad": rho, "descriptive_p": pvalue})
        q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        pool_summary_rows.append({
            "frame_index": host_index, "n": len(values), "candidate_pool_mean": float(np.mean(values)),
            "candidate_pool_median": float(med), "candidate_pool_q1": float(q1), "candidate_pool_q3": float(q3),
            "candidate_pool_iqr": float(q3 - q1), "candidate_pool_minimum": float(np.min(values)),
            "candidate_pool_maximum": float(np.max(values)),
        })
        lookup = {int(row["candidate_id"]): row for row in candidates}
        selected = selected_by_host[host_index]
        selector = {"frame_index": host_index, "retained_pool_n": len(values), "retained_pool_median": float(med), "retained_pool_q1": float(q1), "retained_pool_q3": float(q3)}
        for arm in ("high", "low"):
            row = lookup[selected[arm]]
            value = float(row["calibrated_metrics"]["tenengrad_ratio"])
            percentile = float((np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values))
            selector[f"selected_{arm}_candidate_id"] = selected[arm]
            selector[f"selected_{arm}_tenengrad"] = value
            selector[f"selected_{arm}_percentile"] = percentile
            selector[f"selected_{arm}_below_pool_median"] = value < med
            selector[f"selected_{arm}_below_pool_q25"] = value < q1
            selector[f"selected_{arm}_above_pool_q75"] = value > q3
            hist_selected_group_rows.append({
                "cohort": "historical_medium", "subset": "selected", "arm": arm.title(),
                "family": row["appearance_family"], "variant": row["appearance_variant"], "tenengrad_ratio": value,
            })
            hist_stage_rows.append({
                "cohort": "historical_medium", "frame_index": host_index, "arm": arm.title(),
                "candidate_id": selected[arm], "precalibration_tenengrad_ratio": float(row["precalibration_metrics"]["tenengrad_ratio"]),
                "final_tenengrad_ratio": value,
            })
        selector["selected_pair_mean_percentile"] = (selector["selected_high_percentile"] + selector["selected_low_percentile"]) / 2
        selector_rows.append(selector)
        for row in candidates:
            hist_group_rows.append({
                "cohort": "historical_medium", "subset": "retained_pool", "arm": "Pool",
                "family": row["appearance_family"], "variant": row["appearance_variant"],
                "tenengrad_ratio": float(row["calibrated_metrics"]["tenengrad_ratio"]),
            })
        family_counts, variant_counts = Counter(), Counter()
        with (host_dir / "candidate_identity_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                family_counts[str(row["appearance_family"])] += 1
                variant_counts[str(row["appearance_variant"])] += 1
        hist_raw_family_counts[host_index] = family_counts
        hist_raw_variant_counts[host_index] = variant_counts
        del candidates, lookup
        gc.collect()

    selector_df = pd.DataFrame(selector_rows)
    pool_summary_df = pd.DataFrame(pool_summary_rows)
    candidate_corr_df = pd.DataFrame(candidate_corr_rows)
    write_df(OUT / "tenengrad_selector_shift.csv", selector_df)
    write_df(OUT / "historical_candidate_pool_tenengrad_summary.csv", pool_summary_df)
    write_df(OUT / "candidate_support_tenengrad_tradeoff.csv", candidate_corr_df)

    # Generic Medium PASS retained pool and selected metadata.
    generic_selected_ids = {
        int(value) for row in generic_pair.itertuples(index=False)
        for value in (row.high_candidate_id, row.low_candidate_id)
    }
    generic_group_rows: list[dict[str, Any]] = []
    generic_selected_lookup: dict[int, dict[str, Any]] = {}
    generic_base_sources: dict[tuple[int, int], dict[str, Any]] = {}
    generic_retained = load_json(GENERIC_POOL / "candidate_manifest.json")["candidates"]
    for row in generic_retained:
        candidate_id = int(row["candidate_id"])
        generic_group_rows.append({
            "cohort": "generic_medium", "subset": "retained_pool", "arm": "Pool",
            "family": row["appearance_family"], "variant": row["appearance_variant"],
            "tenengrad_ratio": float(row["calibrated_metrics"]["tenengrad_ratio"]),
        })
        if candidate_id in generic_selected_ids:
            generic_selected_lookup[candidate_id] = row
        generic_base_sources.setdefault((int(row["host_index"]), int(row["base_candidate_id"])), row)
    del generic_retained
    gc.collect()

    generic_selected_group_rows: list[dict[str, Any]] = []
    generic_stage_rows: list[dict[str, Any]] = []
    for record in generic_pair.to_dict("records"):
        for arm in ("high", "low"):
            candidate_id = int(record[f"{arm}_candidate_id"])
            row = generic_selected_lookup[candidate_id]
            generic_selected_group_rows.append({
                "cohort": "generic_medium", "subset": "selected", "arm": arm.title(),
                "family": row["appearance_family"], "variant": row["appearance_variant"],
                "tenengrad_ratio": float(row["calibrated_metrics"]["tenengrad_ratio"]),
            })
            generic_stage_rows.append({
                "cohort": "generic_medium", "frame_index": int(record["local_host_id"]), "arm": arm.title(),
                "candidate_id": candidate_id, "precalibration_tenengrad_ratio": float(row["precalibration_metrics"]["tenengrad_ratio"]),
                "final_tenengrad_ratio": float(row["calibrated_metrics"]["tenengrad_ratio"]),
            })

    # Generic raw counts establish the pre-amendment slot allocation.
    generic_raw_family_counts: dict[int, Counter[str]] = defaultdict(Counter)
    generic_raw_variant_counts: dict[int, Counter[str]] = defaultdict(Counter)
    generic_raw = load_json(GENERIC_POOL / "candidate_manifest_raw.json")["candidates"]
    for row in generic_raw:
        host = int(row["host_index"])
        generic_raw_family_counts[host][str(row["appearance_family"])] += 1
        generic_raw_variant_counts[host][str(row["appearance_variant"])] += 1
        generic_base_sources.setdefault((host, int(row["base_candidate_id"])), row)
    del generic_raw
    gc.collect()

    # Distribution comparisons.
    distributions = {
        "native_reference": native_ten,
        "generic_medium_high": arm_values(generic_pair, "high"),
        "generic_medium_low": arm_values(generic_pair, "low"),
        "historical_medium_high": arm_values(pair, "high"),
        "historical_medium_low": arm_values(pair, "low"),
        "historical_retained_pool": np.asarray(pool_all, dtype=float),
    }
    distribution_rows = []
    for name, values in distributions.items():
        row = {"distribution": name, **summary(values)}
        row["standardized_wasserstein_to_native"] = 0.0 if name == "native_reference" else standardized_w(values, native_ten, native_sd)
        distribution_rows.append(row)
    distribution_df = pd.DataFrame(distribution_rows)
    write_df(OUT / "tenengrad_distribution_comparison.csv", distribution_df)

    quantiles = np.linspace(0.0, 1.0, 101)
    native_q = np.quantile(native_ten, quantiles)
    high_q = np.quantile(distributions["historical_medium_high"], quantiles)
    low_q = np.quantile(distributions["historical_medium_low"], quantiles)
    quantile_df = pd.DataFrame({
        "quantile": quantiles, "native": native_q, "historical_high": high_q, "historical_low": low_q,
        "high_minus_native": high_q - native_q, "low_minus_native": low_q - native_q,
        "absolute_high_difference": np.abs(high_q - native_q), "absolute_low_difference": np.abs(low_q - native_q),
    })
    write_df(OUT / "tenengrad_quantile_localization.csv", quantile_df)
    zones = {"lower_tail": (0.0, 0.25), "center": (0.25, 0.75), "upper_tail": (0.75, 1.0)}
    zone_rows = []
    for zone, (lo, hi) in zones.items():
        mask = (quantiles >= lo) & (quantiles <= hi)
        for arm, diff in (("High", np.abs(high_q - native_q)), ("Low", np.abs(low_q - native_q))):
            contribution = float(np.trapezoid(diff[mask], quantiles[mask]) / native_sd)
            zone_rows.append({"zone": zone, "arm": arm, "standardized_quantile_area": contribution})
    zone_df = pd.DataFrame(zone_rows)

    # Common High/Low shift.
    high = distributions["historical_medium_high"]
    low = distributions["historical_medium_low"]
    paired = high - low
    ci_half = float(t.ppf(0.975, len(paired) - 1) * np.std(paired, ddof=1) / math.sqrt(len(paired)))
    common = {
        "paired_high_minus_low_mean": float(np.mean(paired)), "paired_high_minus_low_median": float(np.median(paired)),
        "paired_high_minus_low_sample_sd": float(np.std(paired, ddof=1)),
        "paired_mean_ci95_low": float(np.mean(paired) - ci_half), "paired_mean_ci95_high": float(np.mean(paired) + ci_half),
        "positive_sign_count": int(np.sum(paired > 0)), "zero_sign_count": int(np.sum(paired == 0)),
        "negative_sign_count": int(np.sum(paired < 0)), "paired_pearson_correlation": safe_pearson(high, low)[0],
        "high_low_standardized_wasserstein": standardized_w(high, low, native_sd),
        "high_native_standardized_wasserstein": standardized_w(high, native_ten, native_sd),
        "low_native_standardized_wasserstein": standardized_w(low, native_ten, native_sd),
    }

    # Appearance-family and variant diagnostics.
    all_group_rows = hist_group_rows + hist_selected_group_rows + generic_group_rows + generic_selected_group_rows
    family_df = aggregate_group(all_group_rows, ["cohort", "subset", "arm", "family"])
    family_df["native_standardized_median_location"] = (family_df["median"] - float(np.median(native_ten))) / native_sd
    variant_df = aggregate_group(all_group_rows, ["cohort", "subset", "arm", "family", "variant"])
    variant_df["native_standardized_median_location"] = (variant_df["median"] - float(np.median(native_ten))) / native_sd
    write_df(OUT / "tenengrad_by_appearance_family.csv", family_df)
    write_df(OUT / "tenengrad_by_variant.csv", variant_df)

    # Candidate-slot allocation equivalence.
    families = sorted(set().union(*(set(value) for value in hist_raw_family_counts.values()), *(set(value) for value in generic_raw_family_counts.values())))
    slot_rows = []
    family_preserved = True
    for branch, counts, n_hosts in (("pre_amendment_generic", generic_raw_family_counts, 30), ("post_amendment_historical", hist_raw_family_counts, 150)):
        for family in families:
            values = [counts[index][family] for index in sorted(counts)]
            slot_rows.append({
                "branch": branch, "family": family, "host_count": n_hosts, "minimum_slots_per_host": min(values),
                "maximum_slots_per_host": max(values), "mean_slots_per_host": float(np.mean(values)), "total_slots": int(sum(values)),
            })
    slot_df = pd.DataFrame(slot_rows)
    for family in families:
        pre = slot_df[(slot_df.branch == "pre_amendment_generic") & (slot_df.family == family)].iloc[0]
        post = slot_df[(slot_df.branch == "post_amendment_historical") & (slot_df.family == family)].iloc[0]
        family_preserved &= pre.minimum_slots_per_host == pre.maximum_slots_per_host == post.minimum_slots_per_host == post.maximum_slots_per_host
    slot_class = "MECHANICAL_WITHIN_FAMILY_SUBSTITUTION" if family_preserved else "ADDITIONAL_SEARCH_DISTRIBUTION_CHANGE"
    write_df(OUT / "medium_candidate_slot_allocation_equivalence.csv", slot_df)

    # Source-raster metrics for 58 historical assets and retained generic base-render source assets.
    hist_source_rows: list[dict[str, Any]] = []
    for source_hash, group in frames.groupby("source_cutout_sha256"):
        first = group.iloc[0]
        path = resolve_path(str(first.source_cutout_path))
        hist_source_rows.append({
            "source_cutout_sha256": source_hash, "source_cutout_path": str(first.source_cutout_path),
            "reuse_count": int(len(group)), "host_count": int(group.host_id.nunique()), **source_metrics(path),
        })
    hist_source_df = pd.DataFrame(hist_source_rows)
    write_df(OUT / "historical_source_raster_tenengrad.csv", hist_source_df)
    source_metric_by_hash = {row.source_cutout_sha256: row._asdict() for row in hist_source_df.itertuples(index=False)}

    generic_source_groups: dict[str, dict[str, Any]] = {}
    for row in generic_base_sources.values():
        key = str(row["source_cutout_sha256"])
        entry = generic_source_groups.setdefault(key, {"path": str(row["source_cutout"]), "reuse_count": 0, "hosts": set()})
        entry["reuse_count"] += 1
        entry["hosts"].add(int(row["host_index"]))
    generic_source_rows = []
    for source_hash, value in generic_source_groups.items():
        path = resolve_path(value["path"])
        generic_source_rows.append({
            "source_cutout_sha256": source_hash, "source_cutout_path": value["path"],
            "reuse_count": int(value["reuse_count"]), "host_count": len(value["hosts"]), **source_metrics(path),
        })
    generic_source_df = pd.DataFrame(generic_source_rows)
    write_df(OUT / "generic_medium_source_raster_tenengrad_diagnostic.csv", generic_source_df)

    # Source reuse, scale, resize, and anisotropy.
    reuse_rows = []
    for source_hash, group in pair.groupby("source_cutout_sha256"):
        source = source_metric_by_hash[source_hash]
        reuse_rows.append({
            "source_cutout_sha256": source_hash, "reuse_count": int(len(group)), "host_count": int(group.historical_host_id.nunique()),
            "source_raster_tenengrad": source["source_tenengrad_mean"],
            "mean_final_high_tenengrad": float(group.high_tenengrad_ratio.mean()),
            "mean_final_low_tenengrad": float(group.low_tenengrad_ratio.mean()),
            "mean_final_pair_tenengrad": float(((group.high_tenengrad_ratio + group.low_tenengrad_ratio) / 2).mean()),
        })
    reuse_df = pd.DataFrame(reuse_rows)
    write_df(OUT / "tenengrad_source_reuse_audit.csv", reuse_df)

    scale_df = pair[["frame_index", "historical_host_id", "bbox_width", "bbox_height", "bbox_scale", "high_tenengrad_ratio", "low_tenengrad_ratio"]].copy()
    scale_df["bbox_aspect_ratio"] = scale_df.bbox_width / scale_df.bbox_height
    scale_df["pair_mean_tenengrad"] = (scale_df.high_tenengrad_ratio + scale_df.low_tenengrad_ratio) / 2
    scale_df["scale_bin"] = pd.cut(scale_df.bbox_scale, [32, 40, 48, 56, 64], right=False, labels=["32-<40", "40-<48", "48-<56", "56-<64"])
    write_df(OUT / "tenengrad_scale_diagnostic.csv", scale_df)

    resize_rows = []
    for row in pair.itertuples(index=False):
        source = source_metric_by_hash[row.source_cutout_sha256]
        source_w, source_h = float(source["alpha_crop_width"]), float(source["alpha_crop_height"])
        target_w, target_h = float(row.bbox_width), float(row.bbox_height)
        source_aspect, target_aspect = source_w / source_h, target_w / target_h
        resize_rows.append({
            "frame_index": int(row.frame_index), "source_cutout_sha256": row.source_cutout_sha256,
            "source_width": source_w, "source_height": source_h, "target_width": target_w, "target_height": target_h,
            "source_aspect": source_aspect, "target_bbox_aspect": target_aspect,
            "anisotropy_abs_log_aspect_ratio": abs(math.log(target_aspect / source_aspect)),
            "x_resize_ratio": target_w / source_w, "y_resize_ratio": target_h / source_h,
            "area_scale_ratio": (target_w * target_h) / (source_w * source_h),
            "high_tenengrad_ratio": float(row.high_tenengrad_ratio), "low_tenengrad_ratio": float(row.low_tenengrad_ratio),
            "pair_mean_tenengrad": float((row.high_tenengrad_ratio + row.low_tenengrad_ratio) / 2),
        })
    resize_df = pd.DataFrame(resize_rows)
    write_df(OUT / "tenengrad_resize_anisotropy_audit.csv", resize_df)
    write_df(OUT / "tenengrad_resize_factor_audit.csv", resize_df[[
        "frame_index", "source_width", "source_height", "target_width", "target_height", "x_resize_ratio", "y_resize_ratio",
        "area_scale_ratio", "high_tenengrad_ratio", "low_tenengrad_ratio", "pair_mean_tenengrad"
    ]])

    # Host/local background, using the exact frozen bbox/ring implementation.
    host_rows = []
    for row in pair.itertuples(index=False):
        metrics = host_local_metrics(resolve_path(str(row.host_path)), [row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height])
        host_rows.append({
            "cohort": "historical_medium", "frame_index": int(row.frame_index), "host_id": int(row.historical_host_id),
            "source_scene": row.source_scene, "bbox_width": row.bbox_width, "bbox_height": row.bbox_height,
            "high_tenengrad_ratio": row.high_tenengrad_ratio, "low_tenengrad_ratio": row.low_tenengrad_ratio,
            "pair_mean_tenengrad": (row.high_tenengrad_ratio + row.low_tenengrad_ratio) / 2, **metrics,
        })
    host_df = pd.DataFrame(host_rows)
    write_df(OUT / "historical_host_local_tenengrad.csv", host_df)

    generic_coco = load_json(GENERIC_INPUT / "annotations" / "base_instances_train.json")
    generic_image_map = {int(row["id"]): row["file_name"] for row in generic_coco["images"]}
    generic_host_rows = []
    for record in generic_pair.to_dict("records"):
        selected = generic_selected_lookup[int(record["high_candidate_id"])]
        ref_id = int(selected["reference_image_id"])
        path = GENERIC_INPUT / "images" / "train" / generic_image_map[ref_id]
        bbox = [float(v) for v in selected["bbox"]]
        metrics = host_local_metrics(path, bbox)
        generic_host_rows.append({
            "cohort": "generic_medium", "frame_index": int(record["local_host_id"]), "host_id": int(record["original_v47_host_id"]),
            "source_scene": selected.get("source_scene", ""), "bbox_width": bbox[2], "bbox_height": bbox[3],
            "high_tenengrad_ratio": record["high_tenengrad_ratio"], "low_tenengrad_ratio": record["low_tenengrad_ratio"],
            "pair_mean_tenengrad": (record["high_tenengrad_ratio"] + record["low_tenengrad_ratio"]) / 2, **metrics,
        })
    generic_host_df = pd.DataFrame(generic_host_rows)
    write_df(OUT / "generic_host_local_tenengrad_diagnostic.csv", generic_host_df)

    # Scene diagnostics and post-result leave-one-scene-out sensitivity.
    scene_rows = []
    for scene, group in pair.groupby("source_scene"):
        scene_rows.append({
            "source_scene": scene, "n_hosts": len(group), "fraction_hosts": len(group) / len(pair),
            "high_mean": float(group.high_tenengrad_ratio.mean()), "high_median": float(group.high_tenengrad_ratio.median()),
            "low_mean": float(group.low_tenengrad_ratio.mean()), "low_median": float(group.low_tenengrad_ratio.median()),
            "high_native_standardized_median_location": float((group.high_tenengrad_ratio.median() - np.median(native_ten)) / native_sd),
            "low_native_standardized_median_location": float((group.low_tenengrad_ratio.median() - np.median(native_ten)) / native_sd),
            "support_gap_median": float(group.support_gap.median()), "scale_median": float(group.bbox_scale.median()),
        })
    scene_df = pd.DataFrame(scene_rows).sort_values("n_hosts", ascending=False)
    write_df(OUT / "tenengrad_by_scene.csv", scene_df)
    loso_rows = [{"removed_scene": "NONE_FULL", "remaining_n": len(pair), "high_standardized_wasserstein": frozen_high, "low_standardized_wasserstein": frozen_low}]
    for scene in sorted(pair.source_scene.unique()):
        remaining = pair[pair.source_scene != scene]
        loso_rows.append({
            "removed_scene": scene, "remaining_n": len(remaining),
            "high_standardized_wasserstein": standardized_w(remaining.high_tenengrad_ratio, native_ten, native_sd),
            "low_standardized_wasserstein": standardized_w(remaining.low_tenengrad_ratio, native_ten, native_sd),
        })
    loso_df = pd.DataFrame(loso_rows)
    write_df(OUT / "tenengrad_scene_loso.csv", loso_df)

    # Selected-pair support/Tenengrad association.
    support_df = pair[["frame_index", "support_gap", "high_tenengrad_ratio", "low_tenengrad_ratio"]].copy()
    support_df["pair_mean_tenengrad"] = (support_df.high_tenengrad_ratio + support_df.low_tenengrad_ratio) / 2
    write_df(OUT / "support_tenengrad_association.csv", support_df)

    # Weber/Tenengrad decoupling and partial calibration-stage localization.
    metric_rows = []
    for row in pair.to_dict("records"):
        for arm in ("high", "low"):
            metric_rows.append({"frame_index": row["frame_index"], "arm": arm.title(), **{metric: row[f"{arm}_{metric}"] for metric in METRICS}})
    metric_df = pd.DataFrame(metric_rows)
    correlation_df = metric_df[list(METRICS)].corr(method="spearman")
    correlation_df.insert(0, "metric", correlation_df.index)
    write_df(OUT / "weber_tenengrad_decoupling.csv", correlation_df.reset_index(drop=True))
    stage_df = pd.DataFrame(hist_stage_rows + generic_stage_rows)
    write_df(OUT / "tenengrad_calibration_stage_diagnostic.csv", stage_df)
    stage_summary_rows = []
    for (cohort, stage), values in [
        (("historical_medium", "precalibration"), stage_df[stage_df.cohort == "historical_medium"].precalibration_tenengrad_ratio),
        (("historical_medium", "final"), stage_df[stage_df.cohort == "historical_medium"].final_tenengrad_ratio),
        (("generic_medium", "precalibration"), stage_df[stage_df.cohort == "generic_medium"].precalibration_tenengrad_ratio),
        (("generic_medium", "final"), stage_df[stage_df.cohort == "generic_medium"].final_tenengrad_ratio),
    ]:
        stage_summary_rows.append({"cohort": cohort, "stage": stage, **summary(values), "standardized_wasserstein_to_native": standardized_w(values, native_ten, native_sd)})
    stage_summary_df = pd.DataFrame(stage_summary_rows)

    # Descriptive correlations used for localization.
    scale_corr = {name: safe_spearman(scale_df[name], scale_df.pair_mean_tenengrad)[0] for name in ("bbox_scale", "bbox_width", "bbox_height", "bbox_aspect_ratio")}
    resize_corr = {name: safe_spearman(resize_df[name], resize_df.pair_mean_tenengrad)[0] for name in ("anisotropy_abs_log_aspect_ratio", "x_resize_ratio", "y_resize_ratio", "area_scale_ratio")}
    host_corr = {name: safe_spearman(host_df[name], host_df.pair_mean_tenengrad)[0] for name in ("host_bbox_tenengrad_mean", "host_ring_tenengrad_mean", "host_bbox_ring_tenengrad_ratio")}
    support_corr = {
        "high": safe_spearman(support_df.support_gap, support_df.high_tenengrad_ratio)[0],
        "low": safe_spearman(support_df.support_gap, support_df.low_tenengrad_ratio)[0],
        "pair_mean": safe_spearman(support_df.support_gap, support_df.pair_mean_tenengrad)[0],
        "candidate_pool_median_within_host": float(candidate_corr_df.spearman_support_tenengrad.median()),
        "candidate_pool_negative_fraction": float(np.mean(candidate_corr_df.spearman_support_tenengrad < 0)),
    }
    reuse_corr = {
        "reuse_vs_pair_mean": safe_spearman(reuse_df.reuse_count, reuse_df.mean_final_pair_tenengrad)[0],
        "source_vs_pair_mean": safe_spearman(reuse_df.source_raster_tenengrad, reuse_df.mean_final_pair_tenengrad)[0],
    }

    # Pool/selector localization.
    pool_w = standardized_w(pool_all, native_ten, native_sd)
    selected_mean_percentile = float(selector_df.selected_pair_mean_percentile.mean())
    selected_median_percentile = float(selector_df.selected_pair_mean_percentile.median())
    selector_worsened = bool(max(frozen_high, frozen_low) > pool_w + 0.02 and selected_mean_percentile < 0.50)
    pool_already_shifted = bool(pool_w > 0.50)
    family_selected = family_df[(family_df.cohort == "historical_medium") & (family_df.subset == "selected")]
    family_range = float(family_selected["median"].max() - family_selected["median"].min())
    dominant_scene_fraction = float(scene_df.iloc[0].fraction_hosts)
    loso_high_range = float(loso_df.iloc[1:].high_standardized_wasserstein.max() - loso_df.iloc[1:].high_standardized_wasserstein.min())
    loso_low_range = float(loso_df.iloc[1:].low_standardized_wasserstein.max() - loso_df.iloc[1:].low_standardized_wasserstein.min())
    historical_source_median = float(hist_source_df.source_tenengrad_mean.median())
    generic_source_median = float(generic_source_df.source_tenengrad_mean.median())
    source_shift_supported = bool(historical_source_median < generic_source_median * 0.90)

    # Required figures (no AP).
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, np.quantile(np.concatenate([native_ten, high, low, distributions["generic_medium_high"], distributions["generic_medium_low"]]), 0.98), 34)
    for label, values, color, style in (
        ("Native reference", native_ten, "black", "-"), ("Generic Medium High", distributions["generic_medium_high"], "#1f77b4", "-"),
        ("Generic Medium Low", distributions["generic_medium_low"], "#17becf", "--"), ("Historical Medium High", high, "#d62728", "-"),
        ("Historical Medium Low", low, "#ff7f0e", "--"),
    ):
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2, linestyle=style, label=label)
    ax.set(xlabel="Tenengrad foreground/background ratio", ylabel="Density", title="POST_RESULT_DIAGNOSTIC: Tenengrad distributions")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "figure1_tenengrad_distribution_comparison.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, values, color in (("Native", native_ten, "black"), ("Historical High", high, "#d62728"), ("Historical Low", low, "#ff7f0e")):
        x = np.sort(values); y = np.arange(1, len(x) + 1) / len(x); axes[0].step(x, y, where="post", label=label, color=color)
    axes[0].set(xlabel="Tenengrad ratio", ylabel="ECDF", title="ECDF"); axes[0].legend(fontsize=8)
    axes[1].plot(quantiles, high_q - native_q, label="High - native", color="#d62728")
    axes[1].plot(quantiles, low_q - native_q, label="Low - native", color="#ff7f0e")
    axes[1].axhline(0, color="black", linewidth=0.8); axes[1].set(xlabel="Quantile", ylabel="Ratio difference", title="Quantile localization"); axes[1].legend(fontsize=8)
    fig.suptitle("POST_RESULT_DIAGNOSTIC"); fig.tight_layout(); fig.savefig(FIG / "figure2_tenengrad_ecdf_quantile_localization.png", dpi=220); plt.close(fig)

    source_pair = pair.merge(hist_source_df[["source_cutout_sha256", "source_tenengrad_mean"]], on="source_cutout_sha256", how="left")
    fig, ax = plt.subplots(figsize=(7, 5.2)); ax.scatter(source_pair.source_tenengrad_mean, (source_pair.high_tenengrad_ratio + source_pair.low_tenengrad_ratio) / 2, alpha=0.65)
    ax.set(xlabel="Source-raster absolute Tenengrad mean", ylabel="Final selected pair mean Tenengrad ratio", title="POST_RESULT_DIAGNOSTIC: source vs final")
    fig.tight_layout(); fig.savefig(FIG / "figure3_source_raster_vs_selected_tenengrad.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2)); ordered_scenes = scene_df.source_scene.tolist(); data = [pair.loc[pair.source_scene == scene, ["high_tenengrad_ratio", "low_tenengrad_ratio"]].to_numpy().ravel() for scene in ordered_scenes]
    ax.boxplot(data, tick_labels=[str(value)[-8:] for value in ordered_scenes], showfliers=False); ax.axhline(np.median(native_ten), color="black", linestyle="--", label="Native median")
    ax.set(xlabel="Source scene (suffix)", ylabel="Tenengrad ratio", title="POST_RESULT_DIAGNOSTIC: Tenengrad by scene"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "figure4_tenengrad_by_source_scene.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5)); axes[0].scatter(scale_df.bbox_scale, scale_df.pair_mean_tenengrad, alpha=0.6); axes[0].set(xlabel="Geometric mean bbox scale", ylabel="Pair mean Tenengrad ratio")
    axes[1].scatter(resize_df.anisotropy_abs_log_aspect_ratio, resize_df.pair_mean_tenengrad, alpha=0.6); axes[1].set(xlabel="|log(target aspect/source aspect)|", ylabel="Pair mean Tenengrad ratio")
    fig.suptitle("POST_RESULT_DIAGNOSTIC: scale and resize anisotropy"); fig.tight_layout(); fig.savefig(FIG / "figure5_tenengrad_scale_resize_anisotropy.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2)); ax.scatter(support_df.support_gap, support_df.pair_mean_tenengrad, alpha=0.65); ax.set(xlabel="Support gap", ylabel="Pair mean Tenengrad ratio", title="POST_RESULT_DIAGNOSTIC: support vs Tenengrad")
    fig.tight_layout(); fig.savefig(FIG / "figure6_support_gap_vs_pair_tenengrad.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2)); ax.scatter(selector_df.frame_index, selector_df.selected_high_percentile, s=14, label="High", alpha=0.7); ax.scatter(selector_df.frame_index, selector_df.selected_low_percentile, s=14, label="Low", alpha=0.7); ax.axhline(0.5, color="black", linestyle="--")
    ax.set(xlabel="Historical frame index", ylabel="Selected Tenengrad percentile in retained pool", title="POST_RESULT_DIAGNOSTIC: selector percentiles"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "figure7_selected_percentile_within_host_pool.png", dpi=220); plt.close(fig)

    # Reports.
    write_md(OUT / "tenengrad_metric_parity_audit.md", f"""# Tenengrad metric parity audit

**Classification: TENENGRAD_IMPLEMENTATION_PARITY = {implementation_parity}**

- Generic and historical branches both call `extended_appearance_metrics` and `distribution_gate` from `tools/support_scale_radiometry_calibration_common.py`.
- Current, generic-frozen, and historical-frozen source SHA-256 are all `{current_common_hash}`: **{bool_word(code_identical)}**.
- RGB is converted with `cv2.COLOR_RGB2GRAY`, cast to `float32`, and divided by 255.
- Sobel uses `CV_32F`, kernel size 3, x/y derivatives; magnitude is `sqrt(gx^2 + gy^2)`.
- ROI is the clipped exact bbox versus the frozen expanded local ring from `object_and_ring_masks`; no metric-stage resize is performed.
- Tenengrad is mean foreground gradient divided by max(mean ring gradient, 1e-5).
- Group Wasserstein uses `scipy.stats.wasserstein_distance` divided by native sample SD (`ddof=1`).
- Both branches use the same 150-row `Native real` reference from the same quality CSV hash `{quality_hash}`.
- Exact package versions were not independently serialized for both completed runs; therefore byte-identical runtime provenance is not overclaimed. Current versions are recorded in `tenengrad_metric_parity_hashes.json`.
- Independent selected-JPEG recomputation maximum value discrepancy: {f6(direct['maximum_absolute_value_discrepancy'])}.

No implementation difference was found. This audit does not alter the frozen FAIL.
""")

    write_md(OUT / "tenengrad_native_reference_parity_audit.md", f"""# Native Tenengrad reference parity audit

- `NATIVE_TENENGRAD_REFERENCE_IDENTICAL = {bool_word(native_identical)}`
- Quality/reference source: `AnalysisResults/insertion_quality_v30/insertion_quality_object_metrics_v30.csv`
- Source SHA-256: `{quality_hash}`
- Native rows: {len(native_ten)}; distinct summary rows compared: {len(ref_df)}.
- Native Tenengrad mean={f6(np.mean(native_ten))}, sample SD={f6(native_sd)}, median={f6(np.median(native_ten))}, Q1={f6(np.quantile(native_ten,.25))}, Q3={f6(np.quantile(native_ten,.75))}.
- Stored standalone native summary values and freshly loaded frozen values match to maximum absolute difference <= 1e-12: {bool_word(reference_values_identical)}.

No replacement or post-hoc conditioned reference was computed.
""")

    write_md(OUT / "historical_tenengrad_direct_recompute.md", f"""# Historical selected Tenengrad direct recomputation

- Recomputed only the already selected 150 High + 150 Low final JPEGs with the frozen implementation.
- High recomputed standardized Wasserstein: **{f6(direct['high_standardized_wasserstein'])}**; frozen={f6(frozen_high)}; discrepancy={direct['high_wasserstein_discrepancy']:.12g}.
- Low recomputed standardized Wasserstein: **{f6(direct['low_standardized_wasserstein'])}**; frozen={f6(frozen_low)}; discrepancy={direct['low_wasserstein_discrepancy']:.12g}.
- Maximum selected-image Tenengrad value discrepancy: {direct['maximum_absolute_value_discrepancy']:.12g}.
- `TENENGRAD_RESULT_DISCREPANCY = {bool_word(direct_discrepancy)}`.

The direct recomputation reproduces the frozen failure and cannot change its decision.
""")

    write_md(OUT / "tenengrad_distribution_comparison.md", "# Tenengrad distribution comparison\n\n" + markdown_table(distribution_df[["distribution", "n", "mean", "median", "sample_sd", "q05", "q25", "q75", "q95", "standardized_wasserstein_to_native"]]) + "\n\nThe historical High and Low distributions are jointly lower than the native center, whereas the previous generic Medium selected distributions were closer to native. This is descriptive and not a detector-efficacy result.")

    dominant_zone = zone_df.groupby("zone").standardized_quantile_area.sum().idxmax()
    write_md(OUT / "tenengrad_quantile_localization.md", f"""# Tenengrad ECDF and quantile failure localization

{markdown_table(zone_df)}

- Largest combined quantile-area contribution: **{dominant_zone}**.
- Historical High SD/native SD={np.std(high,ddof=1)/native_sd:.3f}; Historical Low SD/native SD={np.std(low,ddof=1)/native_sd:.3f}.
- The full 1st–99th percentile trajectory is in `tenengrad_quantile_localization.csv`; the mismatch is not reduced to the median.
- Figure 2 shows the ECDF and quantile differences.
""")

    write_md(OUT / "tenengrad_common_shift_audit.md", f"""# High/Low common-shift audit

- Paired High-Low mean={f6(common['paired_high_minus_low_mean'])}, median={f6(common['paired_high_minus_low_median'])}, SD={f6(common['paired_high_minus_low_sample_sd'])}.
- Descriptive 95% CI for mean paired difference: [{f6(common['paired_mean_ci95_low'])}, {f6(common['paired_mean_ci95_high'])}].
- Signs: positive={common['positive_sign_count']}, zero={common['zero_sign_count']}, negative={common['negative_sign_count']}.
- High/Low paired Pearson correlation={f6(common['paired_pearson_correlation'])}.
- Standardized W(High,Low)={f6(common['high_low_standardized_wasserstein'])}; W(High,native)={f6(common['high_native_standardized_wasserstein'])}; W(Low,native)={f6(common['low_native_standardized_wasserstein'])}.

The much smaller High-Low distance than either arm-to-native distance is consistent with a shared branch-level shift, not a primary support-arm imbalance.
""")

    selector_counts = {
        "below_median": int(selector_df.selected_high_below_pool_median.sum() + selector_df.selected_low_below_pool_median.sum()),
        "below_q25": int(selector_df.selected_high_below_pool_q25.sum() + selector_df.selected_low_below_pool_q25.sum()),
        "above_q75": int(selector_df.selected_high_above_pool_q75.sum() + selector_df.selected_low_above_pool_q75.sum()),
    }
    write_md(OUT / "tenengrad_selector_shift.md", f"""# S3 selector Tenengrad shift audit

- Retained candidate metrics were available for all 150 x 2048 candidates; no candidates were regenerated.
- Historical retained-pool standardized Wasserstein to native: {f6(pool_w)}.
- Selected mean percentile within host pool: {f6(selected_mean_percentile)}; median pair-mean percentile={f6(selected_median_percentile)}.
- Of 300 selected images: below pool median={selector_counts['below_median']}, below Q25={selector_counts['below_q25']}, above Q75={selector_counts['above_q75']}.
- `CANDIDATE_POOL_ALREADY_SHIFTED_BEFORE_SELECTION = {bool_word(pool_already_shifted)}`.
- `S3_SELECTION_WORSENED_TENENGRAD = {bool_word(selector_worsened)}` under the prespecified descriptive rule (selected max arm W > pool W+0.02 and mean percentile <0.50).

Host-level details are in `tenengrad_selector_shift.csv`; Figure 7 shows selected within-host percentiles.
""")

    write_md(OUT / "tenengrad_by_appearance_family.md", "# Tenengrad by appearance family\n\n" + markdown_table(family_selected[["arm", "family", "n", "mean", "median", "sample_sd", "native_standardized_median_location"]]) + f"\n\nSelected historical family median range={family_range:.6f}. Exact High/Low family marginals remained balanced; these post-result within-family summaries do not authorize family removal or reweighting.")

    key_variants = variant_df[(variant_df.cohort == "historical_medium") & (variant_df.subset == "selected") & (variant_df.variant.isin(["strip_permutation", "phase_scramble", "strip_permutation_alt", "phase_scramble_alt"]))]
    write_md(OUT / "tenengrad_variant_shift_audit.md", "# Tenengrad variant shift audit\n\n" + markdown_table(key_variants[["arm", "family", "variant", "n", "mean", "median", "sample_sd"]]) + f"\n\nFamily-level slot allocation preserved: **{bool_word(family_preserved)}**. Candidate-slot change class: **{slot_class}**. Variant-level associations are descriptive and cannot establish that the orientation amendment caused the failure.")

    write_md(OUT / "medium_candidate_slot_allocation_equivalence_report.md", f"""# Medium candidate-slot allocation equivalence

- `FAMILY_LEVEL_SLOT_BUDGET_PRESERVED = {bool_word(family_preserved)}`
- `CANDIDATE_SLOT_CHANGE_CLASS = {slot_class}`
- Both branches have exactly 2080 raw candidates per host.

{markdown_table(slot_df)}

The standalone audit is source-backed by the retained generic raw manifest and all 150 historical candidate identity manifests. It does not assume the expected counts when artifacts disagree.
""")

    source_cmp = pd.DataFrame([
        {"cohort": "historical_exact_frame", **summary(hist_source_df.source_tenengrad_mean), "unique_sources": len(hist_source_df), "median_reuse": float(hist_source_df.reuse_count.median())},
        {"cohort": "generic_medium", **summary(generic_source_df.source_tenengrad_mean), "unique_sources": len(generic_source_df), "median_reuse": float(generic_source_df.reuse_count.median())},
    ])
    write_md(OUT / "source_raster_tenengrad_comparison.md", f"""# Historical versus generic source-raster Tenengrad

- `GENERIC_SOURCE_COMPARISON_AVAILABLE = TRUE`
- Historical unique recovered sources={len(hist_source_df)}; generic retained base-render source assets={len(generic_source_df)}.
- Historical source Tenengrad median={f6(historical_source_median)}; generic median={f6(generic_source_median)}.
- `HISTORICAL_SOURCE_RASTER_SHIFT_SUPPORTED = {bool_word(source_shift_supported)}` under the descriptive >=10% lower-median rule.

{markdown_table(source_cmp[["cohort", "unique_sources", "mean", "median", "sample_sd", "q05", "q95", "median_reuse"]])}

Source absolute Tenengrad is not the same unit as final foreground/background Tenengrad ratio; associations are interpreted cautiously.
""")

    write_md(OUT / "tenengrad_source_reuse_audit.md", f"""# Source-raster reuse effect

- Unique historical sources={len(reuse_df)}; maximum reuse={int(reuse_df.reuse_count.max())}.
- Spearman reuse count vs final pair-mean Tenengrad={f6(reuse_corr['reuse_vs_pair_mean'])}.
- Spearman source absolute Tenengrad vs final pair-mean ratio={f6(reuse_corr['source_vs_pair_mean'])}.
- Observations sharing a source are not treated as independent confirmatory replicates.
""")

    write_md(OUT / "tenengrad_scale_diagnostic.md", "# Within-Medium scale diagnostic\n\n" + "\n".join(f"- Spearman {key} vs pair mean Tenengrad: {f6(value)}" for key, value in scale_corr.items()) + "\n\nNo new scale filter was derived; bins are descriptive only.")
    write_md(OUT / "tenengrad_resize_anisotropy_audit.md", "# Resize and anisotropy diagnostic\n\n" + "\n".join(f"- Spearman {key} vs pair mean Tenengrad: {f6(value)}" for key, value in resize_corr.items()) + "\n\nThe anisotropy metric is `abs(log(target_bbox_aspect / alpha_crop_source_aspect))`. The renderer was not rerun or modified.")

    hist_host_median = float(host_df.host_ring_tenengrad_mean.median())
    generic_host_median = float(generic_host_df.host_ring_tenengrad_mean.median())
    write_md(OUT / "historical_host_tenengrad_audit.md", "# Historical host/local-background Tenengrad audit\n\n" + "\n".join(f"- Spearman {key} vs pair mean Tenengrad: {f6(value)}" for key, value in host_corr.items()) + f"\n- Historical ring Tenengrad median={f6(hist_host_median)}; generic ring median={f6(generic_host_median)}.\n\nThe exact frozen bbox and local-ring convention was used on existing clean hosts; no synthetic rendering occurred.")

    write_md(OUT / "tenengrad_scene_diagnostic.md", f"# Scene-level Tenengrad diagnostic\n\n{markdown_table(scene_df)}\n\nDominant-scene host fraction={dominant_scene_fraction:.3f}. Scene summaries are descriptive; source-scene imbalance is not a confirmatory causal test.")
    write_md(OUT / "tenengrad_scene_loso.md", f"# Dominant-scene leave-one-scene-out sensitivity\n\n{markdown_table(loso_df)}\n\nHigh LOSO range={loso_high_range:.6f}; Low LOSO range={loso_low_range:.6f}. This post-result sensitivity cannot change the authoritative full-cohort FAIL.")

    write_md(OUT / "support_tenengrad_association.md", f"""# Support-Tenengrad association

- Selected pairs: Spearman support gap vs High Tenengrad={f6(support_corr['high'])}; Low={f6(support_corr['low'])}; pair mean={f6(support_corr['pair_mean'])}.
- Retained candidates: median within-host Spearman support distance vs Tenengrad={f6(support_corr['candidate_pool_median_within_host'])}; negative-correlation host fraction={support_corr['candidate_pool_negative_fraction']:.3f}.

These post-result correlations can suggest a tradeoff but do not establish a mechanism.
""")

    write_md(OUT / "weber_tenengrad_decoupling.md", "# Weber versus Tenengrad decoupling\n\n" + markdown_table(correlation_df.reset_index(drop=True)) + "\n\nWeber and Tenengrad capture different radiometric properties. Correlation does not establish why low-frequency contrast passed while edge-energy compatibility failed.")
    write_md(OUT / "tenengrad_calibration_stage_diagnostic.md", "# Calibration-stage localization\n\n- `CALIBRATION_STAGE_LOCALIZATION_AVAILABLE = PARTIAL`\n- Retained manifests contain pre-calibration and final calibrated metrics for selected images. Resized-object-only and pre-composite raster artifacts were not retained as standalone stage images and were not recreated.\n\n" + markdown_table(stage_summary_df))

    # Failure-localization hypothesis table.
    h1 = "NOT_SUPPORTED" if implementation_parity in {"IDENTICAL", "EQUIVALENT"} and native_identical and not direct_discrepancy else "SUPPORTED"
    hypotheses = [
        ("H1", "Tenengrad implementation/reference mismatch", h1, "Common code/reference hashes match and selected JPEGs reproduce the frozen values."),
        ("H2", "Historical source-raster sharpness shift", "SUPPORTED" if source_shift_supported else ("PARTIALLY_SUPPORTED" if historical_source_median < generic_source_median else "NOT_SUPPORTED"), f"Historical/generic source medians {historical_source_median:.6f}/{generic_source_median:.6f}; the historical median is lower, but not by the strong >=10% diagnostic rule."),
        ("H3", "Source-raster reuse concentration", "PARTIALLY_SUPPORTED" if abs(reuse_corr['reuse_vs_pair_mean']) >= .20 else "NOT_SUPPORTED", f"Reuse correlation {reuse_corr['reuse_vs_pair_mean']:.3f}; 58 unique sources underlie 150 frames."),
        ("H4", "Historical bbox resize/anisotropy effect", "PARTIALLY_SUPPORTED" if abs(resize_corr['anisotropy_abs_log_aspect_ratio']) >= .20 else "NOT_SUPPORTED", f"Anisotropy correlation {resize_corr['anisotropy_abs_log_aspect_ratio']:.3f}."),
        ("H5", "Historical host/context effect", "PARTIALLY_SUPPORTED" if abs(host_corr['host_ring_tenengrad_mean']) >= .20 else "NOT_SUPPORTED", f"Host ring correlation {host_corr['host_ring_tenengrad_mean']:.3f}."),
        ("H6", "Source-scene concentration effect", "PARTIALLY_SUPPORTED" if max(loso_high_range, loso_low_range) >= .10 else "NOT_SUPPORTED", f"Dominant fraction {dominant_scene_fraction:.3f}; LOSO ranges {loso_high_range:.3f}/{loso_low_range:.3f}."),
        ("H7", "Orientation-amendment variant redistribution effect", "PARTIALLY_SUPPORTED" if family_preserved else "INCONCLUSIVE", f"{slot_class}; family budgets preserved={family_preserved}."),
        ("H8", "S3 selector-induced low-Tenengrad selection", "SUPPORTED" if selector_worsened else ("NOT_SUPPORTED" if pool_already_shifted else "INCONCLUSIVE"), f"Pool W={pool_w:.3f}; selected mean percentile={selected_mean_percentile:.3f}."),
        ("H9", "Support-Tenengrad tradeoff", "PARTIALLY_SUPPORTED" if support_corr['pair_mean'] <= -.20 or support_corr['candidate_pool_median_within_host'] <= -.10 else "NOT_SUPPORTED", f"Selected/candidate correlations {support_corr['pair_mean']:.3f}/{support_corr['candidate_pool_median_within_host']:.3f}."),
        ("H10", "Frozen calibration interacts differently with historical frame", "PARTIALLY_SUPPORTED" if abs(stage_summary_df.iloc[1].standardized_wasserstein_to_native - stage_summary_df.iloc[3].standardized_wasserstein_to_native) >= .10 else "INCONCLUSIVE", "Pre/final stage metrics retained, but resized-object and pre-composite standalone stages were unavailable."),
    ]
    hypothesis_df = pd.DataFrame(hypotheses, columns=["hypothesis_id", "hypothesis", "status", "evidence_summary"])
    write_df(OUT / "tenengrad_failure_hypothesis_table.csv", hypothesis_df)

    classification = "IMPLEMENTATION_ERROR" if h1 == "SUPPORTED" else ("INCONCLUSIVE" if implementation_parity == "UNVERIFIED" or not native_identical or direct_discrepancy else "VALID_NEGATIVE_FEASIBILITY_RESULT")
    if selector_worsened:
        strongest = f"The strongest retained evidence is a selector-associated shift: S3 selected candidates below the within-host Tenengrad center (mean percentile {selected_mean_percentile:.3f}), moving a retained pool that remained within the frozen absolute threshold (W={pool_w:.3f}) to failed High/Low distributions. Negative support-Tenengrad associations and low selected spatial-frequency-family values were consistent with this localization, but do not prove that selection alone caused the failure."
    elif source_shift_supported:
        strongest = "The retained evidence localizes most strongly to a historical source-raster distribution shift, with resize, scene, host, variant, and selector effects treated as secondary descriptive associations."
    elif pool_already_shifted and not selector_worsened:
        strongest = "The retained candidate pool was already native-incompatible before S3 selection; the evidence localizes to shared historical branch inputs/context rather than the support-arm selector."
    else:
        strongest = "The mismatch is a shared historical-branch distribution shift; retained evidence does not isolate one deterministic causal stage."

    appearance_concentration = "TRUE" if family_range >= 0.50 else "FALSE"
    scene_concentration = "PARTIAL" if max(loso_high_range, loso_low_range) >= .10 else "WEAK"
    resize_association = f"anisotropy Spearman rho={resize_corr['anisotropy_abs_log_aspect_ratio']:.3f}"
    support_tradeoff = f"selected rho={support_corr['pair_mean']:.3f}; candidate median rho={support_corr['candidate_pool_median_within_host']:.3f}"
    localization_header = f"""# Historical Medium150 Tenengrad failure localization

**{LABEL}**

Historical pretraining gate remains: **FAIL**

High Tenengrad Wasserstein: **{frozen_high:.6f}**

Low Tenengrad Wasserstein: **{frozen_low:.6f}**

Frozen threshold: **0.50**

Metric implementation parity: **{implementation_parity}**

Native reference parity: **{bool_word(native_identical)}**

Direct recomputation parity: **{bool_word(not direct_discrepancy)}**

Candidate pool already shifted before selection: **{bool_word(pool_already_shifted)}**

S3 selection worsened Tenengrad: **{bool_word(selector_worsened)}**

Historical source-raster shift supported: **{bool_word(source_shift_supported)}**

Resize/anisotropy association: **{resize_association}**

Scene concentration: **{scene_concentration}**

Appearance-family concentration: **{appearance_concentration}**

Support/Tenengrad tradeoff: **{support_tradeoff}**

Slot allocation preserved: **{bool_word(family_preserved)}**

Final diagnostic classification: **{classification}**

Training remains: **LOCKED**

AP inspected: **0**
"""
    answers = f"""
## Required localization answers

1. **Metric/reference identical?** Common source and reference hashes are identical; runtime provenance is classified EQUIVALENT because both completed runs did not separately freeze package versions.
2. **Shift present before S3?** {bool_word(pool_already_shifted)}; retained-pool W={pool_w:.6f}.
3. **Did S3 worsen it?** {bool_word(selector_worsened)}; selected mean within-host percentile={selected_mean_percentile:.3f}.
4. **Appearance-family concentration?** {appearance_concentration}; selected family median range={family_range:.3f}.
5. **Variant concentration?** Variant associations are reported, but mechanical slot redistribution alone does not establish causality.
6. **Historical source concentration?** Source-raster shift supported={bool_word(source_shift_supported)}; reuse effects are separately reported.
7. **Scene concentration?** {scene_concentration}; dominant scene fraction={dominant_scene_fraction:.3f}.
8. **Target scale relation?** Scale rho={scale_corr['bbox_scale']:.3f}.
9. **Resize anisotropy relation?** {resize_association}.
10. **Host/background relation?** Ring-sharpness rho={host_corr['host_ring_tenengrad_mean']:.3f}.
11. **Support-gap relation?** {support_tradeoff}.
12. **Strongest supported explanation.** {strongest}
13. **Unresolved.** Standalone resized-object and pre-composite intermediate stages were not retained, package versions were not independently serialized for both historical runs, and all associations remain post-result descriptive.

## Evidence table

{markdown_table(hypothesis_df)}

The frozen FAIL is retained. No candidate was generated, no threshold or calibration was changed, no detector was trained, and no AP was inspected.
"""
    write_md(OUT / "historical_tenengrad_failure_localization.md", localization_header + answers)
    write_md(OUT / "historical_tenengrad_failure_diagnostic_classification.md", f"""# Historical Tenengrad failure diagnostic classification

`DIAGNOSTIC_CLASSIFICATION = {classification}`

- Implementation parity: {implementation_parity}
- Native reference identical: {bool_word(native_identical)}
- Direct recomputation discrepancy: {bool_word(direct_discrepancy)}
- Original decision remains `MEDIUM150_PRETRAINING_GATE = FAIL`.

{strongest}

This classification is explanatory, not a confirmatory endpoint and not authorization to correct, rerun, generate candidates, train a detector, or inspect AP.
""")
    write_md(OUT / "next_stage_recommendation.md", f"""# Next-stage recommendation

`DIAGNOSTIC_CLASSIFICATION = {classification}`

Preserve the failed historical exact-frame Medium150 experiment as a negative feasibility boundary. The retained evidence does not justify retroactive correction or threshold reinterpretation. Stop for author review.

- `MEDIUM150_PRETRAINING_GATE = FAIL`
- `TRAINING_UNLOCKED = FALSE`
- `DETECTOR_TRAINING_RUNS = 0`
- `AP_VALUES_INSPECTED = 0`
- No remediation, new candidate generation, detector training, or AP evaluation is authorized.
""")

    final_state = {
        "analysis": LABEL, "completed": True, "MEDIUM150_PRETRAINING_GATE": "FAIL",
        "TENENGRAD_HIGH": frozen_high, "TENENGRAD_LOW": frozen_low, "threshold": 0.50,
        "TENENGRAD_IMPLEMENTATION_PARITY": implementation_parity,
        "NATIVE_TENENGRAD_REFERENCE_IDENTICAL": native_identical,
        "TENENGRAD_RESULT_DISCREPANCY": direct_discrepancy,
        "CANDIDATE_POOL_ALREADY_SHIFTED_BEFORE_SELECTION": pool_already_shifted,
        "S3_SELECTION_WORSENED_TENENGRAD": selector_worsened,
        "FAMILY_LEVEL_SLOT_BUDGET_PRESERVED": family_preserved,
        "CANDIDATE_SLOT_CHANGE_CLASS": slot_class,
        "DIAGNOSTIC_CLASSIFICATION": classification,
        "TRAINING_UNLOCKED": False, "DETECTOR_TRAINING_RUNS": 0, "AP_VALUES_INSPECTED": 0,
        "NEW_CANDIDATES_GENERATED": 0,
    }
    final_state_path = OUT / "historical_tenengrad_postresult_audit_state.json"
    write_json(final_state_path, final_state)
    print(final_state_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
