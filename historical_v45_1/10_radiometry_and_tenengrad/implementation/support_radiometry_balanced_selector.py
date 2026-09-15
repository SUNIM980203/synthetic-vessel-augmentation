"""Deterministic cohort selector for the balanced Medium feasibility study.

This module never renders candidates, trains a detector, evaluates AP, or
opens a scale branch.  It selects one exact-geometry pair per host from an
already frozen 2,048-candidate pool.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from support_scale_radiometry_calibration_common import (
    METRICS,
    evaluate_selected_pairs,
)


FORMULATIONS = (
    "S0_old_support_max",
    "S1_weber_quintile",
    "S2_weber_quintile_family_balance",
    "S3_direct_wasserstein_family_balance",
)

# Defined from the transformation implementation, before balanced selection.
APPEARANCE_FAMILY_BY_VARIANT = {
    "original": "baseline",
    "major_axis_reflection": "texture_realization",
    "major_axis_reflection_strong": "texture_realization",
    "strip_permutation": "texture_realization",
    "strip_permutation_alt": "texture_realization",
    "phase_scramble": "texture_realization",
    "phase_scramble_alt": "texture_realization",
    "longitudinal_phase_0": "spatial_frequency",
    "longitudinal_phase_pi2": "spatial_frequency",
    "longitudinal_phase_pi": "spatial_frequency",
    "longitudinal_phase_3pi2": "spatial_frequency",
    "diagonal_crosshatch": "spatial_frequency",
    "chroma_microstructure": "material_chroma",
    "chroma_microstructure_alt": "material_chroma",
    "microcontrast_detail": "fine_detail",
    "microcontrast_detail_soft": "fine_detail",
}

FAMILY_ORDER = (
    "baseline",
    "texture_realization",
    "spatial_frequency",
    "material_chroma",
    "fine_detail",
)

SUPPORT_MINIMUM = 0.005
SUPPORT_FRACTION_THRESHOLD = 0.040
SUPPORT_FRACTION_REQUIRED = 24
# A sufficient linear condition for the 30-sample median to be >= 0.060.
SUPPORT_MEDIAN_THRESHOLD = 0.060
SUPPORT_MEDIAN_REQUIRED = 16
PAIR_OCCUPANCY_MAX = 0.05
PAIR_METRIC_STANDARDIZED_MAX = 0.15
WEBER_WASSERSTEIN_MAX = 0.50
DIRECT_WEBER_ARM_MEAN_FLOOR = 0.50
SOLVER_TIME_LIMIT_SECONDS = 90.0
SOLVER_MIP_REL_GAP = 0.01


@dataclass(frozen=True)
class SolverAudit:
    success: bool
    status: int
    message: str
    mip_gap: float | None
    objective: float | None
    selected_count: int


def taxonomy_audit(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    observed = sorted({str(row["appearance_variant"]) for row in candidates})
    missing = [name for name in observed if name not in APPEARANCE_FAMILY_BY_VARIANT]
    if missing:
        raise ValueError(f"Unclassified appearance variants: {missing}")
    mismatches = []
    for row in candidates:
        expected = APPEARANCE_FAMILY_BY_VARIANT[str(row["appearance_variant"])]
        if str(row["appearance_family"]) != expected:
            mismatches.append(int(row["candidate_id"]))
    return {
        "observed_variants": observed,
        "all_variants_classified": not missing,
        "manifest_family_mismatch_count": len(mismatches),
        "manifest_family_mismatch_candidate_ids": mismatches[:20],
        "family_order": list(FAMILY_ORDER),
        "rule": "single primary family determined only by transformation implementation name",
    }


def _support_category(gap: float) -> int:
    if gap >= SUPPORT_MEDIAN_THRESHOLD:
        return 2
    if gap >= SUPPORT_FRACTION_THRESHOLD:
        return 1
    return 0


def _pair_row(
    host: int,
    left: dict[str, Any],
    right: dict[str, Any],
    left_distance: float,
    right_distance: float,
    native_summary: dict[str, dict[str, float]],
    quintile_edges: np.ndarray,
) -> dict[str, Any]:
    high, low = (left, right) if left_distance <= right_distance else (right, left)
    high_distance, low_distance = sorted((left_distance, right_distance))
    gap = low_distance - high_distance
    standardized = {
        key: abs(
            float(left["calibrated_metrics"][key])
            - float(right["calibrated_metrics"][key])
        )
        / max(float(native_summary[key]["sample_sd"]), 1e-12)
        for key in METRICS
    }
    return {
        "host_index": host,
        "near_candidate_id": int(high["candidate_id"]),
        "far_candidate_id": int(low["candidate_id"]),
        "near_distance": float(high_distance),
        "far_distance": float(low_distance),
        "support_gap": float(gap),
        "high_family": APPEARANCE_FAMILY_BY_VARIANT[str(high["appearance_variant"])],
        "low_family": APPEARANCE_FAMILY_BY_VARIANT[str(low["appearance_variant"])],
        "high_variant": str(high["appearance_variant"]),
        "low_variant": str(low["appearance_variant"]),
        "high_metrics": {key: float(high["calibrated_metrics"][key]) for key in METRICS},
        "low_metrics": {key: float(low["calibrated_metrics"][key]) for key in METRICS},
        "high_weber_bin": int(
            np.searchsorted(
                quintile_edges,
                float(high["calibrated_metrics"]["absolute_weber_contrast"]),
                side="right",
            )
        ),
        "low_weber_bin": int(
            np.searchsorted(
                quintile_edges,
                float(low["calibrated_metrics"]["absolute_weber_contrast"]),
                side="right",
            )
        ),
        "exact_geometry_orientation": True,
        "absolute_occupancy_difference": abs(
            float(left["alpha_occupancy"]) - float(right["alpha_occupancy"])
        ),
        "standardized_metric_differences": standardized,
    }


def build_pair_options(
    candidates: list[dict[str, Any]],
    distances: dict[int, float],
    native: np.ndarray,
    native_summary: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Enumerate all exact-orientation pairs and retain fixed representatives.

    The full eligible set is used for diagnostic counts.  To make the MILP
    finite and reproducible, each host/signature retains at most six distinct
    representatives: two largest-gap, two largest summed-Weber, and two
    largest minimum-arm-Weber pairs.  The rule is fixed for both development
    and fresh validation.
    """
    taxonomy = taxonomy_audit(candidates)
    quintile_edges = np.quantile(native[:, 0], (0.20, 0.40, 0.60, 0.80))
    by_host_orientation: dict[int, dict[bool, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in candidates:
        candidate_id = int(row["candidate_id"])
        if candidate_id not in distances:
            raise KeyError(f"Missing support distance for candidate {candidate_id}")
        by_host_orientation[int(row["host_index"])][bool(row.get("rotated_90", False))].append(row)

    retained: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    host_option_counts: dict[int, int] = {}
    total_full_eligible = 0

    for host in sorted(by_host_orientation):
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        support_top: list[tuple[float, int, int, dict[str, Any]]] = []
        native_top: list[tuple[float, float, int, int, dict[str, Any]]] = []
        available_weber: list[float] = []
        native_like_count = 0
        support_004_count = 0
        both_count = 0
        eligible_count = 0

        for orientation in sorted(by_host_orientation[host]):
            rows = sorted(
                by_host_orientation[host][orientation],
                key=lambda row: int(row["candidate_id"]),
            )
            count = len(rows)
            metric_values = np.asarray(
                [[float(row["calibrated_metrics"][key]) for key in METRICS] for row in rows],
                dtype=np.float64,
            )
            occupancy = np.asarray([float(row["alpha_occupancy"]) for row in rows])
            support = np.asarray([distances[int(row["candidate_id"])] for row in rows])
            eligible = np.triu(np.ones((count, count), dtype=bool), 1)
            eligible &= np.abs(occupancy[:, None] - occupancy[None, :]) <= PAIR_OCCUPANCY_MAX
            for metric_index, key in enumerate(METRICS):
                tolerance = PAIR_METRIC_STANDARDIZED_MAX * max(
                    float(native_summary[key]["sample_sd"]), 1e-12
                )
                eligible &= (
                    np.abs(
                        metric_values[:, metric_index, None]
                        - metric_values[None, :, metric_index]
                    )
                    <= tolerance
                )
            gap_matrix = np.abs(support[:, None] - support[None, :])
            eligible &= gap_matrix >= SUPPORT_MINIMUM
            left_indices, right_indices = np.where(eligible)
            eligible_count += len(left_indices)
            total_full_eligible += len(left_indices)

            for left_index, right_index in zip(left_indices.tolist(), right_indices.tolist()):
                left, right = rows[left_index], rows[right_index]
                pair = _pair_row(
                    host,
                    left,
                    right,
                    float(support[left_index]),
                    float(support[right_index]),
                    native_summary,
                    quintile_edges,
                )
                gap = float(pair["support_gap"])
                high_weber = float(pair["high_metrics"]["absolute_weber_contrast"])
                low_weber = float(pair["low_metrics"]["absolute_weber_contrast"])
                available_weber.extend((high_weber, low_weber))
                native_like = bool(
                    float(native_summary["absolute_weber_contrast"]["q1"])
                    <= high_weber
                    <= float(native_summary["absolute_weber_contrast"]["q3"])
                    and float(native_summary["absolute_weber_contrast"]["q1"])
                    <= low_weber
                    <= float(native_summary["absolute_weber_contrast"]["q3"])
                )
                native_like_count += int(native_like)
                support_004_count += int(gap >= SUPPORT_FRACTION_THRESHOLD)
                both_count += int(native_like and gap >= SUPPORT_FRACTION_THRESHOLD)
                support_top.append(
                    (
                        -gap,
                        int(pair["near_candidate_id"]),
                        int(pair["far_candidate_id"]),
                        pair,
                    )
                )
                native_cost = abs(
                    high_weber - float(native_summary["absolute_weber_contrast"]["median"])
                ) + abs(
                    low_weber - float(native_summary["absolute_weber_contrast"]["median"])
                )
                native_top.append(
                    (
                        native_cost,
                        -gap,
                        int(pair["near_candidate_id"]),
                        int(pair["far_candidate_id"]),
                        pair,
                    )
                )
                signature = (
                    int(pair["high_weber_bin"]),
                    int(pair["low_weber_bin"]),
                    str(pair["high_family"]),
                    str(pair["low_family"]),
                    _support_category(gap),
                )
                buckets[signature].append(pair)

        if not support_top:
            failure_rows.append(
                {
                    "host_index": host,
                    "eligible_pair_count": 0,
                    "status": "NO_PAIR_LEVEL_FEASIBLE_PAIR",
                }
            )
            continue

        support_top.sort(key=lambda row: row[:3])
        native_top.sort(key=lambda row: row[:4])
        for rows in buckets.values():
            chosen_ids: set[tuple[int, int]] = set()
            orderings = (
                sorted(
                    rows,
                    key=lambda row: (
                        -float(row["support_gap"]),
                        int(row["near_candidate_id"]),
                        int(row["far_candidate_id"]),
                    ),
                )[:2],
                sorted(
                    rows,
                    key=lambda row: (
                        -(
                            float(row["high_metrics"]["absolute_weber_contrast"])
                            + float(row["low_metrics"]["absolute_weber_contrast"])
                        ),
                        -float(row["support_gap"]),
                        int(row["near_candidate_id"]),
                    ),
                )[:2],
                sorted(
                    rows,
                    key=lambda row: (
                        -min(
                            float(row["high_metrics"]["absolute_weber_contrast"]),
                            float(row["low_metrics"]["absolute_weber_contrast"]),
                        ),
                        -float(row["support_gap"]),
                        int(row["near_candidate_id"]),
                    ),
                )[:2],
            )
            for ordering in orderings:
                for pair in ordering:
                    key = (int(pair["near_candidate_id"]), int(pair["far_candidate_id"]))
                    if key not in chosen_ids:
                        retained.append(pair)
                        chosen_ids.add(key)

        host_option_counts[host] = sum(1 for row in retained if int(row["host_index"]) == host)
        best_support = support_top[0][3]
        second_support = support_top[1][3] if len(support_top) > 1 else support_top[0][3]
        best_native = native_top[0][4]
        failure_rows.append(
            {
                "host_index": host,
                "eligible_pair_count": eligible_count,
                "retained_optimizer_option_count": host_option_counts[host],
                "support_max_near_candidate_id": int(best_support["near_candidate_id"]),
                "support_max_far_candidate_id": int(best_support["far_candidate_id"]),
                "support_max_gap": float(best_support["support_gap"]),
                "support_max_high_weber": float(best_support["high_metrics"]["absolute_weber_contrast"]),
                "support_max_low_weber": float(best_support["low_metrics"]["absolute_weber_contrast"]),
                "second_best_gap": float(second_support["support_gap"]),
                "native_matched_near_candidate_id": int(best_native["near_candidate_id"]),
                "native_matched_far_candidate_id": int(best_native["far_candidate_id"]),
                "native_matched_gap": float(best_native["support_gap"]),
                "native_matched_high_weber": float(best_native["high_metrics"]["absolute_weber_contrast"]),
                "native_matched_low_weber": float(best_native["low_metrics"]["absolute_weber_contrast"]),
                "available_weber_min": float(min(available_weber)),
                "available_weber_max": float(max(available_weber)),
                "native_like_weber_pair_count": native_like_count,
                "support_gap_ge_0_04_pair_count": support_004_count,
                "native_like_and_gap_ge_0_04_pair_count": both_count,
                "status": "PASS",
            }
        )

    retained.sort(
        key=lambda row: (
            int(row["host_index"]),
            int(row["near_candidate_id"]),
            int(row["far_candidate_id"]),
        )
    )
    audit = {
        "full_pair_level_eligible_count": total_full_eligible,
        "retained_optimizer_option_count": len(retained),
        "host_option_counts": {str(key): value for key, value in sorted(host_option_counts.items())},
        "native_weber_quintile_edges": [float(value) for value in quintile_edges],
        "taxonomy": taxonomy,
        "representative_rule": (
            "per host/quintile-bin pair/High family/Low family/support category: "
            "top two support, top two summed Weber, top two minimum-arm Weber; candidate-ID tie-break"
        ),
    }
    return retained, failure_rows, audit


def appearance_balance(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    high = Counter(str(row["high_family"]) for row in pairs)
    low = Counter(str(row["low_family"]) for row in pairs)
    rows = []
    for family in FAMILY_ORDER:
        rows.append(
            {
                "family": family,
                "high_count": int(high[family]),
                "low_count": int(low[family]),
                "difference": int(high[family] - low[family]),
            }
        )
    n = max(len(pairs), 1)
    tvd = 0.5 * sum(abs(high[family] - low[family]) / n for family in FAMILY_ORDER)
    maximum = max((abs(row["difference"]) for row in rows), default=0)
    return {
        "rows": rows,
        "total_variation_distance": float(tvd),
        "maximum_absolute_count_difference": int(maximum),
        "criterion": "exact marginal count equality for every frozen family",
        "pass": bool(len(pairs) == 30 and maximum == 0),
    }


def _solve(
    options: list[dict[str, Any]],
    native_summary: dict[str, dict[str, float]],
    formulation: str,
) -> tuple[list[dict[str, Any]], SolverAudit]:
    if formulation not in FORMULATIONS[1:]:
        raise ValueError(f"Unsupported MILP formulation {formulation}")
    count = len(options)
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    constraint_index = 0

    def add(indices: list[int], coefficients: list[float], low: float, high: float) -> None:
        nonlocal constraint_index
        for index, coefficient in zip(indices, coefficients):
            matrix_rows.append(constraint_index)
            matrix_columns.append(index)
            matrix_data.append(float(coefficient))
        lower.append(float(low))
        upper.append(float(high))
        constraint_index += 1

    hosts = sorted({int(row["host_index"]) for row in options})
    for host in hosts:
        indices = [index for index, row in enumerate(options) if int(row["host_index"]) == host]
        add(indices, [1.0] * len(indices), 1.0, 1.0)

    if formulation in (
        "S1_weber_quintile",
        "S2_weber_quintile_family_balance",
    ):
        for bin_index in range(5):
            for arm in ("high", "low"):
                indices = [
                    index
                    for index, row in enumerate(options)
                    if int(row[f"{arm}_weber_bin"]) == bin_index
                ]
                add(indices, [1.0] * len(indices), 6.0, 6.0)

    if formulation in (
        "S2_weber_quintile_family_balance",
        "S3_direct_wasserstein_family_balance",
    ):
        for family in FAMILY_ORDER:
            indices, coefficients = [], []
            for index, row in enumerate(options):
                coefficient = int(str(row["high_family"]) == family) - int(
                    str(row["low_family"]) == family
                )
                if coefficient:
                    indices.append(index)
                    coefficients.append(float(coefficient))
            add(indices, coefficients, 0.0, 0.0)

    indices = [
        index
        for index, row in enumerate(options)
        if float(row["support_gap"]) >= SUPPORT_FRACTION_THRESHOLD
    ]
    add(indices, [1.0] * len(indices), float(SUPPORT_FRACTION_REQUIRED), np.inf)
    indices = [
        index
        for index, row in enumerate(options)
        if float(row["support_gap"]) >= SUPPORT_MEDIAN_THRESHOLD
    ]
    add(indices, [1.0] * len(indices), float(SUPPORT_MEDIAN_REQUIRED), np.inf)

    if formulation == "S3_direct_wasserstein_family_balance":
        all_indices = list(range(count))
        for arm in ("high", "low"):
            add(
                all_indices,
                [
                    float(row[f"{arm}_metrics"]["absolute_weber_contrast"])
                    for row in options
                ],
                30.0 * DIRECT_WEBER_ARM_MEAN_FLOOR,
                np.inf,
            )
            # Frozen deterministic feasibility envelope for every preserved
            # radiometry metric.  Exact Wasserstein/IQR gates are then applied
            # to the returned assignment as hard acceptance criteria.
            for metric in METRICS:
                q1 = float(native_summary[metric]["q1"])
                q3 = float(native_summary[metric]["q3"])
                above = [
                    index
                    for index, row in enumerate(options)
                    if float(row[f"{arm}_metrics"][metric]) >= q1
                ]
                below = [
                    index
                    for index, row in enumerate(options)
                    if float(row[f"{arm}_metrics"][metric]) <= q3
                ]
                add(above, [1.0] * len(above), 16.0, np.inf)
                add(below, [1.0] * len(below), 16.0, np.inf)

    matrix = coo_matrix(
        (matrix_data, (matrix_rows, matrix_columns)),
        shape=(constraint_index, count),
    ).tocsr()
    objective = np.asarray(
        [
            -float(row["support_gap"])
            + 1e-11 * float(row["near_candidate_id"])
            + 1e-16 * float(row["far_candidate_id"])
            for row in options
        ],
        dtype=np.float64,
    )
    result = milp(
        objective,
        integrality=np.ones(count),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={
            "time_limit": SOLVER_TIME_LIMIT_SECONDS,
            "mip_rel_gap": SOLVER_MIP_REL_GAP,
            "presolve": True,
        },
    )
    selected = []
    if result.x is not None:
        selected = [row for row, value in zip(options, result.x) if value > 0.5]
        selected.sort(key=lambda row: int(row["host_index"]))
    audit = SolverAudit(
        success=bool(result.success and len(selected) == len(hosts)),
        status=int(result.status),
        message=str(result.message),
        mip_gap=float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
        objective=float(result.fun) if result.fun is not None else None,
        selected_count=len(selected),
    )
    return selected, audit


def run_formulation(
    options: list[dict[str, Any]],
    candidates_by_id: dict[int, dict[str, Any]],
    native: np.ndarray,
    native_summary: dict[str, dict[str, float]],
    formulation: str,
) -> dict[str, Any]:
    selected, solver = _solve(options, native_summary, formulation)
    output: dict[str, Any] = {
        "formulation": formulation,
        "solver": solver.__dict__,
        "selected_pairs": selected,
        "infeasible": not solver.success,
        "joint_pass": False,
    }
    if not solver.success:
        return output
    evaluation = evaluate_selected_pairs(selected, candidates_by_id, native, native_summary)
    balance = appearance_balance(selected)
    exact_direct_weber = bool(
        evaluation["radiometry_gate"]["relative_near"]["metrics"]["absolute_weber_contrast"]["standardized_wasserstein"]
        <= WEBER_WASSERSTEIN_MAX
        and evaluation["radiometry_gate"]["relative_far"]["metrics"]["absolute_weber_contrast"]["standardized_wasserstein"]
        <= WEBER_WASSERSTEIN_MAX
    )
    balance_required = formulation in (
        "S2_weber_quintile_family_balance",
        "S3_direct_wasserstein_family_balance",
    )
    joint = bool(
        evaluation["support_gate"]["pass"]
        and evaluation["radiometry_gate"]["pass"]
        and exact_direct_weber
        and (balance["pass"] if balance_required else True)
    )
    output.update(
        {
            "evaluation": evaluation,
            "appearance_balance": balance,
            "exact_direct_weber_acceptance": exact_direct_weber,
            "joint_pass": joint,
        }
    )
    return output


def selected_pair_rows(
    pairs: list[dict[str, Any]],
    candidates_by_id: dict[int, dict[str, Any]],
    original_host_offset: int,
) -> list[dict[str, Any]]:
    rows = []
    for pair in sorted(pairs, key=lambda row: int(row["host_index"])):
        high = candidates_by_id[int(pair["near_candidate_id"])]
        low = candidates_by_id[int(pair["far_candidate_id"])]
        rows.append(
            {
                "local_host_id": int(pair["host_index"]),
                "original_v47_host_id": original_host_offset + int(pair["host_index"]),
                "high_candidate_id": int(pair["near_candidate_id"]),
                "low_candidate_id": int(pair["far_candidate_id"]),
                "high_support_distance": float(pair["near_distance"]),
                "low_support_distance": float(pair["far_distance"]),
                "support_gap": float(pair["support_gap"]),
                "high_appearance_variant": str(high["appearance_variant"]),
                "low_appearance_variant": str(low["appearance_variant"]),
                "high_appearance_family": str(pair["high_family"]),
                "low_appearance_family": str(pair["low_family"]),
                "exact_geometry_orientation": bool(pair["exact_geometry_orientation"]),
                "absolute_occupancy_difference": float(pair["absolute_occupancy_difference"]),
                "pairwise_radiometry_pass": bool(
                    all(
                        float(value) <= PAIR_METRIC_STANDARDIZED_MAX
                        for value in pair["standardized_metric_differences"].values()
                    )
                ),
                **{
                    f"high_{key}": float(high["calibrated_metrics"][key])
                    for key in METRICS
                },
                **{
                    f"low_{key}": float(low["calibrated_metrics"][key])
                    for key in METRICS
                },
            }
        )
    return rows
