"""Mechanical 150-host instantiation of the frozen S3 selector.

Scientific thresholds, pair-option construction, family taxonomy, objective,
and solver controls are imported from the frozen 30-host implementation.
Only cardinalities that encode the cohort size are instantiated at N=150.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from support_radiometry_balanced_selector import (
    DIRECT_WEBER_ARM_MEAN_FLOOR,
    FAMILY_ORDER,
    PAIR_METRIC_STANDARDIZED_MAX,
    SOLVER_MIP_REL_GAP,
    SOLVER_TIME_LIMIT_SECONDS,
    SUPPORT_FRACTION_THRESHOLD,
    SUPPORT_MEDIAN_THRESHOLD,
    WEBER_WASSERSTEIN_MAX,
    SolverAudit,
)
from support_scale_radiometry_calibration_common import (
    FROZEN_THRESHOLDS,
    METRICS,
    distribution_gate,
)


EXPECTED_HOSTS = 150
FRACTION_REQUIRED = 120  # exact 0.80 * 150
MEDIAN_SUFFICIENT_REQUIRED = 76  # same strict-majority linearization as 16/30
NATIVE_IQR_MEDIAN_REQUIRED = 76


def appearance_balance_medium150(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    high = Counter(str(row["high_family"]) for row in pairs)
    low = Counter(str(row["low_family"]) for row in pairs)
    rows = [
        {
            "family": family,
            "high_count": int(high[family]),
            "low_count": int(low[family]),
            "difference": int(high[family] - low[family]),
        }
        for family in FAMILY_ORDER
    ]
    n = max(len(pairs), 1)
    tvd = 0.5 * sum(abs(high[family] - low[family]) / n for family in FAMILY_ORDER)
    maximum = max((abs(row["difference"]) for row in rows), default=0)
    return {
        "rows": rows,
        "total_variation_distance": float(tvd),
        "maximum_absolute_count_difference": int(maximum),
        "criterion": "exact marginal High/Low count equality for every frozen remaining family",
        "pass": bool(len(pairs) == EXPECTED_HOSTS and maximum == 0),
    }


def solve_s3_medium150(
    options: list[dict[str, Any]], native_summary: dict[str, dict[str, float]]
) -> tuple[list[dict[str, Any]], SolverAudit]:
    hosts = sorted({int(row["host_index"]) for row in options})
    if hosts != list(range(1, EXPECTED_HOSTS + 1)):
        raise ValueError(f"Expected complete host indices 1..150, got {len(hosts)}")
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

    for host in hosts:
        indices = [index for index, row in enumerate(options) if int(row["host_index"]) == host]
        if not indices:
            raise ValueError(f"No S3 option for host {host}")
        add(indices, [1.0] * len(indices), 1.0, 1.0)

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
        index for index, row in enumerate(options)
        if float(row["support_gap"]) >= SUPPORT_FRACTION_THRESHOLD
    ]
    add(indices, [1.0] * len(indices), FRACTION_REQUIRED, np.inf)
    indices = [
        index for index, row in enumerate(options)
        if float(row["support_gap"]) >= SUPPORT_MEDIAN_THRESHOLD
    ]
    add(indices, [1.0] * len(indices), MEDIAN_SUFFICIENT_REQUIRED, np.inf)

    all_indices = list(range(count))
    for arm in ("high", "low"):
        add(
            all_indices,
            [float(row[f"{arm}_metrics"]["absolute_weber_contrast"]) for row in options],
            EXPECTED_HOSTS * DIRECT_WEBER_ARM_MEAN_FLOOR,
            np.inf,
        )
        for metric in METRICS:
            q1 = float(native_summary[metric]["q1"])
            q3 = float(native_summary[metric]["q3"])
            above = [
                index for index, row in enumerate(options)
                if float(row[f"{arm}_metrics"][metric]) >= q1
            ]
            below = [
                index for index, row in enumerate(options)
                if float(row[f"{arm}_metrics"][metric]) <= q3
            ]
            add(above, [1.0] * len(above), NATIVE_IQR_MEDIAN_REQUIRED, np.inf)
            add(below, [1.0] * len(below), NATIVE_IQR_MEDIAN_REQUIRED, np.inf)

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
    selected: list[dict[str, Any]] = []
    if result.x is not None:
        selected = [row for row, value in zip(options, result.x) if value > 0.5]
        selected.sort(key=lambda row: int(row["host_index"]))
    audit = SolverAudit(
        success=bool(result.success and len(selected) == EXPECTED_HOSTS),
        status=int(result.status),
        message=str(result.message),
        mip_gap=float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
        objective=float(result.fun) if result.fun is not None else None,
        selected_count=len(selected),
    )
    return selected, audit


def evaluate_medium150(
    pairs: list[dict[str, Any]],
    candidates: dict[int, dict[str, Any]],
    native: np.ndarray,
    native_summary: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not pairs:
        return {"support_gate": {"pass": False}, "radiometry_gate": {"pass": False}, "joint_pass": False}
    high = np.asarray(
        [[float(candidates[int(row["near_candidate_id"])]["calibrated_metrics"][key]) for key in METRICS] for row in pairs]
    )
    low = np.asarray(
        [[float(candidates[int(row["far_candidate_id"])]["calibrated_metrics"][key]) for key in METRICS] for row in pairs]
    )
    gaps = np.asarray([float(row["support_gap"]) for row in pairs])
    support = {
        "selected_host_count": len(pairs),
        "all_required_hosts_matched": len(pairs) == EXPECTED_HOSTS,
        "median_gap": float(np.median(gaps)),
        "fraction_gap_ge_0_04": float(np.mean(gaps >= SUPPORT_FRACTION_THRESHOLD)),
        "minimum_gap": float(np.min(gaps)),
        "maximum_gap": float(np.max(gaps)),
    }
    support["pass"] = bool(
        support["all_required_hosts_matched"]
        and support["median_gap"] >= float(FROZEN_THRESHOLDS["support_median_gap_min"])
        and support["fraction_gap_ge_0_04"] >= float(FROZEN_THRESHOLDS["support_fraction_gap_ge_0_04_min"])
        and support["minimum_gap"] >= float(FROZEN_THRESHOLDS["support_minimum_gap"])
    )
    paired_metrics: dict[str, dict[str, Any]] = {}
    for index, key in enumerate(METRICS):
        sd = max(float(native_summary[key]["sample_sd"]), 1e-12)
        value = float(np.median(np.abs(high[:, index] - low[:, index])) / sd)
        paired_metrics[key] = {
            "median_absolute_standardized_difference": value,
            "pass": bool(value <= float(FROZEN_THRESHOLDS["paired_median_absolute_standardized_difference_max"])),
        }
    high_gate = distribution_gate(high, native, native_summary)
    low_gate = distribution_gate(low, native, native_summary)
    radiometry = {
        "high": high_gate,
        "low": low_gate,
        "paired_standardized_difference": {
            "metrics": paired_metrics,
            "pass": bool(all(row["pass"] for row in paired_metrics.values())),
        },
    }
    radiometry["pass"] = bool(
        high_gate["pass"] and low_gate["pass"] and radiometry["paired_standardized_difference"]["pass"]
    )
    return {
        "support_gate": support,
        "radiometry_gate": radiometry,
        "joint_pass": bool(support["pass"] and radiometry["pass"]),
    }


def run_s3_medium150(
    options: list[dict[str, Any]],
    candidates: dict[int, dict[str, Any]],
    native: np.ndarray,
    native_summary: dict[str, dict[str, float]],
) -> dict[str, Any]:
    selected, solver = solve_s3_medium150(options, native_summary)
    output: dict[str, Any] = {
        "formulation": "S3_direct_wasserstein_family_balance",
        "mechanical_cohort_instantiation": {
            "host_count": EXPECTED_HOSTS,
            "fraction_gap_ge_0_040_required_count": FRACTION_REQUIRED,
            "strict_majority_gap_ge_0_060_required_count": MEDIAN_SUFFICIENT_REQUIRED,
            "native_iqr_median_envelope_required_count": NATIVE_IQR_MEDIAN_REQUIRED,
            "scientific_thresholds_changed": False,
            "objective_changed": False,
        },
        "solver": asdict(solver),
        "selected_pairs": selected,
        "infeasible": not solver.success,
        "joint_pass": False,
    }
    if not solver.success:
        return output
    evaluation = evaluate_medium150(selected, candidates, native, native_summary)
    balance = appearance_balance_medium150(selected)
    high_weber = evaluation["radiometry_gate"]["high"]["metrics"]["absolute_weber_contrast"]["standardized_wasserstein"]
    low_weber = evaluation["radiometry_gate"]["low"]["metrics"]["absolute_weber_contrast"]["standardized_wasserstein"]
    direct_weber = bool(high_weber <= WEBER_WASSERSTEIN_MAX and low_weber <= WEBER_WASSERSTEIN_MAX)
    output.update(
        {
            "evaluation": evaluation,
            "appearance_balance": balance,
            "exact_direct_weber_acceptance": direct_weber,
            "joint_pass": bool(evaluation["joint_pass"] and balance["pass"] and direct_weber),
        }
    )
    return output
