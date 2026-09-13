
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROFILE_REQUIRED = {
    "outlet_id",
}

BEHAVIOR_REQUIRED = {
    "outlet_id",
    "behavior_days",
    "behavior_average_daily_demand",
    "behavior_coefficient_variation",
    "behavior_peak_intensity",
}

PERFORMANCE_REQUIRED = {
    "outlet_id",
    "performance_days",
    "total_observed_demand",
    "total_deconstrained_demand",
    "total_estimated_lost_demand",
    "fulfillment_rate",
    "lost_demand_rate",
    "stockout_product_day_rate",
    "performance_score",
    "opportunity_index",
    "risk_index",
    "performance_rank",
    "opportunity_rank",
    "risk_rank",
}

SEGMENT_REQUIRED = {
    "outlet_id",
    "cluster_id",
    "cluster_size",
    "segment_name",
}


def _check_required(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> list[str]:
    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        return [
            f"{name}: missing columns {missing}"
        ]

    return []


def _check_unique_outlets(
    df: pd.DataFrame,
    name: str,
) -> list[str]:
    errors = []

    if "outlet_id" not in df.columns:
        return errors

    duplicated = df["outlet_id"].duplicated()

    if duplicated.any():
        count = int(duplicated.sum())

        errors.append(
            f"{name}: {count} duplicate outlet rows"
        )

    return errors


def _check_finite(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> list[str]:
    errors = []

    existing = [
        column
        for column in columns
        if column in df.columns
    ]

    for column in existing:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if not np.isfinite(
            values.to_numpy()
        ).all():
            errors.append(
                f"{name}: non-finite values in {column}"
            )

    return errors


def _check_bounded(
    df: pd.DataFrame,
    columns: list[str],
    lower: float,
    upper: float,
    name: str,
) -> list[str]:
    errors = []

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = (
            values < lower
        ) | (
            values > upper
        )

        if invalid.any():
            errors.append(
                f"{name}: {column} outside "
                f"[{lower}, {upper}]"
            )

    return errors


def validate_profile(
    profiles: pd.DataFrame,
) -> dict[str, Any]:
    errors = []

    errors.extend(
        _check_required(
            profiles,
            PROFILE_REQUIRED,
            "profiles",
        )
    )

    errors.extend(
        _check_unique_outlets(
            profiles,
            "profiles",
        )
    )

    return {
        "passed": not errors,
        "errors": errors,
        "outlet_count": int(
            profiles["outlet_id"].nunique()
        )
        if "outlet_id" in profiles.columns
        else 0,
    }


def validate_behavior(
    behavior: pd.DataFrame,
) -> dict[str, Any]:
    errors = []

    errors.extend(
        _check_required(
            behavior,
            BEHAVIOR_REQUIRED,
            "behavior",
        )
    )

    errors.extend(
        _check_unique_outlets(
            behavior,
            "behavior",
        )
    )

    errors.extend(
        _check_finite(
            behavior,
            [
                "behavior_average_daily_demand",
                "behavior_coefficient_variation",
                "behavior_peak_intensity",
            ],
            "behavior",
        )
    )

    errors.extend(
        _check_bounded(
            behavior,
            [
                "behavior_average_daily_demand",
                "behavior_coefficient_variation",
                "behavior_peak_intensity",
            ],
            0.0,
            float("inf"),
            "behavior",
        )
    )

    return {
        "passed": not errors,
        "errors": errors,
        "outlet_count": int(
            behavior["outlet_id"].nunique()
        )
        if "outlet_id" in behavior.columns
        else 0,
    }


def validate_performance(
    performance: pd.DataFrame,
) -> dict[str, Any]:
    errors = []

    errors.extend(
        _check_required(
            performance,
            PERFORMANCE_REQUIRED,
            "performance",
        )
    )

    errors.extend(
        _check_unique_outlets(
            performance,
            "performance",
        )
    )

    errors.extend(
        _check_finite(
            performance,
            [
                "total_observed_demand",
                "total_deconstrained_demand",
                "total_estimated_lost_demand",
                "fulfillment_rate",
                "lost_demand_rate",
                "stockout_product_day_rate",
                "performance_score",
                "opportunity_index",
                "risk_index",
            ],
            "performance",
        )
    )

    errors.extend(
        _check_bounded(
            performance,
            [
                "fulfillment_rate",
                "lost_demand_rate",
                "stockout_product_day_rate",
            ],
            0.0,
            1.0,
            "performance",
        )
    )

    errors.extend(
        _check_bounded(
            performance,
            [
                "performance_score",
                "opportunity_index",
                "risk_index",
            ],
            0.0,
            100.0,
            "performance",
        )
    )

    if not performance.empty:
        observed = pd.to_numeric(
            performance["total_observed_demand"],
            errors="coerce",
        )

        demand = pd.to_numeric(
            performance["total_deconstrained_demand"],
            errors="coerce",
        )

        lost = pd.to_numeric(
            performance["total_estimated_lost_demand"],
            errors="coerce",
        )

        if (observed > demand + 1e-8).any():
            errors.append(
                "performance: observed demand "
                "exceeds deconstrained demand"
            )

        if (lost < -1e-8).any():
            errors.append(
                "performance: negative lost demand"
            )

        expected_fulfillment = np.divide(
            observed,
            demand,
            out=np.zeros_like(
                observed.to_numpy(
                    dtype=float
                )
            ),
            where=demand.to_numpy(
                dtype=float
            ) > 0,
        )

        actual_fulfillment = (
            pd.to_numeric(
                performance[
                    "fulfillment_rate"
                ],
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        if not np.allclose(
            expected_fulfillment,
            actual_fulfillment,
            atol=1e-6,
        ):
            errors.append(
                "performance: fulfillment rate "
                "does not reconcile"
            )

    return {
        "passed": not errors,
        "errors": errors,
        "outlet_count": int(
            performance["outlet_id"].nunique()
        )
        if "outlet_id" in performance.columns
        else 0,
    }


def validate_segments(
    segments: pd.DataFrame,
) -> dict[str, Any]:
    errors = []

    errors.extend(
        _check_required(
            segments,
            SEGMENT_REQUIRED,
            "segments",
        )
    )

    errors.extend(
        _check_unique_outlets(
            segments,
            "segments",
        )
    )

    if "cluster_id" in segments.columns:
        if segments["cluster_id"].isna().any():
            errors.append(
                "segments: missing cluster assignments"
            )

    if "segment_name" in segments.columns:
        if segments["segment_name"].isna().any():
            errors.append(
                "segments: missing segment names"
            )

    if {
        "cluster_id",
        "cluster_size",
    }.issubset(segments.columns):
        actual_sizes = (
            segments["cluster_id"]
            .value_counts()
        )

        declared_sizes = (
            segments
            .groupby("cluster_id")[
                "cluster_size"
            ]
            .first()
        )

        for cluster_id, actual_size in (
            actual_sizes.items()
        ):
            declared = int(
                declared_sizes.loc[
                    cluster_id
                ]
            )

            if declared != int(actual_size):
                errors.append(
                    f"segments: cluster "
                    f"{cluster_id} size mismatch"
                )

    return {
        "passed": not errors,
        "errors": errors,
        "outlet_count": int(
            segments["outlet_id"].nunique()
        )
        if "outlet_id" in segments.columns
        else 0,
        "cluster_count": int(
            segments["cluster_id"].nunique()
        )
        if "cluster_id" in segments.columns
        else 0,
    }


def validate_cross_layer(
    profiles: pd.DataFrame,
    behavior: pd.DataFrame,
    performance: pd.DataFrame,
    segments: pd.DataFrame,
) -> dict[str, Any]:
    errors = []

    datasets = {
        "profiles": profiles,
        "behavior": behavior,
        "performance": performance,
        "segments": segments,
    }

    outlet_sets = {}

    for name, df in datasets.items():
        if "outlet_id" not in df.columns:
            errors.append(
                f"{name}: outlet_id missing"
            )
            continue

        outlet_sets[name] = set(
            df["outlet_id"]
        )

    if outlet_sets:
        reference_name = next(
            iter(outlet_sets)
        )

        reference = outlet_sets[
            reference_name
        ]

        for name, values in outlet_sets.items():
            if values != reference:
                missing = sorted(
                    reference - values
                )

                extra = sorted(
                    values - reference
                )

                errors.append(
                    f"cross-layer: {name} "
                    f"outlet mismatch; "
                    f"missing={missing[:10]}, "
                    f"extra={extra[:10]}"
                )

    return {
        "passed": not errors,
        "errors": errors,
        "outlet_count": len(
            next(
                iter(outlet_sets.values())
            )
        )
        if outlet_sets
        else 0,
    }


def validate_outlet_intelligence(
    profiles: pd.DataFrame,
    behavior: pd.DataFrame,
    performance: pd.DataFrame,
    segments: pd.DataFrame,
) -> dict[str, Any]:
    profile_result = validate_profile(
        profiles
    )

    behavior_result = validate_behavior(
        behavior
    )

    performance_result = validate_performance(
        performance
    )

    segment_result = validate_segments(
        segments
    )

    cross_layer_result = (
        validate_cross_layer(
            profiles,
            behavior,
            performance,
            segments,
        )
    )

    all_results = {
        "profiles": profile_result,
        "behavior": behavior_result,
        "performance": performance_result,
        "segments": segment_result,
        "cross_layer": cross_layer_result,
    }

    errors = []

    for result in all_results.values():
        errors.extend(
            result["errors"]
        )

    return {
        "passed": not errors,
        "errors": errors,
        "results": all_results,
        "outlet_count": cross_layer_result[
            "outlet_count"
        ],
    }

