from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.mean(
            np.abs(actual - predicted)
        )
    )


def rmse(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def bias(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.mean(predicted - actual)
    )


def recovery_ratio(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = float(
        np.sum(np.abs(actual))
    )

    if denominator <= 1e-12:
        return 1.0

    return float(
        1.0
        - np.sum(np.abs(actual - predicted))
        / denominator
    )


def constraint_accuracy(
    actual_constrained,
    estimated_constrained,
) -> float:
    actual = np.asarray(
        actual_constrained,
        dtype=bool,
    )

    estimated = np.asarray(
        estimated_constrained,
        dtype=bool,
    )

    if len(actual) == 0:
        return 1.0

    return float(
        np.mean(actual == estimated)
    )


def fulfillment_mae(
    observed,
    demand,
    estimated_fulfillment,
) -> float:
    observed = np.asarray(
        observed,
        dtype=float,
    )

    demand = np.asarray(
        demand,
        dtype=float,
    )

    estimated_fulfillment = np.asarray(
        estimated_fulfillment,
        dtype=float,
    )

    actual_fulfillment = np.divide(
        observed,
        np.maximum(demand, 1e-9),
    )

    actual_fulfillment = np.clip(
        actual_fulfillment,
        0.0,
        1.0,
    )

    return mae(
        actual_fulfillment,
        estimated_fulfillment,
    )


def validate_truth_relationships(
    truth: pd.DataFrame,
) -> dict:
    required = {
        "outlet_id",
        "product_id",
        "date",
        "true_demand",
        "observed_sales",
        "lost_demand",
        "stockout",
    }

    missing = required - set(truth.columns)

    if missing:
        raise ValueError(
            "Truth data missing columns: "
            + ", ".join(sorted(missing))
        )

    true_demand = pd.to_numeric(
        truth["true_demand"],
        errors="coerce",
    )

    observed_sales = pd.to_numeric(
        truth["observed_sales"],
        errors="coerce",
    )

    lost_demand = pd.to_numeric(
        truth["lost_demand"],
        errors="coerce",
    )

    stockout = (
        truth["stockout"]
        .fillna(False)
        .astype(bool)
    )

    return {
        "rows": int(len(truth)),
        "true_demand_ge_observed": bool(
            (true_demand >= observed_sales - 1e-9).all()
        ),
        "lost_demand_nonnegative": bool(
            (lost_demand >= -1e-9).all()
        ),
        "stockout_has_lost_demand": bool(
            (
                (~stockout)
                | (lost_demand > -1e-9)
            ).all()
        ),
        "max_accounting_error": float(
            np.max(
                np.abs(
                    true_demand
                    - observed_sales
                    - lost_demand
                )
            )
        ),
    }


def validate_output_integrity(
    result: pd.DataFrame,
) -> dict:
    required = {
        "outlet_id",
        "product_id",
        "date",
        "quantity_sold",
        "deconstrained_demand",
        "estimated_lost_demand",
        "stockout",
        "censoring_strength",
        "censoring_class",
    }

    missing = required - set(result.columns)

    if missing:
        raise ValueError(
            "Demand intelligence output missing columns: "
            + ", ".join(sorted(missing))
        )

    demand = pd.to_numeric(
        result["deconstrained_demand"],
        errors="coerce",
    )

    observed = pd.to_numeric(
        result["quantity_sold"],
        errors="coerce",
    )

    lost = pd.to_numeric(
        result["estimated_lost_demand"],
        errors="coerce",
    )

    strength = pd.to_numeric(
        result["censoring_strength"],
        errors="coerce",
    )

    stockout = (
        result["stockout"]
        .fillna(False)
        .astype(bool)
    )

    return {
        "rows": int(len(result)),
        "null_demand": int(demand.isna().sum()),
        "null_lost_demand": int(lost.isna().sum()),
        "demand_below_observed": int(
            (
                demand
                < observed - 1e-9
            ).sum()
        ),
        "negative_lost_demand": int(
            (lost < -1e-9).sum()
        ),
        "invalid_censoring_strength": int(
            (
                (strength < -1e-9)
                | (strength > 1.0 + 1e-9)
            ).sum()
        ),
        "stockout_records": int(
            stockout.sum()
        ),
        "censored_records": int(
            result["censored_demand_flag"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if "censored_demand_flag" in result.columns
        else 0,
    }