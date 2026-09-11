from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.metrics import (
    mae,
    rmse,
    smape,
    bias,
)


def calculate_errors(
    actual,
    predicted,
) -> pd.DataFrame:

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "error": predicted - actual,
            "absolute_error": np.abs(
                predicted - actual
            ),
            "squared_error": (
                predicted - actual
            ) ** 2,
        }
    )


def diagnostic_metrics(
    actual,
    predicted,
) -> dict:

    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
        "bias": bias(actual, predicted),
    }


def bias_analysis(
    actual,
    predicted,
) -> dict:

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    overforecast = np.sum(
        errors > 0
    )

    underforecast = np.sum(
        errors < 0
    )

    total = len(errors)

    return {
        "mean_bias": float(
            np.mean(errors)
        ),
        "overforecast_rate": float(
            overforecast / total
        ),
        "underforecast_rate": float(
            underforecast / total
        ),
        "max_overforecast": float(
            np.max(errors)
        ),
        "max_underforecast": float(
            np.min(errors)
        ),
    }



def high_demand_analysis(
    actual,
    predicted,
    percentile: float = 0.90,
) -> dict:

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    threshold = np.quantile(
        actual,
        percentile,
    )

    mask = actual >= threshold

    if not np.any(mask):
        return {
            "threshold": float(threshold),
            "count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "smape": 0.0,
            "bias": 0.0,
            "spike_recall": 0.0,
        }

    high_actual = actual[mask]
    high_predicted = predicted[mask]

    return {
        "threshold": float(threshold),
        "count": int(mask.sum()),
        "mae": mae(
            high_actual,
            high_predicted,
        ),
        "rmse": rmse(
            high_actual,
            high_predicted,
        ),
        "smape": smape(
            high_actual,
            high_predicted,
        ),
        "bias": bias(
            high_actual,
            high_predicted,
        ),
    }


def spike_recall(
    actual,
    predicted,
    percentile: float = 0.90,
) -> float:

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    actual_threshold = np.quantile(
        actual,
        percentile,
    )

    predicted_threshold = np.quantile(
        predicted,
        percentile,
    )

    actual_spikes = (
        actual >= actual_threshold
    )

    detected_spikes = (
        predicted >= predicted_threshold
    )

    if not np.any(actual_spikes):
        return 0.0

    true_positive = np.sum(
        actual_spikes & detected_spikes
    )

    return float(
        true_positive
        / np.sum(actual_spikes)
    )


def group_diagnostics(
    df: pd.DataFrame,
    actual_column: str = "quantity_sold",
    predicted_column: str = "prediction",
    group_column: str = "outlet_id",
) -> pd.DataFrame:

    rows = []

    for group_value, group in df.groupby(
        group_column
    ):

        actual = group[
            actual_column
        ].to_numpy()

        predicted = group[
            predicted_column
        ].to_numpy()

        metrics = diagnostic_metrics(
            actual,
            predicted,
        )

        rows.append(
            {
                group_column: group_value,
                "count": len(group),
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mae", ascending=False)
        .reset_index(drop=True)
    )