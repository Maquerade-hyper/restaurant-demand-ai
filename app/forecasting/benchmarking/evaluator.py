from __future__ import annotations

import time

import numpy as np
import pandas as pd


def mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def smape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    values = np.where(
        denominator == 0,
        0.0,
        2.0
        * np.abs(actual - predicted)
        / denominator,
    )

    return float(
        np.mean(values) * 100.0
    )


def bias(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    return float(
        np.mean(
            predicted - actual
        )
    )


def high_demand_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    quantile: float = 0.90,
) -> tuple[float, float]:

    threshold = float(
        np.quantile(
            actual,
            quantile,
        )
    )

    mask = actual >= threshold

    if not mask.any():
        return 0.0, 0.0

    return (
        mae(
            actual[mask],
            predicted[mask],
        ),
        bias(
            actual[mask],
            predicted[mask],
        ),
    )


def spike_recall(
    actual: np.ndarray,
    predicted: np.ndarray,
    quantile: float = 0.90,
) -> float:

    actual_threshold = float(
        np.quantile(
            actual,
            quantile,
        )
    )

    predicted_threshold = float(
        np.quantile(
            predicted,
            quantile,
        )
    )

    actual_spikes = actual >= actual_threshold

    if not actual_spikes.any():
        return 0.0

    detected = (
        predicted[actual_spikes]
        >= predicted_threshold
    )

    return float(
        np.mean(detected)
    )


def calculate_stability(
    predicted: np.ndarray,
) -> float:

    if len(predicted) < 2:
        return 1.0

    differences = np.diff(
        predicted
    )

    volatility = float(
        np.std(differences)
    )

    scale = float(
        np.mean(
            np.abs(predicted)
        )
    )

    if scale <= 1e-12:
        return 1.0

    normalized = (
        volatility / scale
    )

    # Higher is better.
    return float(
        1.0 / (1.0 + normalized)
    )


def evaluate_predictions(
    model_name: str,
    actual: pd.Series,
    predicted: pd.Series,
    prediction_cost: float = 0.0,
) -> dict:

    actual_values = (
        pd.to_numeric(
            actual,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    predicted_values = (
        pd.to_numeric(
            predicted,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    if len(actual_values) != len(
        predicted_values
    ):
        raise ValueError(
            "actual and predicted lengths differ"
        )

    if len(actual_values) == 0:
        raise ValueError(
            "cannot evaluate empty predictions"
        )

    if not np.isfinite(
        actual_values
    ).all():

        raise ValueError(
            "actual contains non-finite values"
        )

    if not np.isfinite(
        predicted_values
    ).all():

        raise ValueError(
            "predicted contains non-finite values"
        )

    high_mae, high_bias = (
        high_demand_metrics(
            actual_values,
            predicted_values,
        )
    )

    return {
        "model_name": model_name,
        "mae": mae(
            actual_values,
            predicted_values,
        ),
        "rmse": rmse(
            actual_values,
            predicted_values,
        ),
        "smape": smape(
            actual_values,
            predicted_values,
        ),
        "bias": bias(
            actual_values,
            predicted_values,
        ),
        "high_demand_mae": high_mae,
        "high_demand_bias": high_bias,
        "spike_recall": spike_recall(
            actual_values,
            predicted_values,
        ),
        "stability": calculate_stability(
            predicted_values
        ),
        "prediction_cost": float(
            prediction_cost
        ),
    }


def timed_prediction(
    predictor,
    data,
) -> tuple[np.ndarray, float]:

    start = time.perf_counter()

    prediction = predictor(
        data
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.asarray(
            prediction,
            dtype=float,
        ),
        float(elapsed),
    )