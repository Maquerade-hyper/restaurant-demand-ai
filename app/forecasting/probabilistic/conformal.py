from __future__ import annotations

import numpy as np


def _finite_array(values) -> np.ndarray:

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    return array[
        np.isfinite(array)
    ]


def conformal_quantile(
    calibration_actual,
    calibration_prediction,
    coverage: float = 0.90,
) -> float:
    """
    Calculate split-conformal absolute residual radius.

    Nonconformity score:

        |actual - prediction|

    The returned radius produces an approximate marginal
    prediction interval with the requested coverage under
    the usual exchangeability assumptions.
    """

    if not (
        0.0
        <
        coverage
        <
        1.0
    ):
        raise ValueError(
            "coverage must be between 0 and 1"
        )

    actual = np.asarray(
        calibration_actual,
        dtype=float,
    ).reshape(-1)

    prediction = np.asarray(
        calibration_prediction,
        dtype=float,
    ).reshape(-1)

    if len(actual) != len(prediction):
        raise ValueError(
            "actual and prediction must have equal length"
        )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(prediction)
    )

    actual = actual[mask]
    prediction = prediction[mask]

    if len(actual) == 0:
        raise ValueError(
            "No valid calibration observations"
        )

    scores = np.abs(
        actual
        -
        prediction
    )

    n = len(scores)

    # Finite-sample conformal rank.
    rank = int(
        np.ceil(
            (n + 1)
            *
            coverage
        )
    )

    rank = max(
        1,
        min(
            rank,
            n,
        ),
    )

    sorted_scores = np.sort(
        scores
    )

    return float(
        sorted_scores[
            rank - 1
        ]
    )


def conformal_interval(
    point_forecast,
    calibration_actual,
    calibration_prediction,
    coverage: float = 0.90,
    non_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Construct a symmetric split-conformal interval.
    """

    point = np.asarray(
        point_forecast,
        dtype=float,
    ).reshape(-1)

    radius = conformal_quantile(
        calibration_actual,
        calibration_prediction,
        coverage=coverage,
    )

    lower = (
        point
        -
        radius
    )

    upper = (
        point
        +
        radius
    )

    if non_negative:

        lower = np.maximum(
            lower,
            0.0,
        )

        upper = np.maximum(
            upper,
            0.0,
        )

    return (
        lower,
        upper,
        radius,
    )


def asymmetric_conformal_interval(
    point_forecast,
    calibration_actual,
    calibration_prediction,
    lower_coverage: float = 0.10,
    upper_coverage: float = 0.90,
    non_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct an asymmetric empirical conformal-style interval.

    Lower and upper residual quantiles are estimated separately.
    """

    if not (
        0.0
        <
        lower_coverage
        <
        upper_coverage
        <
        1.0
    ):
        raise ValueError(
            "Require 0 < lower_coverage < upper_coverage < 1"
        )

    point = np.asarray(
        point_forecast,
        dtype=float,
    ).reshape(-1)

    actual = np.asarray(
        calibration_actual,
        dtype=float,
    ).reshape(-1)

    prediction = np.asarray(
        calibration_prediction,
        dtype=float,
    ).reshape(-1)

    if len(actual) != len(prediction):
        raise ValueError(
            "Calibration arrays must have equal length"
        )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(prediction)
    )

    residuals = (
        actual[mask]
        -
        prediction[mask]
    )

    if len(residuals) == 0:
        raise ValueError(
            "No valid calibration residuals"
        )

    lower_residual = np.quantile(
        residuals,
        lower_coverage,
    )

    upper_residual = np.quantile(
        residuals,
        upper_coverage,
    )

    lower = (
        point
        +
        lower_residual
    )

    upper = (
        point
        +
        upper_residual
    )

    if non_negative:

        lower = np.maximum(
            lower,
            0.0,
        )

        upper = np.maximum(
            upper,
            0.0,
        )

    return lower, upper