from __future__ import annotations

import numpy as np
import pandas as pd


def interval_coverage(
    actual,
    lower,
    upper,
) -> float:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(lower)
        &
        np.isfinite(upper)
    )

    if not mask.any():
        return float("nan")

    return float(
        np.mean(
            (
                actual[mask]
                >=
                lower[mask]
            )
            &
            (
                actual[mask]
                <=
                upper[mask]
            )
        )
    )


def mean_interval_width(
    lower,
    upper,
) -> float:

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    mask = (
        np.isfinite(lower)
        &
        np.isfinite(upper)
    )

    if not mask.any():
        return float("nan")

    return float(
        np.mean(
            upper[mask]
            -
            lower[mask]
        )
    )


def pinball_loss(
    actual,
    forecast,
    quantile: float,
) -> float:
    """
    Quantile / pinball loss.
    """

    if not (
        0.0
        <
        quantile
        <
        1.0
    ):
        raise ValueError(
            "quantile must be between 0 and 1"
        )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(forecast)
    )

    if not mask.any():
        return float("nan")

    error = (
        actual[mask]
        -
        forecast[mask]
    )

    return float(
        np.mean(
            np.maximum(
                quantile * error,
                (quantile - 1.0) * error,
            )
        )
    )


def evaluate_calibration(
    actual,
    predictions: dict[str, np.ndarray],
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """
    Evaluate quantile and interval calibration.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    ).reshape(-1)

    result = {
        "observations": int(
            len(actual)
        ),
        "quantiles": {},
        "intervals": {},
    }

    # ------------------------------------------------------------
    # Quantiles
    # ------------------------------------------------------------

    for name, forecast in predictions.items():

        if not name.startswith("p"):
            continue

        try:
            q = int(
                name[1:]
            ) / 100.0
        except ValueError:
            continue

        forecast = np.asarray(
            forecast,
            dtype=float,
        ).reshape(-1)

        if len(forecast) != len(actual):
            raise ValueError(
                f"{name} forecast length mismatch"
            )

        loss = pinball_loss(
            actual,
            forecast,
            q,
        )

        empirical_below = float(
            np.mean(
                actual
                <
                forecast
            )
        )

        result[
            "quantiles"
        ][name] = {
            "quantile": q,
            "pinball_loss": loss,
            "empirical_below_rate": empirical_below,
        }

    # ------------------------------------------------------------
    # Intervals
    # ------------------------------------------------------------

    if intervals:

        for name, pair in intervals.items():

            lower, upper = pair

            coverage = interval_coverage(
                actual,
                lower,
                upper,
            )

            width = mean_interval_width(
                lower,
                upper,
            )

            result[
                "intervals"
            ][name] = {
                "coverage": coverage,
                "mean_width": width,
            }

    return result