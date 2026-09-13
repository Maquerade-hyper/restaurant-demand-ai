from __future__ import annotations

import numpy as np


DEFAULT_QUANTILES = (
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
)


def _validate_quantiles(
    quantiles: tuple[float, ...] | list[float],
) -> np.ndarray:

    values = np.asarray(
        quantiles,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "quantiles must be one-dimensional"
        )

    if len(values) == 0:
        raise ValueError(
            "quantiles cannot be empty"
        )

    if np.any(
        (values <= 0.0)
        |
        (values >= 1.0)
    ):
        raise ValueError(
            "quantiles must be strictly between 0 and 1"
        )

    if np.any(
        np.diff(values) < 0
    ):
        raise ValueError(
            "quantiles must be sorted"
        )

    return values


def build_quantile_forecast(
    point_forecast,
    residuals,
    quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
    non_negative: bool = True,
) -> dict[str, np.ndarray]:
    """
    Build probabilistic forecasts from a point forecast
    and empirical residual distribution.

    residual = actual - point_forecast

    Therefore:

        predictive_quantile
            =
        point_forecast
            +
        residual_quantile
    """

    point = np.asarray(
        point_forecast,
        dtype=float,
    )

    residual = np.asarray(
        residuals,
        dtype=float,
    )

    if point.ndim == 0:
        point = point.reshape(1)

    if residual.ndim != 1:
        residual = residual.reshape(-1)

    residual = residual[
        np.isfinite(residual)
    ]

    if len(residual) == 0:
        raise ValueError(
            "At least one finite residual is required"
        )

    qs = _validate_quantiles(
        quantiles
    )

    residual_quantiles = np.quantile(
        residual,
        qs,
    )

    forecasts = {}

    for q, residual_q in zip(
        qs,
        residual_quantiles,
    ):

        values = (
            point
            +
            residual_q
        )

        if non_negative:
            values = np.maximum(
                values,
                0.0,
            )

        forecasts[
            f"p{int(round(q * 100)):02d}"
        ] = values

    # Ensure the median is represented by the point forecast
    # when P50 was requested.
    if 0.50 in qs:

        forecasts["p50"] = np.maximum(
            point,
            0.0,
        ) if non_negative else point.copy()

    return forecasts