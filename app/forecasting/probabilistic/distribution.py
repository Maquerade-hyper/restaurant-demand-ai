from __future__ import annotations

import numpy as np
import pandas as pd


def build_predictive_distribution(
    point_forecast,
    residuals,
    quantiles: tuple[float, ...] = (
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
    ),
    non_negative: bool = True,
) -> pd.DataFrame:
    """
    Build a predictive distribution table.

    Each forecast row contains empirical predictive quantiles
    generated from the calibrated residual distribution.
    """

    point = np.asarray(
        point_forecast,
        dtype=float,
    ).reshape(-1)

    residual = np.asarray(
        residuals,
        dtype=float,
    ).reshape(-1)

    residual = residual[
        np.isfinite(residual)
    ]

    if len(residual) == 0:
        raise ValueError(
            "No valid residuals available"
        )

    qs = np.asarray(
        quantiles,
        dtype=float,
    )

    if np.any(
        (qs <= 0)
        |
        (qs >= 1)
    ):
        raise ValueError(
            "Quantiles must be between 0 and 1"
        )

    if np.any(
        np.diff(qs) < 0
    ):
        raise ValueError(
            "Quantiles must be sorted"
        )

    residual_quantiles = np.quantile(
        residual,
        qs,
    )

    result = pd.DataFrame(
        {
            "point_forecast": point,
        }
    )

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

        result[
            f"p{int(round(q * 100)):02d}"
        ] = values

    # ------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------

    result["predictive_mean"] = (
        point
        +
        np.mean(residual)
    )

    result["predictive_std"] = float(
        np.std(
            residual,
            ddof=1,
        )
    ) if len(residual) > 1 else 0.0

    result["predictive_lower_90"] = (
        result["p05"]
    )

    result["predictive_upper_90"] = (
        result["p95"]
    )

    result["predictive_width_90"] = (
        result["predictive_upper_90"]
        -
        result["predictive_lower_90"]
    )

    result["predictive_lower_80"] = (
        result["p10"]
    )

    result["predictive_upper_80"] = (
        result["p90"]
    )

    result["predictive_width_80"] = (
        result["predictive_upper_80"]
        -
        result["predictive_lower_80"]
    )

    # ------------------------------------------------------------
    # Probability of exceeding point forecast
    # ------------------------------------------------------------

    result["residual_positive_probability"] = float(
        np.mean(
            residual > 0
        )
    )

    result["residual_negative_probability"] = float(
        np.mean(
            residual < 0
        )
    )

    return result