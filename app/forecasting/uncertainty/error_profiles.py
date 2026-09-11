from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_error_profiles(
    actual: pd.Series,
    prediction: pd.Series,
) -> dict:

    actual = pd.Series(
        actual,
        dtype="float64",
    ).reset_index(drop=True)

    prediction = pd.Series(
        prediction,
        dtype="float64",
    ).reset_index(drop=True)

    if len(actual) != len(prediction):
        raise ValueError(
            "Actual and prediction must have "
            "the same length."
        )

    valid = (
        np.isfinite(actual.to_numpy())
        & np.isfinite(prediction.to_numpy())
    )

    actual = actual[valid]
    prediction = prediction[valid]

    if len(actual) == 0:
        raise ValueError(
            "No valid actual/prediction pairs available."
        )

    errors = (
        actual.to_numpy()
        - prediction.to_numpy()
    )

    absolute_errors = np.abs(errors)

    return {
        "count": int(len(errors)),
        "mean_error": float(
            np.mean(errors)
        ),
        "std_error": float(
            np.std(errors, ddof=1)
        ) if len(errors) > 1 else 0.0,
        "mae": float(
            np.mean(absolute_errors)
        ),
        "p50_absolute_error": float(
            np.quantile(
                absolute_errors,
                0.50,
            )
        ),
        "p75_absolute_error": float(
            np.quantile(
                absolute_errors,
                0.75,
            )
        ),
        "p90_absolute_error": float(
            np.quantile(
                absolute_errors,
                0.90,
            )
        ),
        "p95_absolute_error": float(
            np.quantile(
                absolute_errors,
                0.95,
            )
        ),
        "p99_absolute_error": float(
            np.quantile(
                absolute_errors,
                0.99,
            )
        ),
    }