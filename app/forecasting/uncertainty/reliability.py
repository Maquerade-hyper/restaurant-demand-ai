from __future__ import annotations

import numpy as np
import pandas as pd


def interval_coverage(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
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

    covered = (
        (actual >= lower)
        & (actual <= upper)
    )

    return float(
        covered.mean()
    )


def calculate_interval_reliability(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> dict:

    coverage = interval_coverage(
        actual,
        lower,
        upper,
    )

    widths = (
        np.asarray(upper)
        - np.asarray(lower)
    )

    return {
        "coverage": coverage,
        "mean_interval_width": float(
            np.mean(widths)
        ),
        "median_interval_width": float(
            np.median(widths)
        ),
    }