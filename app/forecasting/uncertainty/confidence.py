from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_confidence(
    prediction,
    lower_bound,
    upper_bound,
) -> np.ndarray:

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    lower_bound = np.asarray(
        lower_bound,
        dtype=float,
    )

    upper_bound = np.asarray(
        upper_bound,
        dtype=float,
    )

    interval_width = (
        upper_bound
        - lower_bound
    )

    relative_uncertainty = (
        interval_width
        / np.maximum(
            np.abs(prediction),
            1.0,
        )
    )

    confidence = (
        1.0
        / (
            1.0
            + relative_uncertainty
        )
    )

    return np.clip(
        confidence,
        0.0,
        1.0,
    )


def confidence_label(
    confidence: float,
) -> str:

    if confidence >= 0.80:
        return "high"

    if confidence >= 0.60:
        return "medium"

    return "low"