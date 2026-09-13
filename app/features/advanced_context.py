from __future__ import annotations

import numpy as np
import pandas as pd


def add_advanced_context_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Part 22D - Advanced context interactions.

    Uses only context variables available at forecast time.
    """

    out = df.copy()

    # ------------------------------------------------------------
    # Ensure optional context columns exist
    # ------------------------------------------------------------

    defaults = {
        "temperature": 0.0,
        "temperature_squared": 0.0,
        "is_rainy": 0.0,
        "holiday_active": 0.0,
        "promotion_active": 0.0,
        "event_active": 0.0,
        "tourism_student_interaction": 0.0,
        "business_density_interaction": 0.0,
    }

    for column, default in defaults.items():

        if column not in out.columns:
            out[column] = default

    # ------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------

    out["advanced_temperature_rain"] = (
        out["temperature"].astype(float)
        *
        out["is_rainy"].astype(float)
    )

    out["advanced_temperature_squared_rain"] = (
        out["temperature_squared"].astype(float)
        *
        out["is_rainy"].astype(float)
    )

    # ------------------------------------------------------------
    # Calendar × weather
    # ------------------------------------------------------------

    out["advanced_holiday_rain"] = (
        out["holiday_active"].astype(float)
        *
        out["is_rainy"].astype(float)
    )

    out["advanced_promotion_rain"] = (
        out["promotion_active"].astype(float)
        *
        out["is_rainy"].astype(float)
    )

    out["advanced_event_rain"] = (
        out["event_active"].astype(float)
        *
        out["is_rainy"].astype(float)
    )

    # ------------------------------------------------------------
    # Calendar × operational context
    # ------------------------------------------------------------

    out["advanced_holiday_tourism"] = (
        out["holiday_active"].astype(float)
        *
        out["tourism_student_interaction"].astype(float)
    )

    out["advanced_event_tourism"] = (
        out["event_active"].astype(float)
        *
        out["tourism_student_interaction"].astype(float)
    )

    out["advanced_promotion_business"] = (
        out["promotion_active"].astype(float)
        *
        out["business_density_interaction"].astype(float)
    )

    out["advanced_event_business"] = (
        out["event_active"].astype(float)
        *
        out["business_density_interaction"].astype(float)
    )

    # ------------------------------------------------------------
    # Context magnitude
    # ------------------------------------------------------------

    context_columns = [
        "holiday_active",
        "promotion_active",
        "event_active",
        "is_rainy",
    ]

    out["advanced_context_intensity"] = (
        out[context_columns]
        .astype(float)
        .sum(axis=1)
    )

    # ------------------------------------------------------------
    # Nonlinear context
    # ------------------------------------------------------------

    out["advanced_temperature_abs"] = (
        np.abs(
            out["temperature"].astype(float)
        )
    )

    out["advanced_temperature_rain_abs"] = (
        out["advanced_temperature_abs"]
        *
        out["is_rainy"].astype(float)
    )

    return out