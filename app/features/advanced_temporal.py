from __future__ import annotations

import numpy as np
import pandas as pd


def add_advanced_temporal_features(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Part 22A - Advanced temporal interactions.

    Forecast-time safe.

    Uses only calendar information available for the
    forecast date. No target/sales values are used.
    """

    out = df.copy()

    if date_column not in out.columns:
        raise ValueError(
            f"Missing required column: {date_column}"
        )

    date = pd.to_datetime(out[date_column])

    dow = date.dt.dayofweek.astype(float)
    month = date.dt.month.astype(float)
    day_of_year = date.dt.dayofyear.astype(float)
    week = date.dt.isocalendar().week.astype(float)
    quarter = date.dt.quarter.astype(float)

    # ============================================================
    # HIGHER-ORDER CALENDAR CYCLES
    # ============================================================

    out["advanced_dow_sin_3"] = np.sin(
        2 * np.pi * dow * 3 / 7
    )

    out["advanced_dow_cos_3"] = np.cos(
        2 * np.pi * dow * 3 / 7
    )

    out["advanced_month_sin_2"] = np.sin(
        2 * np.pi * month * 2 / 12
    )

    out["advanced_month_cos_2"] = np.cos(
        2 * np.pi * month * 2 / 12
    )

    out["advanced_week_sin"] = np.sin(
        2 * np.pi * week / 52
    )

    out["advanced_week_cos"] = np.cos(
        2 * np.pi * week / 52
    )

    out["advanced_day_year_sin_2"] = np.sin(
        2 * np.pi * day_of_year * 2 / 365
    )

    out["advanced_day_year_cos_2"] = np.cos(
        2 * np.pi * day_of_year * 2 / 365
    )

    # ============================================================
    # CALENDAR INTERACTIONS
    # ============================================================

    is_weekend = (
        (dow >= 5).astype(float)
    )

    out["advanced_weekend_month"] = (
        is_weekend * month
    )

    out["advanced_weekend_quarter"] = (
        is_weekend * quarter
    )

    out["advanced_weekend_year_position"] = (
        is_weekend *
        (day_of_year / 365.0)
    )

    # ============================================================
    # MONTH / QUARTER BOUNDARIES
    # ============================================================

    out["advanced_month_start"] = (
        date.dt.is_month_start.astype(int)
    )

    out["advanced_month_end"] = (
        date.dt.is_month_end.astype(int)
    )

    out["advanced_quarter_start"] = (
        date.dt.is_quarter_start.astype(int)
    )

    out["advanced_quarter_end"] = (
        date.dt.is_quarter_end.astype(int)
    )

    # ============================================================
    # SAFE CONTEXT INTERACTIONS
    # ============================================================

    for column in [
        "holiday_active",
        "promotion_active",
        "event_active",
        "is_rainy",
    ]:
        if column not in out.columns:
            out[column] = 0.0

    out["advanced_holiday_weekend"] = (
        out["holiday_active"].astype(float)
        * is_weekend
    )

    out["advanced_promotion_weekend"] = (
        out["promotion_active"].astype(float)
        * is_weekend
    )

    out["advanced_event_weekend"] = (
        out["event_active"].astype(float)
        * is_weekend
    )

    out["advanced_holiday_month"] = (
        out["holiday_active"].astype(float)
        * month
    )

    out["advanced_promotion_month"] = (
        out["promotion_active"].astype(float)
        * month
    )

    return out