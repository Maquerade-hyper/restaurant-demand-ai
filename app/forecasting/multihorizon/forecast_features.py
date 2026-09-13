from __future__ import annotations

import pandas as pd

from app.features.time_features import create_time_features
from app.features.lag_features import create_lag_features
from app.features.rolling_features import create_rolling_features
from app.features.trend_features import create_trend_features
from app.features.volatility_features import create_volatility_features
from app.features.order_velocity import create_order_velocity_features
from app.features.seasonality_features import create_seasonality_features
from app.features.context_features import create_context_features


# ============================================================
# FORECAST-TIME FEATURE BUILDER
# ============================================================
#
# Purpose:
# Build features for ONE future forecast date using only:
#
#   1. Historical observed sales
#   2. Previous forecasts
#   3. Known future context
#
# The target value for the date being forecast is NEVER used.
#
# Example:
#
# D+1 -> actual history
# D+2 -> actual history + D+1 prediction
# D+3 -> actual history + D+1 + D+2 predictions
#
# ============================================================


def build_forecast_features(
    history: pd.DataFrame,
    future_row: pd.DataFrame,
    target_column: str = "quantity_sold",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:

    if group_columns is None:
        group_columns = [
            "outlet_id",
            "product_id",
        ]

    if history.empty:
        raise ValueError(
            "History cannot be empty."
        )

    if future_row.empty:
        raise ValueError(
            "Future row cannot be empty."
        )

    required_columns = set(
        group_columns
        + [
            "date",
            target_column,
        ]
    )

    missing_history = (
        required_columns
        - set(history.columns)
    )

    if missing_history:
        raise ValueError(
            f"History missing columns: "
            f"{sorted(missing_history)}"
        )

    missing_future = (
        set(group_columns + ["date"])
        - set(future_row.columns)
    )

    if missing_future:
        raise ValueError(
            f"Future row missing columns: "
            f"{sorted(missing_future)}"
        )

    # --------------------------------------------------------
    # Copy inputs
    # --------------------------------------------------------

    historical = history.copy()
    future = future_row.copy()

    historical["date"] = pd.to_datetime(
        historical["date"]
    )

    future["date"] = pd.to_datetime(
        future["date"]
    )

    # --------------------------------------------------------
    # Future target MUST NOT exist
    # --------------------------------------------------------

    #
    # We deliberately set the future target to NaN.
    #
    # This prevents the future day's own demand from
    # entering any lag/rolling calculation.
    #

    future[target_column] = float("nan")

    # --------------------------------------------------------
    # Keep only relevant groups
    # --------------------------------------------------------

    groups = future[
        group_columns
    ].drop_duplicates()

    historical = historical.merge(
        groups,
        on=group_columns,
        how="inner",
    )

    # --------------------------------------------------------
    # Combine historical observations + forecast date
    # --------------------------------------------------------

    combined = pd.concat(
        [
            historical,
            future,
        ],
        ignore_index=True,
        sort=False,
    )

    combined = combined.sort_values(
        group_columns + ["date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 1. TIME FEATURES
    # --------------------------------------------------------

    combined = create_time_features(
        combined,
        date_column="date",
    )

    # --------------------------------------------------------
    # 2. LAG FEATURES
    # --------------------------------------------------------

    combined = create_lag_features(
        combined,
        target_column=target_column,
        group_columns=group_columns,
    )

    # --------------------------------------------------------
    # 3. ROLLING FEATURES
    # --------------------------------------------------------

    combined = create_rolling_features(
        combined,
        target_column=target_column,
        group_columns=group_columns,
    )

    # --------------------------------------------------------
    # 4. TREND FEATURES
    # --------------------------------------------------------

    combined = create_trend_features(
        combined,
        target_column=target_column,
        group_columns=group_columns,
    )

    # --------------------------------------------------------
    # 5. VOLATILITY FEATURES
    # --------------------------------------------------------

    combined = create_volatility_features(
        combined,
        target_column=target_column,
        group_columns=group_columns,
    )

    # --------------------------------------------------------
    # 6. ORDER VELOCITY
    # --------------------------------------------------------

    combined = create_order_velocity_features(
        combined,
        target_column=target_column,
        group_columns=group_columns,
    )

    # --------------------------------------------------------
    # 7. SEASONALITY
    # --------------------------------------------------------

    combined = create_seasonality_features(
        combined
    )

    # --------------------------------------------------------
    # 8. CONTEXT
    # --------------------------------------------------------

    combined = create_context_features(
        combined
    )

    # --------------------------------------------------------
    # Return ONLY the requested future row(s)
    # --------------------------------------------------------

    future_keys = future[
        group_columns + ["date"]
    ].copy()

    result = combined.merge(
        future_keys,
        on=group_columns + ["date"],
        how="inner",
    )

    # --------------------------------------------------------
    # Safety check:
    #
    # The target for the forecast date must remain unknown.
    # --------------------------------------------------------

    if result[target_column].notna().any():
        raise RuntimeError(
            "Forecast-time leakage detected: "
            "future target value is present."
        )

    return result.reset_index(drop=True)


# ============================================================
# RECURSIVE HISTORY APPENDER
# ============================================================


def append_forecast_to_history(
    history: pd.DataFrame,
    forecast_row: pd.DataFrame,
    prediction: float,
    target_column: str = "quantity_sold",
) -> pd.DataFrame:

    if forecast_row.empty:
        raise ValueError(
            "Forecast row cannot be empty."
        )

    if prediction < 0:
        prediction = 0.0

    row = forecast_row.copy()

    row[target_column] = float(
        prediction
    )

    updated = pd.concat(
        [
            history,
            row,
        ],
        ignore_index=True,
        sort=False,
    )

    updated["date"] = pd.to_datetime(
        updated["date"]
    )

    updated = updated.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    return updated