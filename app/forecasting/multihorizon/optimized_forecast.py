from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "day_of_year",
    "quarter",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_3",
    "rolling_std_3",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
    "trend_1d",
    "trend_7d",
    "trend_14d",
    "trend_ratio_7d",
    "trend_ratio_14d",
    "trend_acceleration",
    "volatility_3",
    "coefficient_variation_3",
    "volatility_7",
    "coefficient_variation_7",
    "volatility_14",
    "coefficient_variation_14",
    "volatility_28",
    "coefficient_variation_28",
    "order_velocity_3d",
    "order_velocity_7d",
    "order_velocity_14d",
    "order_velocity_change",
    "dow_sin_2",
    "dow_cos_2",
    "month_sin_2",
    "month_cos_2",
    "quarter_sin",
    "quarter_cos",
    "weekend_month_interaction",
]


def _rolling_stats(
    states: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized equivalent of:

        shifted.groupby(...).rolling(window).mean()
        shifted.groupby(...).rolling(window).std()

    states contains the historical values available before
    the forecast date.

    Each row represents one outlet/product series.
    """

    if states.shape[1] < window:
        shape = states.shape[0]
        return (
            np.full(shape, np.nan),
            np.full(shape, np.nan),
        )

    values = states[:, -window:]

    mean = np.mean(
        values,
        axis=1,
    )

    std = np.std(
        values,
        axis=1,
        ddof=1,
    )

    return mean, std


def build_batch_feature_matrix(
    states: list[np.ndarray],
    forecast_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build all 51 model features for every series using
    vectorized NumPy operations.

    No per-series feature function calls.
    No groupby.
    No sorting.
    No rolling pandas operations.
    """

    if not states:
        return pd.DataFrame(
            columns=MODEL_FEATURES
        )

    # ---------------------------------------------------------
    # Normalize states into a fixed 2-D matrix.
    #
    # Every series has at least 28 historical observations
    # for the production forecasting path.
    # ---------------------------------------------------------

    n_series = len(states)

    max_history = 28

    history = np.empty(
        (
            n_series,
            max_history,
        ),
        dtype=np.float64,
    )

    for i, state in enumerate(states):

        values = np.asarray(
            state,
            dtype=np.float64,
        )

        if len(values) < max_history:
            padded = np.full(
                max_history,
                np.nan,
                dtype=np.float64,
            )

            padded[-len(values):] = values

            history[i] = padded

        else:
            history[i] = values[-max_history:]

    date = pd.Timestamp(
        forecast_date
    )

    # ---------------------------------------------------------
    # CALENDAR FEATURES
    # ---------------------------------------------------------

    year = date.year
    month = date.month
    day = date.day
    day_of_week = date.dayofweek
    week_of_year = int(
        date.isocalendar().week
    )
    day_of_year = date.dayofyear
    quarter = date.quarter
    is_weekend = int(
        day_of_week >= 5
    )

    # ---------------------------------------------------------
    # LAGS
    # ---------------------------------------------------------

    lag_1 = history[:, -1]
    lag_2 = history[:, -2]
    lag_3 = history[:, -3]
    lag_7 = history[:, -7]
    lag_14 = history[:, -14]
    lag_28 = history[:, -28]

    # ---------------------------------------------------------
    # ROLLING STATISTICS
    # ---------------------------------------------------------

    rolling = {}

    for window in (
        3,
        7,
        14,
        28,
    ):
        mean, std = _rolling_stats(
            history,
            window,
        )

        rolling[window] = (
            mean,
            std,
        )

    rolling_mean_3, rolling_std_3 = rolling[3]
    rolling_mean_7, rolling_std_7 = rolling[7]
    rolling_mean_14, rolling_std_14 = rolling[14]
    rolling_mean_28, rolling_std_28 = rolling[28]

    # ---------------------------------------------------------
    # TREND
    # ---------------------------------------------------------

    trend_1d = (
        lag_1 - lag_2
    )

    trend_7d = (
        lag_1 - lag_7
    )

    trend_14d = (
        lag_1 - lag_14
    )

    trend_ratio_7d = np.divide(
        lag_1,
        lag_7,
        out=np.full(
            n_series,
            np.nan,
        ),
        where=(
            np.isfinite(lag_7)
            & (lag_7 != 0)
        ),
    )

    trend_ratio_14d = np.divide(
        lag_1,
        lag_14,
        out=np.full(
            n_series,
            np.nan,
        ),
        where=(
            np.isfinite(lag_14)
            & (lag_14 != 0)
        ),
    )

    trend_acceleration = (
        trend_7d - trend_14d
    )

    # ---------------------------------------------------------
    # VOLATILITY
    # ---------------------------------------------------------

    volatility_3 = rolling_std_3
    volatility_7 = rolling_std_7
    volatility_14 = rolling_std_14
    volatility_28 = rolling_std_28

    coefficient_variation_3 = np.divide(
        rolling_std_3,
        rolling_mean_3,
        out=np.full(
            n_series,
            np.nan,
        ),
        where=(
            np.isfinite(rolling_mean_3)
            & (rolling_mean_3 != 0)
        ),
    )

    coefficient_variation_7 = np.divide(
        rolling_std_7,
        rolling_mean_7,
        out=np.full(
            n_series,
            np.nan,
        ),
        where=(
            np.isfinite(rolling_mean_7)
            & (rolling_mean_7 != 0)
        ),
    )

    coefficient_variation_14 = np.divide(
        rolling_std_14,
        rolling_mean_14,
        out=np.full(
            n_series,
            np.nan,
        ),
        where=(
            np.isfinite(rolling_mean_14)
            & (rolling_mean_14 != 0)
        ),
    )

    coefficient_variation_28 = np.divide(
        rolling_std_28,
        rolling_mean_28,
        out=np.full(
            n_series,
            np.nan,
        ),
        where=(
            np.isfinite(rolling_mean_28)
            & (rolling_mean_28 != 0)
        ),
    )

    # ---------------------------------------------------------
    # ORDER VELOCITY
    # ---------------------------------------------------------

    order_velocity_3d = rolling_mean_3
    order_velocity_7d = rolling_mean_7
    order_velocity_14d = rolling_mean_14

    order_velocity_change = (
        order_velocity_3d
        - order_velocity_14d
    )

    # ---------------------------------------------------------
    # SEASONALITY
    # ---------------------------------------------------------

    dow_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    dow_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )

    month_sin = np.sin(
        2 * np.pi * (month - 1) / 12
    )

    month_cos = np.cos(
        2 * np.pi * (month - 1) / 12
    )

    dow_sin_2 = np.sin(
        4 * np.pi * day_of_week / 7
    )

    dow_cos_2 = np.cos(
        4 * np.pi * day_of_week / 7
    )

    month_sin_2 = np.sin(
        4 * np.pi * (month - 1) / 12
    )

    month_cos_2 = np.cos(
        4 * np.pi * (month - 1) / 12
    )

    quarter_sin = np.sin(
        2 * np.pi * (quarter - 1) / 4
    )

    quarter_cos = np.cos(
        2 * np.pi * (quarter - 1) / 4
    )

    weekend_month_interaction = (
        is_weekend * month
    )

    # ---------------------------------------------------------
    # SINGLE MATRIX
    # ---------------------------------------------------------

    matrix = np.column_stack(
        [
            np.full(n_series, year),
            np.full(n_series, month),
            np.full(n_series, day),
            np.full(n_series, day_of_week),
            np.full(n_series, week_of_year),
            np.full(n_series, day_of_year),
            np.full(n_series, quarter),
            np.full(n_series, is_weekend),

            np.full(n_series, dow_sin),
            np.full(n_series, dow_cos),
            np.full(n_series, month_sin),
            np.full(n_series, month_cos),

            lag_1,
            lag_2,
            lag_3,
            lag_7,
            lag_14,
            lag_28,

            rolling_mean_3,
            rolling_std_3,
            rolling_mean_7,
            rolling_std_7,
            rolling_mean_14,
            rolling_std_14,
            rolling_mean_28,
            rolling_std_28,

            trend_1d,
            trend_7d,
            trend_14d,
            trend_ratio_7d,
            trend_ratio_14d,
            trend_acceleration,

            volatility_3,
            coefficient_variation_3,
            volatility_7,
            coefficient_variation_7,
            volatility_14,
            coefficient_variation_14,
            volatility_28,
            coefficient_variation_28,

            order_velocity_3d,
            order_velocity_7d,
            order_velocity_14d,
            order_velocity_change,

            np.full(n_series, dow_sin_2),
            np.full(n_series, dow_cos_2),
            np.full(n_series, month_sin_2),
            np.full(n_series, month_cos_2),
            np.full(n_series, quarter_sin),
            np.full(n_series, quarter_cos),
            np.full(
                n_series,
                weekend_month_interaction,
            ),
        ]
    )

    return pd.DataFrame(
        matrix,
        columns=MODEL_FEATURES,
    )


def append_prediction(
    state: np.ndarray,
    prediction: float,
    max_history: int = 28,
) -> np.ndarray:

    prediction = max(
        float(prediction),
        0.0,
    )

    state = np.asarray(
        state,
        dtype=np.float64,
    )

    if len(state) < max_history:
        return np.append(
            state,
            prediction,
        )

    return np.concatenate(
        [
            state[-(max_history - 1):],
            np.array(
                [prediction],
                dtype=np.float64,
            ),
        ]
    )