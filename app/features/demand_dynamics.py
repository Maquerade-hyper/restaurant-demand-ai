from __future__ import annotations

import numpy as np
import pandas as pd


def add_demand_dynamics(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
) -> pd.DataFrame:
    """
    Part 22B - Advanced demand dynamics.

    All demand-derived features are strictly historical.

    For date t:

        lag_1 = demand(t-1)

    Therefore the current target is never used to create
    the current-row demand features.
    """

    required = {
        "date",
        "outlet_id",
        "product_id",
        target_column,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        return df.copy()

    out = df.copy()

    out["date"] = pd.to_datetime(
        out["date"]
    )

    # Preserve original row order.
    out["_advanced_original_order"] = np.arange(
        len(out)
    )

    # Sort chronologically for every outlet-product series.
    out = (
        out.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    group_keys = [
        "outlet_id",
        "product_id",
    ]

    # ============================================================
    # HISTORICAL DEMAND
    # ============================================================

    out["_advanced_historical_demand"] = (
        out.groupby(
            group_keys,
            sort=False,
        )[target_column]
        .shift(1)
        .astype(float)
    )

    history = out["_advanced_historical_demand"]

    grouped = out.groupby(
        group_keys,
        sort=False,
    )

    # ============================================================
    # LAGS
    # ============================================================

    out["advanced_demand_lag_1"] = history

    out["advanced_demand_lag_2"] = (
        grouped["_advanced_historical_demand"]
        .shift(1)
    )

    out["advanced_demand_lag_7"] = (
        grouped["_advanced_historical_demand"]
        .shift(6)
    )

    out["advanced_demand_lag_14"] = (
        grouped["_advanced_historical_demand"]
        .shift(13)
    )

    out["advanced_demand_lag_28"] = (
        grouped["_advanced_historical_demand"]
        .shift(27)
    )

    lag1 = out["advanced_demand_lag_1"]
    lag2 = out["advanced_demand_lag_2"]
    lag7 = out["advanced_demand_lag_7"]
    lag14 = out["advanced_demand_lag_14"]
    lag28 = out["advanced_demand_lag_28"]

    # ============================================================
    # DIFFERENCES
    # ============================================================

    out["advanced_demand_diff_1"] = (
        lag1 - lag2
    )

    out["advanced_demand_diff_7"] = (
        lag1 - lag7
    )

    out["advanced_demand_diff_14"] = (
        lag1 - lag14
    )

    out["advanced_demand_diff_28"] = (
        lag1 - lag28
    )

    # ============================================================
    # RATIOS
    # ============================================================

    eps = 1e-6

    out["advanced_lag_ratio_1_7"] = (
        lag1
        / lag7.replace(0, np.nan)
    )

    out["advanced_lag_ratio_1_14"] = (
        lag1
        / lag14.replace(0, np.nan)
    )

    out["advanced_lag_ratio_1_28"] = (
        lag1
        / lag28.replace(0, np.nan)
    )

    out["advanced_lag_ratio_7_14"] = (
        lag7
        / lag14.replace(0, np.nan)
    )

    out["advanced_lag_ratio_14_28"] = (
        lag14
        / lag28.replace(0, np.nan)
    )

    # ============================================================
    # CAUSAL ROLLING FEATURES
    # ============================================================

    history_group = out.groupby(
        group_keys,
        sort=False,
    )["_advanced_historical_demand"]

    def rolling_mean(
        window: int,
    ) -> pd.Series:

        return (
            history_group
            .rolling(
                window=window,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=group_keys,
                drop=True,
            )
        )

    def rolling_std(
        window: int,
    ) -> pd.Series:

        return (
            history_group
            .rolling(
                window=window,
                min_periods=2,
            )
            .std()
            .reset_index(
                level=group_keys,
                drop=True,
            )
        )

    out["advanced_mean_3"] = rolling_mean(3)
    out["advanced_mean_7"] = rolling_mean(7)
    out["advanced_mean_14"] = rolling_mean(14)
    out["advanced_mean_28"] = rolling_mean(28)

    out["advanced_std_7"] = rolling_std(7)
    out["advanced_std_14"] = rolling_std(14)
    out["advanced_std_28"] = rolling_std(28)

    # ============================================================
    # MOMENTUM
    # ============================================================

    out["advanced_momentum_3_7"] = (
        out["advanced_mean_3"]
        - out["advanced_mean_7"]
    )

    out["advanced_momentum_7_14"] = (
        out["advanced_mean_7"]
        - out["advanced_mean_14"]
    )

    out["advanced_momentum_14_28"] = (
        out["advanced_mean_14"]
        - out["advanced_mean_28"]
    )

    # ============================================================
    # ACCELERATION
    # ============================================================

    out["advanced_acceleration"] = (
        out["advanced_momentum_3_7"]
        - out["advanced_momentum_7_14"]
    )

    # ============================================================
    # PERSISTENCE
    # ============================================================

    # Primary public contract.
    out["advanced_persistence"] = (
        lag1
        / (
            out["advanced_mean_7"].abs()
            + eps
        )
    )

    # More specific variants retained.
    out["advanced_persistence_7"] = (
        out["advanced_persistence"]
    )

    out["advanced_persistence_14"] = (
        lag1
        / (
            out["advanced_mean_14"].abs()
            + eps
        )
    )

    # ============================================================
    # INTENSITY
    # ============================================================

    out["advanced_intensity_7"] = (
        out["advanced_mean_7"]
        / (
            out["advanced_mean_28"].abs()
            + eps
        )
    )

    out["advanced_intensity_14"] = (
        out["advanced_mean_14"]
        / (
            out["advanced_mean_28"].abs()
            + eps
        )
    )

    # ============================================================
    # NON-ZERO / INTERMITTENCY
    # ============================================================

    out["advanced_nonzero_rate_28"] = (
        history_group
        .rolling(
            window=28,
            min_periods=1,
        )
        .apply(
            lambda x: float(
                np.mean(
                    np.asarray(x) > 0
                )
            ),
            raw=True,
        )
        .reset_index(
            level=group_keys,
            drop=True,
        )
    )

    out["advanced_intermittency_28"] = (
        1.0
        - out["advanced_nonzero_rate_28"]
    )

    # ============================================================
    # COEFFICIENT OF VARIATION
    # ============================================================

    out["advanced_cv_7"] = (
        out["advanced_std_7"]
        / (
            out["advanced_mean_7"].abs()
            + eps
        )
    )

    out["advanced_cv_14"] = (
        out["advanced_std_14"]
        / (
            out["advanced_mean_14"].abs()
            + eps
        )
    )

    out["advanced_cv_28"] = (
        out["advanced_std_28"]
        / (
            out["advanced_mean_28"].abs()
            + eps
        )
    )

    # ============================================================
    # CLEAN NUMERICAL VALUES
    # ============================================================

    numeric_columns = out.select_dtypes(
        include=[np.number]
    ).columns

    out[numeric_columns] = (
        out[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    # ============================================================
    # REMOVE INTERNAL COLUMNS
    # ============================================================

    out.drop(
        columns=[
            "_advanced_historical_demand",
        ],
        inplace=True,
        errors="ignore",
    )

    # ============================================================
    # RESTORE ORIGINAL ORDER
    # ============================================================

    out = (
        out
        .sort_values(
            "_advanced_original_order"
        )
        .drop(
            columns=[
                "_advanced_original_order",
            ]
        )
        .reset_index(drop=True)
    )

    return out