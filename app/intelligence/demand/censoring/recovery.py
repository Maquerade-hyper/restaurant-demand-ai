from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_post_stockout_recovery(
    df: pd.DataFrame,
    recovery_window: int = 7,
) -> pd.DataFrame:

    result = df.copy()

    result["date"] = pd.to_datetime(result["date"])

    groups = ["outlet_id", "product_id"]

    previous_stockout = (
        result.groupby(groups)["stockout"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    result["stockout_end"] = (
        previous_stockout
        & ~result["stockout"]
    )

    result["recovery_event"] = (
        result["stockout_end"]
    )

    result["days_since_stockout"] = (
        result.groupby(groups)["stockout"]
        .transform(
            lambda x: (
                x.astype(int)
                .groupby(
                    x.ne(x.shift()).cumsum()
                )
                .cumcount()
            )
        )
    )

    clean_demand = (
        result["deconstrained_demand"]
        .where(~result["stockout"])
    )

    result["clean_reference_demand"] = (
        clean_demand
        .groupby(
            [
                result["outlet_id"],
                result["product_id"],
            ]
        )
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    recovery_window,
                    min_periods=1,
                )
                .median()
            )
        )
    )

    result["recovery_ratio"] = (
        result["deconstrained_demand"]
        / result["clean_reference_demand"].clip(
            lower=1e-9
        )
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result["recovery_ratio"] = (
        result["recovery_ratio"]
        .fillna(1.0)
        .clip(0.0, 5.0)
    )

    result["post_stockout_recovery_flag"] = (
        result["recovery_event"]
        & (
            result["recovery_ratio"] > 1.10
        )
    )

    result["recovery_strength"] = np.where(
        result["recovery_event"],
        (
            result["recovery_ratio"] - 1.0
        ).clip(0.0, 1.0),
        0.0,
    )

    return result


def summarize_recovery(
    df: pd.DataFrame,
) -> pd.DataFrame:

    recovery = df[
        df["recovery_event"]
    ].copy()

    if recovery.empty:
        return pd.DataFrame(
            columns=[
                "outlet_id",
                "product_id",
                "recovery_events",
                "strong_recovery_events",
                "average_recovery_ratio",
                "max_recovery_ratio",
            ]
        )

    return (
        recovery.groupby(
            ["outlet_id", "product_id"],
            as_index=False,
        )
        .agg(
            recovery_events=(
                "recovery_event",
                "sum",
            ),
            strong_recovery_events=(
                "post_stockout_recovery_flag",
                "sum",
            ),
            average_recovery_ratio=(
                "recovery_ratio",
                "mean",
            ),
            max_recovery_ratio=(
                "recovery_ratio",
                "max",
            ),
        )
    )