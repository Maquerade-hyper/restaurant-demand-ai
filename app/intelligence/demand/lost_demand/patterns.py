from __future__ import annotations

import numpy as np
import pandas as pd


def add_loss_patterns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect recurring lost-demand patterns.

    The logic uses only historical estimated lost demand.
    """

    required = {
        "outlet_id",
        "product_id",
        "date",
        "estimated_lost_demand",
        "lost_demand_rate",
        "stockout",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result = result.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    group_columns = [
        "outlet_id",
        "product_id",
    ]

    group = result.groupby(
        group_columns,
        sort=False,
    )

    result["lost_demand_7d"] = (
        group["estimated_lost_demand"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    7,
                    min_periods=1,
                )
                .sum()
            )
        )
    )

    result["lost_demand_28d"] = (
        group["estimated_lost_demand"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    28,
                    min_periods=1,
                )
                .sum()
            )
        )
    )

    result["lost_demand_rate_7d"] = (
        group["lost_demand_rate"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    7,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    result["stockout_days_7d"] = (
        result.groupby(["outlet_id", "product_id"])["stockout"]
        .transform(
            lambda x: (
                x.shift(1)
                .astype("boolean")
                .fillna(False)
                .astype(int)
                .rolling(7, min_periods=1)
                .sum()
            )
        )
    )

    result["stockout_days_28d"] = (
        result.groupby(["outlet_id", "product_id"])["stockout"]
        .transform(
            lambda x: (
                x.shift(1)
                .astype("boolean")
                .fillna(False)
                .astype(int)
                .rolling(28, min_periods=1)
                .sum()
            )
        )
    )
    result["loss_event_start"] = (
        result["estimated_lost_demand"] > 0
    ) & ~(
        group["estimated_lost_demand"]
        .shift(1)
        .fillna(0)
        > 0
    )

    result["recurring_loss_flag"] = (
        result["lost_demand_28d"]
        > 0
    ) & (
        result["stockout_days_28d"]
        >= 2
    )

    result["persistent_loss_flag"] = (
        result["stockout_days_7d"]
        >= 3
    )

    # Recent loss acceleration.
    result["loss_acceleration"] = (
        result["lost_demand_7d"]
        / (
            result["lost_demand_28d"]
            / 4.0
        ).replace(
            0,
            np.nan,
        )
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(1.0)

    return result


def classify_loss_severity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign operational severity to estimated lost demand.
    """

    result = df.copy()

    rate = (
        result["lost_demand_rate"]
        .fillna(0.0)
    )

    recent_loss = (
        result["lost_demand_7d"]
        .fillna(0.0)
    )

    stockout_days = (
        result["stockout_days_7d"]
        .fillna(0.0)
    )

    conditions = [
        (
            rate >= 0.50
        ) | (
            stockout_days >= 5
        ),

        (
            rate >= 0.25
        ) | (
            stockout_days >= 3
        ),

        (
            rate >= 0.10
        ) | (
            recent_loss > 0
        ),
    ]

    choices = [
        "critical",
        "high",
        "moderate",
    ]

    result["loss_severity"] = np.select(
        conditions,
        choices,
        default="low",
    )

    return result