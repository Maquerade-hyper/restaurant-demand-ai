from __future__ import annotations

import numpy as np
import pandas as pd

from .events import build_stockout_events


def calculate_censoring_signals(
    df: pd.DataFrame,
    pre_window: int = 7,
) -> pd.DataFrame:
    """
    Identify whether observed demand is likely censored by stockout.

    Uses only information available around the observed stockout event.
    """

    result = build_stockout_events(df)

    groups = ["outlet_id", "product_id"]

    result["pre_stockout_demand"] = (
        result.groupby(groups)["deconstrained_demand"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    pre_window,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    result["pre_stockout_sales"] = (
        result.groupby(groups)["quantity_sold"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    pre_window,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    result["sales_demand_gap"] = (
        result["deconstrained_demand"]
        - result["quantity_sold"]
    ).clip(lower=0)

    result["censoring_ratio"] = (
        result["sales_demand_gap"]
        / result["deconstrained_demand"].clip(lower=1e-9)
    ).clip(0.0, 1.0)

    result["pre_stockout_demand_pressure"] = (
        result["pre_stockout_demand"]
        / result["pre_stockout_sales"].clip(lower=1e-9)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result["pre_stockout_demand_pressure"] = (
        result["pre_stockout_demand_pressure"]
        .fillna(1.0)
        .clip(lower=0.0, upper=5.0)
    )

    result["censored_demand_flag"] = (
        result["stockout"]
        & (
            (
                result["estimated_lost_demand"] > 0
            )
            | (
                result["censoring_ratio"] > 0
            )
        )
    )

    result["censoring_strength"] = np.where(
        result["stockout"],
        (
            0.50 * result["censoring_ratio"]
            + 0.30
            * (
                result["estimated_lost_demand"]
                / result["deconstrained_demand"].clip(
                    lower=1e-9
                )
            ).clip(0.0, 1.0)
            + 0.20
            * (
                result["pre_stockout_demand_pressure"]
                .sub(1.0)
                .clip(lower=0.0, upper=1.0)
            )
        ),
        0.0,
    )

    result["censoring_strength"] = (
        result["censoring_strength"]
        .clip(0.0, 1.0)
    )

    return result


def classify_censoring(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    strength = result["censoring_strength"]

    result["censoring_class"] = np.select(
        [
            ~result["stockout"],
            strength >= 0.75,
            strength >= 0.50,
            strength >= 0.25,
        ],
        [
            "not_censored",
            "strong",
            "moderate",
            "weak",
        ],
        default="minimal",
    )

    return result