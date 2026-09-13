from __future__ import annotations

import numpy as np
import pandas as pd


SEVERITY_WEIGHT = {
    "low": 1.0,
    "moderate": 1.5,
    "high": 2.5,
    "critical": 4.0,
}


def calculate_priority_score(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a prioritization score for operational action.

    Score combines:

        lost demand
        +
        lost demand rate
        +
        recurrence
        +
        recent acceleration
        +
        severity

    This is an intelligence ranking, not a monetary estimate.
    """

    result = df.copy()

    lost = (
        result["estimated_lost_demand"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    rate = (
        result["lost_demand_rate"]
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    recurrence = (
        result.get(
            "stockout_days_28d",
            pd.Series(
                0.0,
                index=result.index,
            ),
        )
        .fillna(0.0)
    )

    acceleration = (
        result.get(
            "loss_acceleration",
            pd.Series(
                1.0,
                index=result.index,
            ),
        )
        .fillna(1.0)
        .clip(
            lower=0.0,
            upper=10.0,
        )
    )

    severity = (
        result.get(
            "loss_severity",
            pd.Series(
                "low",
                index=result.index,
            ),
        )
        .map(SEVERITY_WEIGHT)
        .fillna(1.0)
    )

    # Log transform prevents very large outlets from
    # completely dominating the ranking.
    volume_component = np.log1p(
        lost
    )

    rate_component = (
        rate * 10.0
    )

    recurrence_component = (
        np.minimum(
            recurrence,
            28.0,
        )
        / 7.0
    )

    acceleration_component = (
        np.minimum(
            acceleration,
            4.0,
        )
    )

    result["priority_score"] = (
        volume_component
        * (
            1.0
            + rate_component
        )
        * (
            1.0
            + recurrence_component
        )
        * (
            0.5
            + acceleration_component
        )
        * severity
    )

    result["priority_rank"] = (
        result["priority_score"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    result["priority_band"] = np.select(
        [
            result["priority_score"] >= 50,
            result["priority_score"] >= 20,
            result["priority_score"] >= 5,
        ],
        [
            "critical",
            "high",
            "medium",
        ],
        default="low",
    )

    return result