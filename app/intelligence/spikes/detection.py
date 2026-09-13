from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "outlet_id",
    "product_id",
    "date",
}


def detect_demand_spikes(
    df: pd.DataFrame,
    demand_column: str = "quantity_sold",
    baseline_window: int = 28,
    minimum_history: int = 7,
) -> pd.DataFrame:
    """
    Part 23A - Demand Spike Detection.

    Detects unusual demand relative to the historical behavior
    of each outlet-product series.

    Causal contract:

        baseline(t) uses only demand from t-1 and earlier.

    Current demand is used only to determine whether the current
    observation is a spike label. It is never included in the
    baseline itself.
    """

    required = (
        REQUIRED_COLUMNS
        | {demand_column}
    )

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if baseline_window < 2:
        raise ValueError(
            "baseline_window must be >= 2"
        )

    if minimum_history < 1:
        raise ValueError(
            "minimum_history must be >= 1"
        )

    if df.empty:
        return df.copy()

    out = df.copy()

    out["date"] = pd.to_datetime(
        out["date"]
    )

    out["_p23_original_order"] = np.arange(
        len(out)
    )

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

    demand = pd.to_numeric(
        out[demand_column],
        errors="coerce",
    ).fillna(0.0)

    demand = demand.clip(
        lower=0.0
    )

    out["_p23_demand"] = demand

    # ============================================================
    # STRICT HISTORICAL BASELINE
    # ============================================================

    history = (
        out.groupby(
            group_keys,
            sort=False,
        )["_p23_demand"]
        .shift(1)
    )

    out["_p23_history"] = history

    history_group = out.groupby(
        group_keys,
        sort=False,
    )["_p23_demand"]

    # Historical rolling mean.
    out["spike_baseline_mean"] = (
        history_group
        .rolling(
            window=baseline_window,
            min_periods=minimum_history,
        )
        .mean()
        .reset_index(
            level=group_keys,
            drop=True,
        )
    )

    # Historical rolling standard deviation.
    out["spike_baseline_std"] = (
        history_group
        .rolling(
            window=baseline_window,
            min_periods=minimum_history,
        )
        .std()
        .reset_index(
            level=group_keys,
            drop=True,
        )
    )

    # Historical rolling median.
    out["spike_baseline_median"] = (
        history_group
        .rolling(
            window=baseline_window,
            min_periods=minimum_history,
        )
        .median()
        .reset_index(
            level=group_keys,
            drop=True,
        )
    )

    # Historical observation count.
    out["spike_history_count"] = (
        history_group
        .rolling(
            window=baseline_window,
            min_periods=1,
        )
        .count()
        .reset_index(
            level=group_keys,
            drop=True,
        )
    )

    # ============================================================
    # DEMAND DEVIATION
    # ============================================================

    eps = 1e-6

    baseline = (
        out["spike_baseline_mean"]
        .fillna(
            out["spike_baseline_median"]
        )
    )

    out["spike_baseline"] = baseline

    out["spike_absolute_lift"] = (
        out["_p23_demand"]
        - baseline
    )

    out["spike_ratio"] = (
        out["_p23_demand"]
        /
        (
            baseline.abs()
            + eps
        )
    )

    out["spike_percentage_lift"] = (
        (
            out["_p23_demand"]
            -
            baseline
        )
        /
        (
            baseline.abs()
            + eps
        )
    )

    # ============================================================
    # Z-SCORE
    # ============================================================

    out["spike_z_score"] = (
        (
            out["_p23_demand"]
            -
            baseline
        )
        /
        (
            out["spike_baseline_std"].abs()
            + eps
        )
    )

    # ============================================================
    # ROBUST DEVIATION
    # ============================================================

    # Rolling historical median absolute deviation.
    historical_deviation = (
        (
            history
            -
            out["spike_baseline_median"]
        )
        .abs()
    )

    mad_group = historical_deviation.groupby(
        [
            out["outlet_id"],
            out["product_id"],
        ],
        sort=False,
    )

    # A robust fallback based on the available rolling std
    # is used when MAD is unavailable.
    out["spike_mad_proxy"] = (
        historical_deviation
    )

    robust_scale = (
        out["spike_baseline_std"]
        .fillna(0.0)
    )

    out["spike_robust_score"] = (
        (
            out["_p23_demand"]
            -
            baseline
        ).abs()
        /
        (
            robust_scale.abs()
            + eps
        )
    )

    # ============================================================
    # HISTORY AVAILABILITY
    # ============================================================

    out["spike_history_ready"] = (
        out["spike_history_count"]
        >= minimum_history
    )

    # ============================================================
    # SPIKE FLAGS
    # ============================================================

    positive_deviation = (
        out["spike_absolute_lift"]
        > 0
    )

    ratio_signal = (
        out["spike_ratio"]
        >= 1.50
    )

    z_signal = (
        out["spike_z_score"]
        >= 2.0
    )

    # A spike requires sufficient history and both:
    #
    #   1. materially elevated demand
    #   2. statistically unusual demand
    #
    out["is_demand_spike"] = (
        out["spike_history_ready"]
        &
        positive_deviation
        &
        ratio_signal
        &
        z_signal
    )

    # ============================================================
    # SPIKE SCORE
    # ============================================================

    ratio_component = (
        (
            out["spike_ratio"]
            - 1.0
        )
        / 1.0
    ).clip(
        lower=0.0,
        upper=3.0,
    )

    z_component = (
        out["spike_z_score"]
        / 4.0
    ).clip(
        lower=0.0,
        upper=3.0,
    )

    out["spike_score"] = (
        0.50 * ratio_component
        +
        0.50 * z_component
    ).clip(
        lower=0.0,
        upper=3.0,
    )

    # ============================================================
    # DIRECTION
    # ============================================================

    out["demand_deviation_direction"] = np.where(
        out["spike_absolute_lift"] > 0,
        "above_baseline",
        np.where(
            out["spike_absolute_lift"] < 0,
            "below_baseline",
            "at_baseline",
        ),
    )

    # ============================================================
    # CLEANUP
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

    out = (
        out
        .sort_values(
            "_p23_original_order"
        )
        .reset_index(drop=True)
    )

    out.drop(
        columns=[
            "_p23_original_order",
            "_p23_demand",
            "_p23_history",
            "spike_mad_proxy",
        ],
        inplace=True,
        errors="ignore",
    )

    return out