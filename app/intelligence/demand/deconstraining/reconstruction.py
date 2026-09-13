from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_median(values: pd.Series) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        values.median()
    )


def _safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        values.mean()
    )


def build_clean_reference(
    df: pd.DataFrame,
    lookback_days: int = 56,
    minimum_clean_days: int = 7,
) -> pd.DataFrame:
    """
    Build a demand reference using only observations that were
    not identified as stockout-constrained.

    The reference combines:

    1. Recent clean observations.
    2. Day-of-week clean observations.
    3. Product/outlet historical behavior.

    No future information is used.
    """

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result = result.sort_values(
        ["outlet_id", "product_id", "date"]
    ).reset_index(drop=True)

    group_columns = [
        "outlet_id",
        "product_id",
    ]

    result["clean_sales"] = np.where(
        ~result["stockout"],
        result["quantity_sold"],
        np.nan,
    )

    group = result.groupby(
        group_columns,
        sort=False,
    )

    result["recent_clean_mean"] = (
        group["clean_sales"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    lookback_days,
                    min_periods=minimum_clean_days,
                )
                .mean()
            )
        )
    )

    result["recent_clean_median"] = (
        group["clean_sales"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    lookback_days,
                    min_periods=minimum_clean_days,
                )
                .median()
            )
        )
    )

    result["day_of_week"] = (
        result["date"].dt.dayofweek
    )

    # Only previous clean observations are eligible.
    result["dow_clean_sum"] = (
        result.assign(
            clean_value=result["clean_sales"]
        )
        .groupby(
            group_columns + ["day_of_week"],
            sort=False,
        )["clean_value"]
        .transform(
            lambda x: x.shift(1).expanding().sum()
        )
    )

    result["dow_clean_count"] = (
        result.assign(
            clean_value=result["clean_sales"]
        )
        .groupby(
            group_columns + ["day_of_week"],
            sort=False,
        )["clean_value"]
        .transform(
            lambda x: (
                x.notna()
                .shift(1)
                .astype("boolean")
                .fillna(False)
                .expanding()
                .sum()
            )
        )
    )

    result["dow_clean_mean"] = (
        result["dow_clean_sum"]
        / result["dow_clean_count"].replace(
            0,
            np.nan,
        )
    )

    # Long-run clean reference.
    result["all_clean_mean"] = (
        group["clean_sales"]
        .transform(
            lambda x: (
                x.shift(1)
                .expanding(
                    min_periods=minimum_clean_days
                )
                .mean()
            )
        )
    )

    return result


def reconstruct_demand(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate unconstrained demand.

    The observed sales quantity is always treated as a lower bound.

    For clean days:
        estimated demand = observed sales

    For constrained days:
        estimated demand is reconstructed from historical clean demand.
    """

    result = df.copy()

    recent_mean = (
        result["recent_clean_mean"]
    )

    recent_median = (
        result["recent_clean_median"]
    )

    dow_mean = (
        result["dow_clean_mean"]
    )

    all_mean = (
        result["all_clean_mean"]
    )

    # Hierarchical fallback.
    reference = recent_mean.copy()

    reference = reference.fillna(
        recent_median
    )

    reference = reference.fillna(
        dow_mean
    )

    reference = reference.fillna(
        all_mean
    )

    # Final fallback to observed sales.
    reference = reference.fillna(
        result["quantity_sold"]
    )

    # ---------------------------------------------------------
    # Recency / stability blend
    # ---------------------------------------------------------

    reference_components = pd.concat(
        [
            recent_mean.rename("recent"),
            recent_median.rename("median"),
            dow_mean.rename("dow"),
            all_mean.rename("all"),
        ],
        axis=1,
    )

    available_count = (
        reference_components.notna()
        .sum(axis=1)
    )

    mean_reference = (
        reference_components.mean(
            axis=1,
            skipna=True,
        )
    )

    reference = np.where(
        available_count >= 2,
        (
            0.45 * reference_components["recent"].fillna(
                mean_reference
            )
            + 0.25 * reference_components["median"].fillna(
                mean_reference
            )
            + 0.20 * reference_components["dow"].fillna(
                mean_reference
            )
            + 0.10 * reference_components["all"].fillna(
                mean_reference
            )
        ),
        reference,
    )

    reference = pd.Series(
        reference,
        index=result.index,
    )

    # ---------------------------------------------------------
    # Stockout-duration adjustment
    # ---------------------------------------------------------

    duration = (
        result["stockout_duration"]
        .fillna(0)
        .astype(float)
    )

    # Longer stockouts provide stronger evidence that observed
    # sales are censored. We modestly increase the reconstructed
    # demand estimate rather than applying an arbitrary large jump.
    duration_factor = np.minimum(
        1.0 + 0.025 * np.maximum(
            duration - 1.0,
            0.0,
        ),
        1.10,
    )

    reference = (
        reference
        * duration_factor
    )

    # ---------------------------------------------------------
    # Observed sales lower bound
    # ---------------------------------------------------------

    estimated = np.maximum(
        result["quantity_sold"].astype(float),
        reference.astype(float),
    )

    # Clean observations should remain exactly observed.
    estimated = np.where(
        result["stockout"],
        estimated,
        result["quantity_sold"],
    )

    result["deconstrained_demand"] = (
        pd.Series(
            estimated,
            index=result.index,
        )
        .clip(lower=0.0)
    )

    result["estimated_lost_demand"] = (
        result["deconstrained_demand"]
        - result["quantity_sold"]
    ).clip(
        lower=0.0
    )

    result["estimated_fulfillment_rate"] = np.where(
        result["deconstrained_demand"] > 0,
        (
            result["quantity_sold"]
            / result["deconstrained_demand"]
        ).clip(
            0.0,
            1.0,
        ),
        1.0,
    )

    result["demand_constrained_estimated"] = (
        result["estimated_lost_demand"] > 0
    )

    return result