from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_OUTLET_COLUMNS = {
    "outlet_id",
    "outlet_type",
    "country",
    "region",
    "city",
    "location_type",
    "capacity",
    "delivery_available",
    "takeaway_available",
    "bar_available",
    "kitchen_type",
    "tourism_level",
    "business_area",
    "residential_area",
    "student_area",
}

REQUIRED_DEMAND_COLUMNS = {
    "outlet_id",
    "product_id",
    "date",
    "quantity_sold",
    "deconstrained_demand",
    "estimated_lost_demand",
    "stockout",
}


def _safe_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    return np.where(
        denominator.abs() > 1e-12,
        numerator / denominator,
        0.0,
    )


def _classify_volume(value, q33, q66):
    if value <= q33:
        return "low"
    if value <= q66:
        return "medium"
    return "high"


def _classify_volatility(value, q33, q66):
    if value <= q33:
        return "stable"
    if value <= q66:
        return "moderate"
    return "volatile"


def _classify_weekend_behavior(ratio):
    if ratio >= 1.20:
        return "weekend_heavy"
    if ratio <= 0.85:
        return "weekday_heavy"
    return "balanced"


def _classify_trend(value):
    if value >= 0.10:
        return "growing"
    if value <= -0.10:
        return "declining"
    return "stable"


def _classify_stockout_risk(value):
    if value >= 0.20:
        return "high"
    if value >= 0.10:
        return "moderate"
    return "low"


def _classify_lost_demand_risk(value):
    if value >= 0.15:
        return "high"
    if value >= 0.05:
        return "moderate"
    return "low"


def _calculate_trend(group: pd.DataFrame) -> float:
    group = group.sort_values("date")

    daily_demand = (
        group.groupby("date")["deconstrained_demand"]
        .sum()
        .sort_index()
    )

    if len(daily_demand) < 14:
        return 0.0

    split = max(1, len(daily_demand) // 2)

    first_mean = float(
        daily_demand.iloc[:split].mean()
    )

    last_mean = float(
        daily_demand.iloc[split:].mean()
    )

    if abs(first_mean) < 1e-12:
        return 0.0

    return float(
        (last_mean - first_mean) / first_mean
    )


def _calculate_trends(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate outlet-level demand trend without DataFrameGroupBy.apply,
    avoiding pandas future behavior changes.
    """

    records = []

    for outlet_id, group in daily.groupby(
        "outlet_id",
        sort=False,
    ):
        records.append(
            {
                "outlet_id": outlet_id,
                "demand_trend": _calculate_trend(group),
            }
        )

    return pd.DataFrame(records)


def build_outlet_profiles(
    outlets: pd.DataFrame,
    demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one behavioral profile per outlet.

    Demand input must come from the demand-intelligence layer and must
    not contain or require demand_truth.csv.
    """

    missing_outlets = (
        REQUIRED_OUTLET_COLUMNS
        - set(outlets.columns)
    )

    if missing_outlets:
        raise ValueError(
            f"Missing outlet columns: "
            f"{sorted(missing_outlets)}"
        )

    missing_demand = (
        REQUIRED_DEMAND_COLUMNS
        - set(demand.columns)
    )

    if missing_demand:
        raise ValueError(
            f"Missing demand columns: "
            f"{sorted(missing_demand)}"
        )

    outlet_df = outlets.copy()
    demand_df = demand.copy()

    outlet_df["outlet_id"] = (
        outlet_df["outlet_id"].astype(str)
    )

    demand_df["outlet_id"] = (
        demand_df["outlet_id"].astype(str)
    )

    demand_df["date"] = pd.to_datetime(
        demand_df["date"],
        errors="coerce",
    )

    if demand_df["date"].isna().any():
        raise ValueError(
            "Demand data contains invalid dates."
        )

    numeric_columns = [
        "quantity_sold",
        "deconstrained_demand",
        "estimated_lost_demand",
    ]

    for column in numeric_columns:
        demand_df[column] = pd.to_numeric(
            demand_df[column],
            errors="coerce",
        ).fillna(0.0)

    demand_df["stockout"] = (
        demand_df["stockout"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    demand_df["deconstrained_demand"] = (
        demand_df["deconstrained_demand"]
        .clip(lower=0.0)
    )

    demand_df["estimated_lost_demand"] = (
        demand_df["estimated_lost_demand"]
        .clip(lower=0.0)
    )

    # ---------------------------------------------------------
    # Daily outlet aggregation
    # ---------------------------------------------------------

    daily = (
        demand_df.groupby(
            ["outlet_id", "date"],
            as_index=False,
        )
        .agg(
            observed_demand=(
                "quantity_sold",
                "sum",
            ),
            deconstrained_demand=(
                "deconstrained_demand",
                "sum",
            ),
            estimated_lost_demand=(
                "estimated_lost_demand",
                "sum",
            ),
            stockout_products=(
                "stockout",
                "sum",
            ),
            total_products=(
                "product_id",
                "nunique",
            ),
        )
    )

    daily["lost_demand_rate"] = _safe_ratio(
        daily["estimated_lost_demand"],
        daily["deconstrained_demand"],
    )

    daily["fulfillment_rate"] = np.clip(
        _safe_ratio(
            daily["observed_demand"],
            daily["deconstrained_demand"],
        ),
        0.0,
        1.0,
    )

    daily["stockout_affected"] = (
        daily["stockout_products"] > 0
    )

    # ---------------------------------------------------------
    # Outlet-level base metrics
    # ---------------------------------------------------------

    profiles = (
        daily.groupby("outlet_id")
        .agg(
            active_days=("date", "nunique"),
            average_daily_demand=(
                "deconstrained_demand",
                "mean",
            ),
            total_demand=(
                "deconstrained_demand",
                "sum",
            ),
            total_observed_sales=(
                "observed_demand",
                "sum",
            ),
            total_lost_demand=(
                "estimated_lost_demand",
                "sum",
            ),
            demand_std=(
                "deconstrained_demand",
                "std",
            ),
            demand_min=(
                "deconstrained_demand",
                "min",
            ),
            demand_max=(
                "deconstrained_demand",
                "max",
            ),
            stockout_affected_days=(
                "stockout_affected",
                "sum",
            ),
            stockout_product_days=(
                "stockout_products",
                "sum",
            ),
            total_product_days=(
                "total_products",
                "sum",
            ),
            average_fulfillment_rate=(
                "fulfillment_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    profiles["demand_std"] = (
        profiles["demand_std"]
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Correct stockout metrics
    # ---------------------------------------------------------

    profiles["stockout_product_day_rate"] = (
        _safe_ratio(
            profiles["stockout_product_days"],
            profiles["total_product_days"],
        )
    )

    profiles["stockout_affected_day_rate"] = (
        _safe_ratio(
            profiles["stockout_affected_days"],
            profiles["active_days"],
        )
    )

    # Keep stockout_rate as the corrected primary metric.
    profiles["stockout_rate"] = (
        profiles["stockout_product_day_rate"]
    )

    # ---------------------------------------------------------
    # Demand metrics
    # ---------------------------------------------------------

    profiles["demand_cv"] = _safe_ratio(
        profiles["demand_std"],
        profiles["average_daily_demand"],
    )

    profiles["lost_demand_rate"] = _safe_ratio(
        profiles["total_lost_demand"],
        profiles["total_demand"],
    )

    # ---------------------------------------------------------
    # Weekday / weekend behavior
    # ---------------------------------------------------------

    daily["day_of_week"] = (
        daily["date"].dt.dayofweek
    )

    daily["is_weekend"] = (
        daily["day_of_week"] >= 5
    )

    weekday = (
        daily.loc[~daily["is_weekend"]]
        .groupby("outlet_id")[
            "deconstrained_demand"
        ]
        .mean()
        .rename("weekday_average_demand")
    )

    weekend = (
        daily.loc[daily["is_weekend"]]
        .groupby("outlet_id")[
            "deconstrained_demand"
        ]
        .mean()
        .rename("weekend_average_demand")
    )

    profiles = profiles.merge(
        weekday,
        on="outlet_id",
        how="left",
    )

    profiles = profiles.merge(
        weekend,
        on="outlet_id",
        how="left",
    )

    profiles["weekday_average_demand"] = (
        profiles["weekday_average_demand"]
        .fillna(0.0)
    )

    profiles["weekend_average_demand"] = (
        profiles["weekend_average_demand"]
        .fillna(0.0)
    )

    profiles["weekend_weekday_ratio"] = (
        _safe_ratio(
            profiles["weekend_average_demand"],
            profiles["weekday_average_demand"],
        )
    )

    # ---------------------------------------------------------
    # Product breadth
    # ---------------------------------------------------------

    product_breadth = (
        demand_df.groupby("outlet_id")[
            "product_id"
        ]
        .nunique()
        .rename("active_product_count")
        .reset_index()
    )

    profiles = profiles.merge(
        product_breadth,
        on="outlet_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Demand trend
    # ---------------------------------------------------------

    trend = _calculate_trends(daily)

    profiles = profiles.merge(
        trend,
        on="outlet_id",
        how="left",
    )

    profiles["demand_trend"] = (
        profiles["demand_trend"]
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Behavioral classifications
    # ---------------------------------------------------------

    volume_q33 = (
        profiles["average_daily_demand"]
        .quantile(0.33)
    )

    volume_q66 = (
        profiles["average_daily_demand"]
        .quantile(0.66)
    )

    volatility_q33 = (
        profiles["demand_cv"]
        .quantile(0.33)
    )

    volatility_q66 = (
        profiles["demand_cv"]
        .quantile(0.66)
    )

    profiles["demand_volume_class"] = (
        profiles["average_daily_demand"]
        .apply(
            lambda x: _classify_volume(
                x,
                volume_q33,
                volume_q66,
            )
        )
    )

    profiles["demand_stability_class"] = (
        profiles["demand_cv"]
        .apply(
            lambda x: _classify_volatility(
                x,
                volatility_q33,
                volatility_q66,
            )
        )
    )

    profiles["weekend_behavior"] = (
        profiles["weekend_weekday_ratio"]
        .apply(_classify_weekend_behavior)
    )

    profiles["demand_trend_class"] = (
        profiles["demand_trend"]
        .apply(_classify_trend)
    )

    profiles["stockout_risk_class"] = (
        profiles["stockout_product_day_rate"]
        .apply(_classify_stockout_risk)
    )

    profiles["lost_demand_risk_class"] = (
        profiles["lost_demand_rate"]
        .apply(_classify_lost_demand_risk)
    )

    # ---------------------------------------------------------
    # Static outlet characteristics
    # ---------------------------------------------------------

    static_columns = [
        "outlet_id",
        "outlet_type",
        "country",
        "region",
        "city",
        "location_type",
        "cuisine",
        "capacity",
        "delivery_available",
        "takeaway_available",
        "bar_available",
        "kitchen_type",
        "tourism_level",
        "business_area",
        "residential_area",
        "student_area",
    ]

    static_columns = [
        column
        for column in static_columns
        if column in outlet_df.columns
    ]

    profiles = profiles.merge(
        outlet_df[
            static_columns
        ].drop_duplicates("outlet_id"),
        on="outlet_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Behavioral profile
    # ---------------------------------------------------------

    profiles["outlet_behavior_class"] = (
        profiles["demand_volume_class"]
        + "_"
        + profiles["demand_stability_class"]
    )

    # ---------------------------------------------------------
    # Operational risk
    #
    # Risk is based on meaningful outlet-level signals:
    #   - lost demand
    #   - product-day stockouts
    #
    # We deliberately do not classify an outlet as high risk
    # merely because one product stocked out on one day.
    # ---------------------------------------------------------

    profiles["operational_risk_class"] = np.select(
        [
            (
                (profiles["lost_demand_rate"] >= 0.15)
                | (
                    profiles[
                        "stockout_product_day_rate"
                    ] >= 0.20
                )
            ),
            (
                (profiles["lost_demand_rate"] >= 0.05)
                | (
                    profiles[
                        "stockout_product_day_rate"
                    ] >= 0.10
                )
            ),
        ],
        [
            "high",
            "moderate",
        ],
        default="low",
    )

    # ---------------------------------------------------------
    # Reproducible output
    # ---------------------------------------------------------

    profiles = (
        profiles
        .sort_values("outlet_id")
        .reset_index(drop=True)
    )

    return profiles