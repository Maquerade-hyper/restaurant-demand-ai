from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "outlet_id",
    "product_id",
    "date",
    "deconstrained_demand",
}


def _safe_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    return np.where(
        denominator.abs() > 1e-12,
        numerator / denominator,
        0.0,
    )


def _classify_concentration(value):
    if value >= 0.50:
        return "highly_concentrated"
    if value >= 0.30:
        return "concentrated"
    if value >= 0.15:
        return "balanced"
    return "highly_distributed"


def _classify_peak_intensity(value):
    if value >= 1.75:
        return "very_high"
    if value >= 1.40:
        return "high"
    if value >= 1.15:
        return "moderate"
    return "low"


def _classify_consistency(value):
    if value <= 0.15:
        return "highly_consistent"
    if value <= 0.30:
        return "consistent"
    if value <= 0.50:
        return "variable"
    return "highly_variable"


def _classify_demand_shape(
    weekday_weekend_ratio: float,
    peak_intensity: float,
    cv: float,
):
    if cv >= 0.50 and peak_intensity >= 1.40:
        return "volatile_peak_driven"

    if weekday_weekend_ratio >= 1.20:
        return "weekday_driven"

    if weekday_weekend_ratio <= 0.85:
        return "weekend_driven"

    if peak_intensity >= 1.40:
        return "peak_driven"

    if cv <= 0.15:
        return "stable"

    return "balanced_variable"


def _calculate_trend(group: pd.DataFrame) -> float:
    daily = (
        group.groupby("date")["deconstrained_demand"]
        .sum()
        .sort_index()
    )

    if len(daily) < 14:
        return 0.0

    split = max(1, len(daily) // 2)

    first_mean = float(
        daily.iloc[:split].mean()
    )

    last_mean = float(
        daily.iloc[split:].mean()
    )

    if abs(first_mean) <= 1e-12:
        return 0.0

    return float(
        (last_mean - first_mean) / first_mean
    )


def _calculate_outlet_behavior(group: pd.DataFrame) -> dict:
    group = group.sort_values("date").copy()

    daily = (
        group.groupby("date")["deconstrained_demand"]
        .sum()
        .sort_index()
    )

    if daily.empty:
        return {}

    mean_demand = float(daily.mean())
    std_demand = float(daily.std(ddof=0))
    median_demand = float(daily.median())

    q25 = float(daily.quantile(0.25))
    q75 = float(daily.quantile(0.75))

    peak_demand = float(daily.max())

    peak_intensity = (
        peak_demand / mean_demand
        if mean_demand > 1e-12
        else 0.0
    )

    coefficient_variation = (
        std_demand / mean_demand
        if mean_demand > 1e-12
        else 0.0
    )

    interquartile_range = q75 - q25

    consistency_ratio = (
        interquartile_range / median_demand
        if median_demand > 1e-12
        else 0.0
    )

    # Weekday / weekend behavior.
    temp = group.copy()

    temp["day_of_week"] = (
        temp["date"].dt.dayofweek
    )

    temp["is_weekend"] = (
        temp["day_of_week"] >= 5
    )

    daily_temp = (
        temp.groupby(
            ["date", "is_weekend"],
            as_index=False,
        )["deconstrained_demand"]
        .sum()
    )

    weekday_values = daily_temp.loc[
        ~daily_temp["is_weekend"],
        "deconstrained_demand",
    ]

    weekend_values = daily_temp.loc[
        daily_temp["is_weekend"],
        "deconstrained_demand",
    ]

    weekday_mean = (
        float(weekday_values.mean())
        if not weekday_values.empty
        else 0.0
    )

    weekend_mean = (
        float(weekend_values.mean())
        if not weekend_values.empty
        else 0.0
    )

    weekend_weekday_ratio = (
        weekend_mean / weekday_mean
        if weekday_mean > 1e-12
        else 0.0
    )

    # Demand concentration by product.
    product_demand = (
        group.groupby("product_id")[
            "deconstrained_demand"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    total_product_demand = float(
        product_demand.sum()
    )

    if total_product_demand > 1e-12:
        top_3_product_share = float(
            product_demand.head(3).sum()
            / total_product_demand
        )

        top_5_product_share = float(
            product_demand.head(5).sum()
            / total_product_demand
        )
    else:
        top_3_product_share = 0.0
        top_5_product_share = 0.0

    # High/low demand day frequency.
    high_threshold = float(
        daily.quantile(0.90)
    )

    low_threshold = float(
        daily.quantile(0.10)
    )

    high_demand_day_rate = float(
        (daily >= high_threshold).mean()
    )

    low_demand_day_rate = float(
        (daily <= low_threshold).mean()
    )

    trend = _calculate_trend(group)

    return {
        "behavior_days": int(len(daily)),
        "behavior_average_daily_demand": mean_demand,
        "behavior_median_daily_demand": median_demand,
        "behavior_std_daily_demand": std_demand,
        "behavior_min_daily_demand": float(daily.min()),
        "behavior_max_daily_demand": peak_demand,
        "behavior_q25_daily_demand": q25,
        "behavior_q75_daily_demand": q75,
        "behavior_interquartile_range": interquartile_range,
        "behavior_coefficient_variation": coefficient_variation,
        "behavior_consistency_ratio": consistency_ratio,
        "behavior_peak_intensity": peak_intensity,
        "behavior_weekday_average": weekday_mean,
        "behavior_weekend_average": weekend_mean,
        "behavior_weekend_weekday_ratio": weekend_weekday_ratio,
        "behavior_high_demand_day_rate": high_demand_day_rate,
        "behavior_low_demand_day_rate": low_demand_day_rate,
        "behavior_top_3_product_share": top_3_product_share,
        "behavior_top_5_product_share": top_5_product_share,
        "behavior_product_count": int(
            product_demand.shape[0]
        ),
        "behavior_trend": trend,
    }


def build_outlet_demand_behavior(
    demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build temporal demand-behavior profiles for each outlet.

    This function uses deconstrained historical demand only.
    It does not require or consume demand_truth.csv.
    """

    missing = REQUIRED_COLUMNS - set(demand.columns)

    if missing:
        raise ValueError(
            f"Missing demand columns: {sorted(missing)}"
        )

    df = demand.copy()

    df["outlet_id"] = (
        df["outlet_id"].astype(str)
    )

    df["product_id"] = (
        df["product_id"].astype(str)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError(
            "Demand data contains invalid dates."
        )

    df["deconstrained_demand"] = pd.to_numeric(
        df["deconstrained_demand"],
        errors="coerce",
    ).fillna(0.0)

    df["deconstrained_demand"] = (
        df["deconstrained_demand"]
        .clip(lower=0.0)
    )

    records = []

    for outlet_id, group in df.groupby(
        "outlet_id",
        sort=False,
    ):
        metrics = _calculate_outlet_behavior(group)

        metrics["outlet_id"] = outlet_id

        records.append(metrics)

    if not records:
        return pd.DataFrame(
            columns=["outlet_id"]
        )

    result = pd.DataFrame(records)

    # ---------------------------------------------------------
    # Cross-outlet behavioral classifications
    # ---------------------------------------------------------

    result["demand_concentration_class"] = (
        result["behavior_top_3_product_share"]
        .apply(_classify_concentration)
    )

    result["peak_intensity_class"] = (
        result["behavior_peak_intensity"]
        .apply(_classify_peak_intensity)
    )

    result["demand_consistency_class"] = (
        result["behavior_coefficient_variation"]
        .apply(_classify_consistency)
    )

    result["demand_shape_class"] = result.apply(
        lambda row: _classify_demand_shape(
            row[
                "behavior_weekend_weekday_ratio"
            ],
            row["behavior_peak_intensity"],
            row[
                "behavior_coefficient_variation"
            ],
        ),
        axis=1,
    )

    result["trend_class"] = np.select(
        [
            result["behavior_trend"] >= 0.10,
            result["behavior_trend"] <= -0.10,
        ],
        [
            "growing",
            "declining",
        ],
        default="stable",
    )

    result = (
        result
        .sort_values("outlet_id")
        .reset_index(drop=True)
    )

    return result