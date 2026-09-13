
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "outlet_id",
    "date",
    "quantity_sold",
    "deconstrained_demand",
    "estimated_lost_demand",
    "estimated_fulfillment_rate",
    "stockout",
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)

    minimum = values.min()
    maximum = values.max()

    if maximum <= minimum:
        return pd.Series(0.5, index=series.index)

    return (values - minimum) / (maximum - minimum)


def _classify_score(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "moderate"
    return "low"


def _classify_index(value: float) -> str:
    if value >= 75.0:
        return "high"
    if value >= 45.0:
        return "moderate"
    return "low"


def _trend(first: float, last: float) -> float:
    if first <= 0:
        return 0.0

    return float((last - first) / first)


def _calculate_outlet_trend(group: pd.DataFrame) -> float:
    ordered = group.sort_values("date")

    if len(ordered) < 14:
        return 0.0

    daily = (
        ordered.groupby("date", as_index=False)["deconstrained_demand"]
        .sum()
        .sort_values("date")
    )

    if len(daily) < 14:
        return 0.0

    window = max(7, len(daily) // 4)

    first_mean = float(
        daily["deconstrained_demand"].iloc[:window].mean()
    )

    last_mean = float(
        daily["deconstrained_demand"].iloc[-window:].mean()
    )

    return _trend(first_mean, last_mean)


def _calculate_performance_row(
    outlet_id: str,
    group: pd.DataFrame,
) -> dict:
    ordered = group.sort_values("date")

    observed = float(ordered["quantity_sold"].sum())
    demand = float(ordered["deconstrained_demand"].sum())
    lost = float(ordered["estimated_lost_demand"].sum())

    fulfillment = _safe_ratio(observed, demand)
    lost_rate = _safe_ratio(lost, demand)

    stockout_product_days = float(
        ordered["stockout"].sum()
    )

    total_product_days = float(len(ordered))

    stockout_rate = _safe_ratio(
        stockout_product_days,
        total_product_days,
    )

    affected_days = int(
        ordered.groupby("date")["stockout"]
        .any()
        .sum()
    )

    total_days = int(
        ordered["date"].nunique()
    )

    stockout_affected_day_rate = _safe_ratio(
        affected_days,
        total_days,
    )

    daily_demand = (
        ordered.groupby("date")["deconstrained_demand"]
        .sum()
    )

    average_daily_demand = float(
        daily_demand.mean()
    )

    demand_std = float(
        daily_demand.std(ddof=0)
    )

    cv = (
        _safe_ratio(
            demand_std,
            average_daily_demand,
        )
        if average_daily_demand > 0
        else 0.0
    )

    high_threshold = float(
        daily_demand.quantile(0.90)
    )

    high_demand_rate = _safe_ratio(
        float(
            (daily_demand >= high_threshold).sum()
        ),
        float(len(daily_demand)),
    )

    peak_intensity = (
        _safe_ratio(
            float(daily_demand.max()),
            average_daily_demand,
        )
        if average_daily_demand > 0
        else 0.0
    )

    trend = _calculate_outlet_trend(ordered)

    # ---------------------------------
    # Transparent performance score
    # ---------------------------------

    performance_raw = (
        0.50 * fulfillment
        + 0.20 * max(
            0.0,
            min(
                1.0,
                (trend + 0.20) / 0.40,
            ),
        )
        + 0.15 * (1.0 - min(cv, 1.0))
        + 0.15 * (
            1.0
            - min(
                stockout_rate / 0.25,
                1.0,
            )
        )
    )

    performance_score = float(
        np.clip(
            performance_raw,
            0.0,
            1.0,
        )
        * 100.0
    )

    # ---------------------------------
    # Opportunity score
    # ---------------------------------

    opportunity_raw = (
        0.55 * lost_rate
        + 0.25 * min(
            stockout_rate / 0.25,
            1.0,
        )
        + 0.20 * max(
            0.0,
            min(
                1.0,
                (trend + 0.20) / 0.40,
            ),
        )
    )

    opportunity_score = float(
        np.clip(
            opportunity_raw,
            0.0,
            1.0,
        )
        * 100.0
    )

    # ---------------------------------
    # Operational risk score
    # ---------------------------------

    risk_raw = (
        0.45 * min(
            lost_rate / 0.25,
            1.0,
        )
        + 0.35 * min(
            stockout_rate / 0.25,
            1.0,
        )
        + 0.20 * min(
            cv / 0.50,
            1.0,
        )
    )

    operational_risk_score = float(
        np.clip(
            risk_raw,
            0.0,
            1.0,
        )
        * 100.0
    )

    return {
        "outlet_id": outlet_id,
        "performance_days": total_days,
        "total_observed_demand": observed,
        "total_deconstrained_demand": demand,
        "total_estimated_lost_demand": lost,
        "fulfillment_rate": fulfillment,
        "lost_demand_rate": lost_rate,
        "stockout_product_day_rate": stockout_rate,
        "stockout_affected_day_rate": stockout_affected_day_rate,
        "stockout_days": affected_days,
        "stockout_product_days": int(
            stockout_product_days
        ),
        "average_daily_demand": average_daily_demand,
        "demand_std": demand_std,
        "coefficient_variation": cv,
        "high_demand_day_rate": high_demand_rate,
        "peak_intensity": peak_intensity,
        "demand_growth_rate": trend,
        "performance_score": performance_score,
        "opportunity_score": opportunity_score,
        "operational_risk_score": operational_risk_score,
        "performance_class": _classify_score(
            performance_score / 100.0
        ),
    }


def build_outlet_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    numeric_columns = [
        "quantity_sold",
        "deconstrained_demand",
        "estimated_lost_demand",
        "estimated_fulfillment_rate",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        ).fillna(0.0)

    data["stockout"] = (
        data["stockout"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    rows = []

    for outlet_id, group in data.groupby(
        "outlet_id"
    ):
        rows.append(
            _calculate_performance_row(
                outlet_id,
                group,
            )
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # ---------------------------------
    # Population-relative indexes
    # ---------------------------------

    result["opportunity_index"] = (
        100.0
        * (
            0.60
            * _minmax(
                result[
                    "total_estimated_lost_demand"
                ]
            )
            + 0.40
            * _minmax(
                result[
                    "opportunity_score"
                ]
            )
        )
    )

    result["risk_index"] = (
        100.0
        * (
            0.60
            * _minmax(
                result[
                    "operational_risk_score"
                ]
            )
            + 0.40
            * _minmax(
                result[
                    "total_estimated_lost_demand"
                ]
            )
        )
    )

    # ---------------------------------
    # Classes are derived from the
    # final population-relative indexes
    # ---------------------------------

    result["opportunity_class"] = (
        result["opportunity_index"]
        .apply(_classify_index)
    )

    result["operational_risk_class"] = (
        result["risk_index"]
        .apply(_classify_index)
    )

    # ---------------------------------
    # Rankings
    # ---------------------------------

    result["performance_rank"] = (
        result["performance_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result["opportunity_rank"] = (
        result["opportunity_index"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result["risk_rank"] = (
        result["risk_index"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return (
        result
        .sort_values("performance_rank")
        .reset_index(drop=True)
    )

