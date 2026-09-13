import pandas as pd

from app.intelligence.outlet.behavior import (
    build_outlet_demand_behavior,
)
from app.intelligence.outlet.service import (
    OutletProfilingService,
)


def make_demand():
    rows = []

    for day in pd.date_range(
        "2025-01-01",
        periods=30,
    ):
        for outlet_id in ["O001", "O002"]:
            for product_id in [
                "P001",
                "P002",
                "P003",
            ]:

                if outlet_id == "O001":
                    base = 10
                else:
                    base = 5

                weekend = day.dayofweek >= 5

                if outlet_id == "O001" and weekend:
                    demand = base * 1.5
                else:
                    demand = base

                rows.append(
                    {
                        "outlet_id": outlet_id,
                        "product_id": product_id,
                        "date": day,
                        "quantity_sold": demand,
                        "deconstrained_demand": demand,
                    }
                )

    return pd.DataFrame(rows)


def test_behavior_returns_one_row_per_outlet():
    result = build_outlet_demand_behavior(
        make_demand()
    )

    assert len(result) == 2
    assert result["outlet_id"].nunique() == 2


def test_behavior_metrics_exist():
    result = build_outlet_demand_behavior(
        make_demand()
    )

    required = {
        "behavior_average_daily_demand",
        "behavior_median_daily_demand",
        "behavior_std_daily_demand",
        "behavior_peak_intensity",
        "behavior_coefficient_variation",
        "behavior_weekday_average",
        "behavior_weekend_average",
        "behavior_weekend_weekday_ratio",
        "behavior_high_demand_day_rate",
        "behavior_low_demand_day_rate",
        "behavior_top_3_product_share",
        "behavior_top_5_product_share",
        "behavior_product_count",
        "behavior_trend",
        "demand_concentration_class",
        "peak_intensity_class",
        "demand_consistency_class",
        "demand_shape_class",
        "trend_class",
    }

    assert required.issubset(result.columns)


def test_behavior_values_are_non_negative_where_required():
    result = build_outlet_demand_behavior(
        make_demand()
    )

    numeric_columns = [
        "behavior_average_daily_demand",
        "behavior_median_daily_demand",
        "behavior_std_daily_demand",
        "behavior_min_daily_demand",
        "behavior_max_daily_demand",
        "behavior_peak_intensity",
        "behavior_coefficient_variation",
        "behavior_high_demand_day_rate",
        "behavior_low_demand_day_rate",
        "behavior_top_3_product_share",
        "behavior_top_5_product_share",
    ]

    for column in numeric_columns:
        assert (
            result[column] >= 0
        ).all()


def test_behavior_rates_are_bounded():
    result = build_outlet_demand_behavior(
        make_demand()
    )

    bounded_columns = [
        "behavior_high_demand_day_rate",
        "behavior_low_demand_day_rate",
        "behavior_top_3_product_share",
        "behavior_top_5_product_share",
    ]

    for column in bounded_columns:
        assert (
            result[column] <= 1
        ).all()


def test_weekend_behavior_detects_weekend_effect():
    result = build_outlet_demand_behavior(
        make_demand()
    )

    row = result.loc[
        result["outlet_id"] == "O001"
    ].iloc[0]

    assert (
        row["behavior_weekend_weekday_ratio"]
        > 1.0
    )


def test_behavior_classes_exist():
    result = build_outlet_demand_behavior(
        make_demand()
    )

    assert result[
        "demand_shape_class"
    ].notna().all()

    assert result[
        "demand_consistency_class"
    ].notna().all()

    assert result[
        "trend_class"
    ].notna().all()


def test_service_behavior_interface():
    service = OutletProfilingService()

    result = service.analyze_behavior(
        make_demand()
    )

    assert len(result) == 2


def test_top_behavioral_outlets():
    service = OutletProfilingService()

    behavior = service.analyze_behavior(
        make_demand()
    )

    result = service.top_behavioral_outlets(
        behavior,
        limit=1,
    )

    assert len(result) == 1
    assert (
        "outlet_id" in result.columns
    )