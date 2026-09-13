import pandas as pd
import pytest

from app.intelligence.outlet.performance import (
    build_outlet_performance,
)
from app.intelligence.outlet.service import (
    OutletProfilingService,
)


def sample_data():
    dates = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    rows = []

    for outlet in ["O001", "O002"]:
        for date in dates:
            rows.append(
                {
                    "outlet_id": outlet,
                    "date": date,
                    "quantity_sold": 90.0,
                    "deconstrained_demand": 100.0,
                    "estimated_lost_demand": 10.0,
                    "estimated_fulfillment_rate": 0.90,
                    "stockout": outlet == "O002",
                }
            )

    return pd.DataFrame(rows)


def test_one_row_per_outlet():
    result = build_outlet_performance(sample_data())

    assert len(result) == 2
    assert result["outlet_id"].nunique() == 2


def test_required_metrics_exist():
    result = build_outlet_performance(sample_data())

    required = {
        "performance_score",
        "opportunity_score",
        "operational_risk_score",
        "fulfillment_rate",
        "lost_demand_rate",
        "stockout_product_day_rate",
        "opportunity_index",
        "risk_index",
        "performance_rank",
        "opportunity_rank",
        "risk_rank",
    }

    assert required.issubset(result.columns)


def test_scores_are_bounded():
    result = build_outlet_performance(sample_data())

    for column in [
        "performance_score",
        "opportunity_score",
        "operational_risk_score",
        "opportunity_index",
        "risk_index",
    ]:
        assert result[column].between(0, 100).all()


def test_fulfillment_is_correct():
    result = build_outlet_performance(sample_data())

    assert result["fulfillment_rate"].iloc[0] == pytest.approx(0.90)


def test_lost_demand_rate_is_correct():
    result = build_outlet_performance(sample_data())

    assert result["lost_demand_rate"].iloc[0] == pytest.approx(0.10)


def test_stockout_rate_is_product_day_rate():
    result = build_outlet_performance(sample_data())

    o001 = result.loc[
        result["outlet_id"] == "O001",
        "stockout_product_day_rate",
    ].iloc[0]

    o002 = result.loc[
        result["outlet_id"] == "O002",
        "stockout_product_day_rate",
    ].iloc[0]

    assert o001 == pytest.approx(0.0)
    assert o002 == pytest.approx(1.0)


def test_ranks_are_unique_for_distinct_scores():
    result = build_outlet_performance(sample_data())

    assert result["performance_rank"].nunique() == 2
    assert result["risk_rank"].nunique() == 2


def test_service_exposes_performance():
    service = OutletProfilingService()

    assert hasattr(service, "analyze_performance")
    assert hasattr(service, "top_performing_outlets")
    assert hasattr(service, "top_opportunity_outlets")
    assert hasattr(service, "top_operational_risk_outlets")


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        build_outlet_performance(
            pd.DataFrame(
                {
                    "outlet_id": ["O001"],
                }
            )
        )