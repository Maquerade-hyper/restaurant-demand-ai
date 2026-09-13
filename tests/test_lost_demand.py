import numpy as np
import pandas as pd

from app.intelligence.demand.lost_demand import (
    LostDemandIntelligenceService,
)


def make_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=40,
        freq="D",
    )

    rows = []

    for i, date in enumerate(dates):

        if i < 25:

            observed = 10.0
            demand = 10.0
            lost = 0.0
            stockout = False

        elif i < 30:

            observed = 5.0
            demand = 10.0
            lost = 5.0
            stockout = True

        else:

            observed = 10.0
            demand = 10.0
            lost = 0.0
            stockout = False

        rows.append(
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": date,
                "quantity_sold": observed,
                "deconstrained_demand": demand,
                "estimated_lost_demand": lost,
                "stockout": stockout,
            }
        )

    return pd.DataFrame(rows)


def test_service_returns_rows():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.analyze(df)

    assert len(result) == len(df)


def test_lost_demand_rate_is_valid():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.analyze(df)

    assert (
        result["lost_demand_rate"]
        >= 0
    ).all()

    assert (
        result["lost_demand_rate"]
        <= 1
    ).all()


def test_lost_demand_flag():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.analyze(df)

    constrained = result[
        result["estimated_lost_demand"] > 0
    ]

    assert (
        constrained["lost_demand_flag"]
    ).all()


def test_stockout_pattern_detected():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.analyze(df)

    stockout_rows = result[
        result["stockout"]
    ]

    assert len(stockout_rows) == 5

    assert (
        stockout_rows["stockout_days_7d"]
        >= 0
    ).all()


def test_severity_exists():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.analyze(df)

    valid = {
        "low",
        "moderate",
        "high",
        "critical",
    }

    assert set(
        result["loss_severity"]
    ).issubset(valid)


def test_priority_score_exists():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.analyze(df)

    assert (
        "priority_score"
        in result.columns
    )

    assert (
        result["priority_score"]
        >= 0
    ).all()


def test_outlet_summary():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.summarize_outlets(
        df
    )

    assert len(result) == 1

    assert (
        result.iloc[0][
            "total_estimated_lost_demand"
        ]
        > 0
    )


def test_product_summary():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.summarize_products(
        df
    )

    assert len(result) == 1


def test_outlet_product_summary():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.summarize_outlet_products(
        df
    )

    assert len(result) == 1


def test_top_opportunities():

    df = make_data()

    service = LostDemandIntelligenceService()

    result = service.top_opportunities(
        df,
        limit=5,
    )

    assert len(result) <= 5

    assert (
        "priority_score"
        in result.columns
    )


def test_no_truth_required():

    df = make_data()

    assert "true_demand" not in df.columns
    assert "lost_demand_truth" not in df.columns

    service = LostDemandIntelligenceService()

    result = service.analyze(df)

    assert not result.empty