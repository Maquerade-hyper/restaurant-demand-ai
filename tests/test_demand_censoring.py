from __future__ import annotations

import pandas as pd

from app.intelligence.demand.censoring import (
    DemandCensoringService,
)


def make_data() -> pd.DataFrame:

    dates = pd.date_range(
        "2025-01-01",
        periods=30,
        freq="D",
    )

    rows = []

    for i, date in enumerate(dates):

        stockout = 10 <= i <= 14

        observed = 10.0 if not stockout else 2.0
        demand = 10.0 if not stockout else 10.0
        lost = demand - observed

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

    result = DemandCensoringService().analyze(
        make_data()
    )

    assert len(result) == 30


def test_stockout_event_detected():

    result = DemandCensoringService().summarize_events(
        make_data()
    )

    assert len(result) == 1
    assert result.iloc[0]["event_duration_days"] == 5


def test_censored_demand_detected():

    result = DemandCensoringService().analyze(
        make_data()
    )

    stockout = result[result["stockout"]]

    assert stockout["censored_demand_flag"].all()


def test_censoring_strength_valid():

    result = DemandCensoringService().analyze(
        make_data()
    )

    assert result["censoring_strength"].between(
        0,
        1,
    ).all()


def test_censoring_class_exists():

    result = DemandCensoringService().analyze(
        make_data()
    )

    assert "censoring_class" in result.columns
    assert (
        result.loc[
            result["stockout"],
            "censoring_class",
        ]
        .isin(
            [
                "weak",
                "moderate",
                "strong",
            ]
        )
        .all()
    )


def test_event_lost_demand():

    result = DemandCensoringService().summarize_events(
        make_data()
    )

    assert result.iloc[0]["event_lost_demand"] == 40.0


def test_recovery_event_detected():

    result = DemandCensoringService().analyze(
        make_data()
    )

    assert result["recovery_event"].any()


def test_outlet_summary():

    result = DemandCensoringService().summarize_outlets(
        make_data()
    )

    assert len(result) == 1
    assert result.iloc[0]["stockout_days"] == 5


def test_product_summary():

    result = DemandCensoringService().summarize_products(
        make_data()
    )

    assert len(result) == 1
    assert result.iloc[0]["censored_days"] == 5


def test_top_opportunities():

    result = DemandCensoringService().top_censored_opportunities(
        make_data(),
        limit=5,
    )

    assert len(result) > 0
    assert (
        "censoring_priority"
        in result.columns
    )


def test_no_truth_required():

    df = make_data()

    assert "true_demand" not in df.columns
    assert "lost_demand_truth" not in df.columns

    result = DemandCensoringService().analyze(df)

    assert len(result) == len(df)