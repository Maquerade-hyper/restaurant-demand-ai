import numpy as np
import pandas as pd

from app.intelligence.demand.deconstraining import (
    DemandDeconstrainingService,
    DeconstrainedDemandEstimator,
)


def make_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=40,
        freq="D",
    )

    rows = []

    for i, date in enumerate(dates):

        observed = 10.0

        stockout = False
        closing_stock = 20.0

        if i >= 30:
            stockout = True
            observed = 5.0
            closing_stock = 0.0

        rows.append(
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": date,
                "quantity_sold": observed,
            }
        )

    sales = pd.DataFrame(rows)

    inventory = pd.DataFrame(
        {
            "outlet_id": ["O001"] * 40,
            "product_id": ["P001"] * 40,
            "date": dates,
            "closing_stock": [
                20.0 if i < 30 else 0.0
                for i in range(40)
            ],
            "stockout": [
                False if i < 30 else True
                for i in range(40)
            ],
        }
    )

    return sales, inventory


def test_deconstrained_estimator_returns_rows():

    sales, inventory = make_data()

    estimator = DeconstrainedDemandEstimator(
        lookback_days=28,
        minimum_clean_days=7,
    )

    result = estimator.fit_transform(
        sales=sales,
        inventory=inventory,
    )

    assert len(result) == 40


def test_clean_days_equal_observed_sales():

    sales, inventory = make_data()

    service = DemandDeconstrainingService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    clean = result[
        ~result["stockout"]
    ]

    assert np.allclose(
        clean["deconstrained_demand"],
        clean["quantity_sold"],
    )


def test_stockout_demand_not_below_observed_sales():

    sales, inventory = make_data()

    service = DemandDeconstrainingService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    assert (
        result["deconstrained_demand"]
        >= result["quantity_sold"]
    ).all()


def test_stockout_days_can_recover_hidden_demand():

    sales, inventory = make_data()

    service = DemandDeconstrainingService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    stockout = result[
        result["stockout"]
    ]

    assert (
        stockout["deconstrained_demand"]
        > stockout["quantity_sold"]
    ).any()


def test_lost_demand_never_negative():

    sales, inventory = make_data()

    service = DemandDeconstrainingService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    assert (
        result["estimated_lost_demand"]
        >= 0
    ).all()


def test_fulfillment_between_zero_and_one():

    sales, inventory = make_data()

    service = DemandDeconstrainingService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    assert (
        result["estimated_fulfillment_rate"]
        >= 0
    ).all()

    assert (
        result["estimated_fulfillment_rate"]
        <= 1
    ).all()


def test_no_truth_columns_required():

    sales, inventory = make_data()

    # Explicitly verify that the estimator works without
    # any causal truth information.
    assert "true_demand" not in sales.columns
    assert "lost_demand" not in sales.columns

    estimator = DeconstrainedDemandEstimator()

    result = estimator.fit_transform(
        sales=sales,
        inventory=inventory,
    )

    assert "deconstrained_demand" in result.columns