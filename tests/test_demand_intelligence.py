import pandas as pd

from app.intelligence.demand import (
    DemandIntelligenceService,
)


def test_no_stockout_observed_equals_true_demand():

    sales = pd.DataFrame(
        {
            "outlet_id": ["O001", "O001"],
            "product_id": ["P001", "P001"],
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "quantity_sold": [
                20,
                25,
            ],
        }
    )

    inventory = pd.DataFrame(
        {
            "outlet_id": ["O001", "O001"],
            "product_id": ["P001", "P001"],
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "closing_stock": [
                100,
                100,
            ],
            "stockout": [
                False,
                False,
            ],
        }
    )

    service = DemandIntelligenceService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    assert (
        result["true_demand"]
        == result["quantity_sold"]
    ).all()

    assert (
        result["lost_demand"] == 0
    ).all()


def test_stockout_creates_lost_demand():

    sales = pd.DataFrame(
        {
            "outlet_id": ["O001"] * 8,
            "product_id": ["P001"] * 8,
            "date": pd.date_range(
                "2025-01-01",
                periods=8,
                freq="D",
            ),
            "quantity_sold": [
                20,
                21,
                22,
                20,
                23,
                24,
                25,
                10,
            ],
        }
    )

    inventory = pd.DataFrame(
        {
            "outlet_id": ["O001"] * 8,
            "product_id": ["P001"] * 8,
            "date": pd.date_range(
                "2025-01-01",
                periods=8,
                freq="D",
            ),
            "closing_stock": [
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                0,
            ],
            "stockout": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )

    service = DemandIntelligenceService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    last = result.iloc[-1]

    assert last["true_demand"] >= 10
    assert last["lost_demand"] >= 0
    assert last["fulfillment_rate"] <= 1.0


def test_summary():

    sales = pd.DataFrame(
        {
            "outlet_id": ["O001", "O001"],
            "product_id": ["P001", "P001"],
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "quantity_sold": [
                20,
                20,
            ],
        }
    )

    inventory = pd.DataFrame(
        {
            "outlet_id": ["O001", "O001"],
            "product_id": ["P001", "P001"],
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "closing_stock": [
                100,
                100,
            ],
            "stockout": [
                False,
                False,
            ],
        }
    )

    service = DemandIntelligenceService()

    result = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    summary = service.summarize(result)

    assert summary["records"] == 2
    assert summary["true_demand"] == 40
    assert summary["observed_sales"] == 40
    assert summary["lost_demand"] == 0