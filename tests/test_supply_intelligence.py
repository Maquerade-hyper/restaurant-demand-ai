import pandas as pd
import pytest

from app.intelligence.supply import (
    SupplyIntelligenceService,
)


def sample_demand():
    return pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P002",
                "date": "2025-12-01",
                "demand": 20.0,
            },
            {
                "outlet_id": "O001",
                "product_id": "P002",
                "date": "2025-12-02",
                "demand": 30.0,
            },
            {
                "outlet_id": "O001",
                "product_id": "P004",
                "date": "2025-12-01",
                "demand": 10.0,
            },
        ]
    )


def sample_inventory():
    return pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "date": "2025-12-01",
                "product_id": "P002",
                "closing_stock": 20.0,
                "received_stock": 5.0,
            },
            {
                "outlet_id": "O001",
                "date": "2025-12-02",
                "product_id": "P002",
                "closing_stock": 10.0,
                "received_stock": 0.0,
            },
            {
                "outlet_id": "O001",
                "date": "2025-12-01",
                "product_id": "P004",
                "closing_stock": 50.0,
                "received_stock": 0.0,
            },
        ]
    )


def test_complete_supply_intelligence():

    service = SupplyIntelligenceService(
        safety_stock_days=1.0
    )

    demand = sample_demand()
    inventory = sample_inventory()

    result = service.analyze(
        demand=demand,
        inventory=inventory,
    )

    assert not result.empty

    required_columns = {
        "outlet_id",
        "supply_product_id",
        "required_supply",
        "lead_time_demand",
        "safety_stock",
        "target_inventory",
        "inventory_position",
        "shortage",
        "recommended_order_quantity",
        "supply_risk",
        "order_required",
    }

    assert required_columns.issubset(
        result.columns
    )

    assert (
        result["required_supply"] >= 0
    ).all()

    assert (
        result["recommended_order_quantity"] >= 0
    ).all()

    assert (
        result["supply_risk_ratio"]
        .between(0, 1)
        .all()
    )

    assert set(
        result["supply_risk"].unique()
    ).issubset(
        {"low", "moderate", "high"}
    )

    assert (
        result["order_required"]
        ==
        (
            result["recommended_order_quantity"] > 0
        )
    ).all()


def test_bom_demand_conversion():

    service = SupplyIntelligenceService()

    demand = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P002",
                "date": "2025-12-01",
                "demand": 25.0,
            }
        ]
    )

    result = service.demand_to_supply(demand)

    chicken = result[
        result["supply_product_id"] == "P002"
    ]

    assert len(chicken) == 1
    assert chicken.iloc[0]["required_supply"] == 25.0


def test_supplier_constraints():

    service = SupplyIntelligenceService()

    demand = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P002",
                "date": "2025-12-01",
                "demand": 100.0,
            }
        ]
    )

    inventory = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "date": "2025-12-01",
                "product_id": "P002",
                "closing_stock": 0.0,
                "received_stock": 0.0,
            }
        ]
    )

    result = service.analyze(
        demand=demand,
        inventory=inventory,
    )

    quantity = result.iloc[0][
        "recommended_order_quantity"
    ]

    # Chicken supplier contract:
    # MOQ = 5, multiple = 5
    assert quantity % 5 == 0


def test_validation_rejects_truth_columns():

    service = SupplyIntelligenceService()

    result = pd.DataFrame(
        {
            "outlet_id": ["O001"],
            "date": ["2025-12-01"],
            "supply_product_id": ["P002"],
            "required_supply": [10.0],
            "current_inventory": [5.0],
            "incoming_stock": [0.0],
            "lead_time_demand": [30.0],
            "safety_stock": [10.0],
            "target_inventory": [40.0],
            "inventory_position": [5.0],
            "shortage": [35.0],
            "recommended_order_quantity": [35.0],
            "supply_risk_ratio": [0.125],
            "supply_risk": ["high"],
            "order_required": [True],
            "true_demand": [10.0],
        }
    )

    validation = service.validate(result)

    assert validation["passed"] is False
    assert any(
        "truth columns" in error
        for error in validation["errors"]
    )


def test_zero_demand_requires_no_order():

    service = SupplyIntelligenceService()

    demand = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P002",
                "date": "2025-12-01",
                "demand": 0.0,
            }
        ]
    )

    inventory = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "date": "2025-12-01",
                "product_id": "P002",
                "closing_stock": 100.0,
                "received_stock": 0.0,
            }
        ]
    )

    result = service.analyze(
        demand=demand,
        inventory=inventory,
    )

    assert (
        result.iloc[0]["recommended_order_quantity"]
        == 0
    )