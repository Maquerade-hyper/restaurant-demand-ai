from __future__ import annotations

import math

import pandas as pd


DEFAULT_SUPPLIER_CONTRACT = pd.DataFrame(
    [
        (
            "P001", "SUP-MILK", 2, 1.0, 1.0,
        ),
        (
            "P002", "SUP-MEAT", 3, 5.0, 5.0,
        ),
        (
            "P003", "SUP-MEAT", 3, 5.0, 5.0,
        ),
        (
            "P004", "SUP-GRAINS", 3, 10.0, 10.0,
        ),
        (
            "P005", "SUP-GRAINS", 3, 10.0, 10.0,
        ),
        (
            "P006", "SUP-DAIRY", 2, 5.0, 5.0,
        ),
        (
            "P007", "SUP-OIL", 4, 5.0, 5.0,
        ),
        (
            "P008", "SUP-DAIRY", 2, 30.0, 30.0,
        ),
        (
            "P009", "SUP-BAKERY", 2, 20.0, 20.0,
        ),
        (
            "P010", "SUP-BAKERY", 2, 20.0, 20.0,
        ),
        (
            "P011", "SUP-BAKERY", 2, 20.0, 20.0,
        ),
        (
            "P012", "SUP-BAKERY", 2, 20.0, 20.0,
        ),
        (
            "P013", "SUP-BAKERY", 2, 20.0, 20.0,
        ),
        (
            "P014", "SUP-BEVERAGE", 3, 24.0, 24.0,
        ),
        (
            "P015", "SUP-BEVERAGE", 3, 24.0, 24.0,
        ),
        (
            "P016", "SUP-BEVERAGE", 3, 20.0, 20.0,
        ),
        (
            "P017", "SUP-BEVERAGE", 3, 20.0, 20.0,
        ),
        (
            "P018", "SUP-PRODUCE", 2, 5.0, 5.0,
        ),
        (
            "P019", "SUP-PRODUCE", 2, 5.0, 5.0,
        ),
        (
            "P020", "SUP-PACKAGING", 5, 50.0, 50.0,
        ),
    ],
    columns=[
        "supply_product_id",
        "supplier_id",
        "lead_time_days",
        "minimum_order_quantity",
        "order_multiple",
    ],
)


def normalize_supplier_contract(
    contract: pd.DataFrame | None = None,
) -> pd.DataFrame:

    result = (
        DEFAULT_SUPPLIER_CONTRACT.copy()
        if contract is None
        else contract.copy()
    )

    required = {
        "supply_product_id",
        "supplier_id",
        "lead_time_days",
        "minimum_order_quantity",
        "order_multiple",
    }

    missing = required - set(result.columns)

    if missing:
        raise ValueError(
            f"Supplier contract missing columns: "
            f"{sorted(missing)}"
        )

    numeric_columns = [
        "lead_time_days",
        "minimum_order_quantity",
        "order_multiple",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    if (result["lead_time_days"] < 0).any():
        raise ValueError("lead_time_days cannot be negative")

    if (result["minimum_order_quantity"] <= 0).any():
        raise ValueError(
            "minimum_order_quantity must be > 0"
        )

    if (result["order_multiple"] <= 0).any():
        raise ValueError("order_multiple must be > 0")

    return result.reset_index(drop=True)


def apply_supplier_constraints(
    quantity: float,
    minimum_order_quantity: float,
    order_multiple: float,
) -> float:

    quantity = max(float(quantity), 0.0)
    minimum_order_quantity = max(
        float(minimum_order_quantity),
        0.0,
    )
    order_multiple = max(
        float(order_multiple),
        1e-9,
    )

    if quantity <= 0:
        return 0.0

    constrained = max(
        quantity,
        minimum_order_quantity,
    )

    return (
        math.ceil(
            constrained / order_multiple
        )
        * order_multiple
    )