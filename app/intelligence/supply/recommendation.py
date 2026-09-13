from __future__ import annotations

import numpy as np
import pandas as pd

from .supplier import apply_supplier_constraints


def calculate_supply_recommendations(
    supply_demand: pd.DataFrame,
    inventory: pd.DataFrame,
    supplier_contract: pd.DataFrame,
    safety_stock_days: float = 1.0,
) -> pd.DataFrame:

    required_supply_columns = {
        "outlet_id",
        "date",
        "supply_product_id",
        "required_supply",
    }

    missing = (
        required_supply_columns
        - set(supply_demand.columns)
    )

    if missing:
        raise ValueError(
            f"Supply demand missing columns: {sorted(missing)}"
        )

    required_inventory_columns = {
        "outlet_id",
        "date",
        "product_id",
        "closing_stock",
        "received_stock",
    }

    missing = (
        required_inventory_columns
        - set(inventory.columns)
    )

    if missing:
        raise ValueError(
            f"Inventory missing columns: {sorted(missing)}"
        )

    inv = inventory.copy()

    inv = inv.rename(
        columns={
            "product_id": "supply_product_id",
            "closing_stock": "current_inventory",
            "received_stock": "incoming_stock",
        }
    )

    inv["date"] = pd.to_datetime(inv["date"])

    daily = (
        supply_demand
        .groupby(
            [
                "outlet_id",
                "date",
                "supply_product_id",
            ],
            as_index=False,
        )
        .agg(
            required_supply=(
                "required_supply",
                "sum",
            )
        )
    )

    daily["supply_product_id"] = (
        daily["supply_product_id"].astype(str)
    )

    inv["supply_product_id"] = (
        inv["supply_product_id"].astype(str)
    )

    daily = daily.merge(
        inv[
            [
                "outlet_id",
                "date",
                "supply_product_id",
                "current_inventory",
                "incoming_stock",
            ]
        ],
        on=[
            "outlet_id",
            "date",
            "supply_product_id",
        ],
        how="left",
    )

    daily["current_inventory"] = (
        daily["current_inventory"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    daily["incoming_stock"] = (
        daily["incoming_stock"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    daily = daily.merge(
        supplier_contract,
        on="supply_product_id",
        how="left",
    )

    daily["lead_time_days"] = (
        daily["lead_time_days"]
        .fillna(3)
        .astype(float)
    )

    daily["minimum_order_quantity"] = (
        daily["minimum_order_quantity"]
        .fillna(0.0)
        .astype(float)
    )

    daily["order_multiple"] = (
        daily["order_multiple"]
        .fillna(1.0)
        .astype(float)
    )

    daily["lead_time_demand"] = (
        daily["required_supply"]
        * daily["lead_time_days"]
    )

    daily["safety_stock"] = (
        daily["required_supply"]
        * float(safety_stock_days)
    )

    daily["target_inventory"] = (
        daily["lead_time_demand"]
        + daily["safety_stock"]
    )

    daily["inventory_position"] = (
        daily["current_inventory"]
        + daily["incoming_stock"]
    )

    daily["shortage"] = (
        daily["target_inventory"]
        - daily["inventory_position"]
    ).clip(lower=0.0)

    daily["recommended_order_quantity"] = [
        apply_supplier_constraints(
            quantity=shortage,
            minimum_order_quantity=moq,
            order_multiple=multiple,
        )
        for shortage, moq, multiple in zip(
            daily["shortage"],
            daily["minimum_order_quantity"],
            daily["order_multiple"],
        )
    ]

    daily["supply_risk_ratio"] = np.where(
        daily["target_inventory"] > 0,
        (
            daily["inventory_position"]
            / daily["target_inventory"]
        ).clip(0.0, 1.0),
        1.0,
    )

    daily["supply_risk"] = np.select(
        [
            daily["supply_risk_ratio"] < 0.50,
            daily["supply_risk_ratio"] < 0.80,
        ],
        [
            "high",
            "moderate",
        ],
        default="low",
    )

    daily["order_required"] = (
        daily["recommended_order_quantity"] > 0
    )

    return daily