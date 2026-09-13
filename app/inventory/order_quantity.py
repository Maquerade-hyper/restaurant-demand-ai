from __future__ import annotations

import math


def calculate_order_quantity(
    reorder_point: float,
    inventory_position: float,
    forecast_demand: float = 0.0,
    review_period_days: int = 0,
) -> float:

    if reorder_point < 0:
        raise ValueError(
            "reorder_point cannot be negative."
        )

    if inventory_position < 0:
        inventory_position = 0.0

    if forecast_demand < 0:
        raise ValueError(
            "forecast_demand cannot be negative."
        )

    if review_period_days < 0:
        raise ValueError(
            "review_period_days cannot be negative."
        )

    review_demand = (
        forecast_demand
        * review_period_days
    )

    target_inventory = (
        reorder_point
        + review_demand
    )

    order_quantity = (
        target_inventory
        - inventory_position
    )

    return float(
        max(order_quantity, 0.0)
    )


def round_to_pack_size(
    quantity: float,
    pack_size: float,
) -> float:

    if quantity < 0:
        raise ValueError(
            "quantity cannot be negative."
        )

    if pack_size <= 0:
        raise ValueError(
            "pack_size must be greater than zero."
        )

    return float(
        math.ceil(
            quantity / pack_size
        )
        * pack_size
    )