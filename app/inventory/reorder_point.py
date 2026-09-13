from __future__ import annotations


def calculate_reorder_point(
    lead_time_demand: float,
    safety_stock: float,
) -> float:

    if lead_time_demand < 0:
        raise ValueError(
            "lead_time_demand cannot be negative."
        )

    if safety_stock < 0:
        raise ValueError(
            "safety_stock cannot be negative."
        )

    return float(
        lead_time_demand
        + safety_stock
    )


def should_reorder(
    inventory_position: float,
    reorder_point: float,
) -> bool:

    return (
        inventory_position
        <= reorder_point
    )