import math
import random


def calculate_observed_sales(
    true_demand: float,
    available_stock: float,
) -> dict:

    true_demand = max(float(true_demand), 0.0)
    available_stock = max(float(available_stock), 0.0)

    observed_sales = min(
        true_demand,
        available_stock,
    )

    lost_demand = max(
        true_demand - observed_sales,
        0.0,
    )

    stockout = (
        true_demand > 0
        and available_stock <= true_demand
    )

    return {
        "true_demand": true_demand,
        "observed_sales": observed_sales,
        "lost_demand": lost_demand,
        "stockout": stockout,
    }


def calculate_closing_stock(
    opening_stock: float,
    received_stock: float,
    observed_sales: float,
    wastage: float = 0.0,
) -> float:

    closing_stock = (
        opening_stock
        + received_stock
        - observed_sales
        - wastage
    )

    return max(closing_stock, 0.0)


def calculate_wastage(
    available_stock: float,
    observed_sales: float,
    shelf_life_days: int | None = None,
    rng: random.Random | None = None,
) -> float:

    rng = rng or random.Random()

    available_stock = max(available_stock, 0.0)
    observed_sales = max(observed_sales, 0.0)

    unused_stock = max(
        available_stock - observed_sales,
        0.0,
    )

    if unused_stock <= 0:
        return 0.0

    # Short shelf-life products have higher wastage risk.
    if shelf_life_days is None:
        wastage_rate = 0.01
    elif shelf_life_days <= 3:
        wastage_rate = 0.05
    elif shelf_life_days <= 7:
        wastage_rate = 0.03
    else:
        wastage_rate = 0.01

    noise = rng.uniform(0.7, 1.3)

    return min(
        unused_stock * wastage_rate * noise,
        unused_stock,
    )


def supplier_lead_time(
    supplier_type: str = "standard",
    rng: random.Random | None = None,
) -> int:

    rng = rng or random.Random()

    ranges = {
        "local": (1, 2),
        "standard": (2, 5),
        "regional": (3, 7),
        "international": (5, 14),
    }

    low, high = ranges.get(
        supplier_type,
        ranges["standard"],
    )

    return rng.randint(low, high)


def reorder_point(
    average_daily_demand: float,
    lead_time_days: int,
    safety_stock_days: float = 2.0,
) -> float:

    average_daily_demand = max(
        average_daily_demand,
        0.0,
    )

    lead_time_days = max(
        lead_time_days,
        0,
    )

    safety_stock_days = max(
        safety_stock_days,
        0.0,
    )

    return average_daily_demand * (
        lead_time_days + safety_stock_days
    )


def safety_stock(
    average_daily_demand: float,
    demand_std: float,
    lead_time_days: int,
    service_level_z: float = 1.65,
) -> float:

    average_daily_demand = max(
        average_daily_demand,
        0.0,
    )

    demand_std = max(
        demand_std,
        0.0,
    )

    lead_time_days = max(
        lead_time_days,
        0,
    )

    return service_level_z * demand_std * math.sqrt(
        lead_time_days
    )


def inventory_position(
    on_hand: float,
    on_order: float,
    backorders: float = 0.0,
) -> float:

    return max(
        on_hand + on_order - backorders,
        0.0,
    )