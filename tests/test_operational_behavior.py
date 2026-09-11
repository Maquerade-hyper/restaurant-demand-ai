import random

from app.data.generators.operational_behavior import (
    calculate_closing_stock,
    calculate_observed_sales,
    calculate_wastage,
    inventory_position,
    reorder_point,
    safety_stock,
    supplier_lead_time,
)


def test_stockout_limits_observed_sales():

    result = calculate_observed_sales(
        true_demand=100,
        available_stock=60,
    )

    assert result["observed_sales"] == 60
    assert result["lost_demand"] == 40
    assert result["stockout"] is True


def test_no_stockout():

    result = calculate_observed_sales(
        true_demand=50,
        available_stock=100,
    )

    assert result["observed_sales"] == 50
    assert result["lost_demand"] == 0
    assert result["stockout"] is False


def test_closing_stock():

    closing = calculate_closing_stock(
        opening_stock=100,
        received_stock=50,
        observed_sales=80,
        wastage=5,
    )

    assert closing == 65


def test_wastage_non_negative():

    value = calculate_wastage(
        available_stock=100,
        observed_sales=50,
        shelf_life_days=3,
        rng=random.Random(42),
    )

    assert 0 <= value <= 50


def test_supplier_lead_time():

    value = supplier_lead_time(
        supplier_type="local",
        rng=random.Random(42),
    )

    assert 1 <= value <= 2


def test_reorder_point():

    value = reorder_point(
        average_daily_demand=100,
        lead_time_days=3,
        safety_stock_days=2,
    )

    assert value == 500


def test_safety_stock():

    value = safety_stock(
        average_daily_demand=100,
        demand_std=10,
        lead_time_days=4,
    )

    assert value > 0


def test_inventory_position():

    value = inventory_position(
        on_hand=100,
        on_order=50,
        backorders=20,
    )

    assert value == 130