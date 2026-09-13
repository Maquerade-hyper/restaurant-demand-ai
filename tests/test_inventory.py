from app.inventory.lead_time import (
    calculate_lead_time_demand,
)

from app.inventory.safety_stock import (
    calculate_safety_stock,
)

from app.inventory.reorder_point import (
    calculate_reorder_point,
    should_reorder,
)

from app.inventory.order_quantity import (
    calculate_order_quantity,
    round_to_pack_size,
)

from app.inventory.service import (
    InventoryService,
)


def test_lead_time_demand():

    result = calculate_lead_time_demand(
        [10, 20, 30, 40],
        lead_time_days=3,
    )

    assert result == 60.0


def test_safety_stock():

    result = calculate_safety_stock(
        demand_std=10,
        lead_time_days=4,
        service_level=0.95,
    )

    assert result > 0


def test_reorder_point():

    result = calculate_reorder_point(
        lead_time_demand=100,
        safety_stock=25,
    )

    assert result == 125.0


def test_should_reorder():

    assert should_reorder(
        inventory_position=100,
        reorder_point=125,
    )

    assert not should_reorder(
        inventory_position=150,
        reorder_point=125,
    )


def test_order_quantity():

    result = calculate_order_quantity(
        reorder_point=125,
        inventory_position=50,
    )

    assert result == 75.0


def test_pack_rounding():

    result = round_to_pack_size(
        quantity=74,
        pack_size=10,
    )

    assert result == 80.0


def test_inventory_service():

    service = InventoryService()

    result = service.recommend(
        product_id="P001",
        outlet_id="O001",
        daily_forecast=[
            30,
            32,
            35,
            31,
        ],
        demand_std=5,
        lead_time_days=3,
        inventory_position=20,
        service_level=0.95,
        review_period_days=1,
        pack_size=5,
    )

    assert result["lead_time_demand"] == 97.0
    assert result["safety_stock"] > 0
    assert result["reorder_point"] > 97
    assert result[
        "recommended_order_quantity"
    ] >= 0