from datetime import date

from app.data.generators.behavior import (
    demand_multiplier,
    seasonal_multiplier,
    weekday_multiplier,
)


def test_weekday_multiplier():
    monday = date(2025, 1, 6)
    saturday = date(2025, 1, 11)

    assert weekday_multiplier(saturday) > weekday_multiplier(monday)


def test_seasonality_positive():
    value = seasonal_multiplier(date(2025, 6, 1))

    assert value > 0


def test_demand_multiplier_positive():
    value = demand_multiplier(
        day=date(2025, 6, 1),
        outlet_type="restaurant",
        category="meat",
        location_type="tourist",
        tourism_level=0.8,
    )

    assert value > 0