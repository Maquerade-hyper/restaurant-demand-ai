from datetime import date

from app.data.generators.calendar_behavior import (
    calendar_multiplier,
    holiday_multiplier,
    religious_period_multiplier,
)


def test_normal_day():
    value = calendar_multiplier(date(2025, 3, 15))

    assert value > 0


def test_holiday_increases_demand():
    normal = holiday_multiplier(None)
    holiday = holiday_multiplier("national", 1.0)

    assert holiday > normal


def test_religious_period_effect():
    normal = religious_period_multiplier(None)
    eid = religious_period_multiplier("eid")

    assert eid > normal


def test_calendar_multiplier_positive():
    value = calendar_multiplier(
        date(2025, 12, 25),
        holiday_type="religious",
        holiday_importance=1.0,
        religious_period="christmas",
    )

    assert value > 1.0