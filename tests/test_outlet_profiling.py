import pandas as pd

from app.intelligence.outlet.profiling import build_outlet_profiles
from app.intelligence.outlet.service import OutletProfilingService


def make_outlets():
    return pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "outlet_type": "restaurant",
                "country": "India",
                "region": "Delhi",
                "city": "Delhi",
                "location_type": "business",
                "cuisine": "Indian",
                "capacity": 100,
                "delivery_available": True,
                "takeaway_available": True,
                "bar_available": False,
                "kitchen_type": "standard",
                "tourism_level": 0.4,
                "business_area": 0.9,
                "residential_area": 0.2,
                "student_area": 0.1,
            },
            {
                "outlet_id": "O002",
                "outlet_type": "cloud_kitchen",
                "country": "India",
                "region": "Mumbai",
                "city": "Mumbai",
                "location_type": "residential",
                "cuisine": "Multi",
                "capacity": 40,
                "delivery_available": True,
                "takeaway_available": False,
                "bar_available": False,
                "kitchen_type": "delivery",
                "tourism_level": 0.3,
                "business_area": 0.2,
                "residential_area": 0.9,
                "student_area": 0.3,
            },
        ]
    )


def make_demand():
    rows = []

    for day in pd.date_range("2025-01-01", periods=30):
        for outlet_id in ["O001", "O002"]:
            for product_id in ["P001", "P002"]:
                base = 10 if outlet_id == "O001" else 6

                stockout = (
                    outlet_id == "O001"
                    and day.day >= 25
                )

                observed = base if not stockout else base * 0.5
                demand = base
                lost = demand - observed

                rows.append(
                    {
                        "outlet_id": outlet_id,
                        "product_id": product_id,
                        "date": day,
                        "quantity_sold": observed,
                        "deconstrained_demand": demand,
                        "estimated_lost_demand": lost,
                        "stockout": stockout,
                    }
                )

    return pd.DataFrame(rows)


def test_build_profiles_returns_one_row_per_outlet():
    result = build_outlet_profiles(
        make_outlets(),
        make_demand(),
    )

    assert len(result) == 2
    assert result["outlet_id"].nunique() == 2


def test_required_profile_metrics_exist():
    result = build_outlet_profiles(
        make_outlets(),
        make_demand(),
    )

    required = {
        "average_daily_demand",
        "total_demand",
        "total_lost_demand",
        "demand_cv",
        "stockout_rate",
        "lost_demand_rate",
        "weekend_weekday_ratio",
        "demand_trend",
        "active_product_count",
        "demand_volume_class",
        "demand_stability_class",
        "weekend_behavior",
        "demand_trend_class",
        "operational_risk_class",
        "outlet_behavior_class",
    }

    assert required.issubset(result.columns)


def test_lost_demand_is_non_negative():
    result = build_outlet_profiles(
        make_outlets(),
        make_demand(),
    )

    assert (result["total_lost_demand"] >= 0).all()
    assert (result["lost_demand_rate"] >= 0).all()


def test_stockout_rate_is_valid():
    result = build_outlet_profiles(
        make_outlets(),
        make_demand(),
    )

    assert (result["stockout_rate"] >= 0).all()
    assert (result["stockout_rate"] <= 1).all()


def test_outlet_static_attributes_are_preserved():
    result = build_outlet_profiles(
        make_outlets(),
        make_demand(),
    )

    row = result.loc[
        result["outlet_id"] == "O001"
    ].iloc[0]

    assert row["outlet_type"] == "restaurant"
    assert row["city"] == "Delhi"
    assert bool(row["delivery_available"]) is True


def test_service_returns_profiles():
    service = OutletProfilingService()

    result = service.analyze(
        outlets=make_outlets(),
        demand=make_demand(),
    )

    assert len(result) == 2


def test_service_returns_top_risk_outlets():
    service = OutletProfilingService()

    profiles = service.analyze(
        outlets=make_outlets(),
        demand=make_demand(),
    )

    result = service.top_risk_outlets(
        profiles,
        limit=1,
    )

    assert len(result) == 1
    assert "operational_risk_class" in result.columns


def test_stockout_rate_is_product_day_rate():
    result = build_outlet_profiles(
        make_outlets(),
        make_demand(),
    )

    assert "stockout_product_day_rate" in result.columns
    assert "stockout_affected_day_rate" in result.columns

    assert (
        result["stockout_product_day_rate"]
        <= 1.0
    ).all()

    assert (
        result["stockout_product_day_rate"]
        >= 0.0
    ).all()