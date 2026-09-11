from app.data.generators.external_behavior import (
    event_multiplier,
    external_multiplier,
    promotion_multiplier,
    weather_multiplier,
)


def test_weather_multiplier_positive():
    value = weather_multiplier(
        temperature=25,
        precipitation=2,
        humidity=50,
        weather_condition="clear",
    )

    assert value > 0


def test_promotion_increases_demand():
    normal = promotion_multiplier()
    promoted = promotion_multiplier(
        discount=20,
        promotion_type="discount",
    )

    assert promoted > normal


def test_near_event_stronger_than_far_event():
    near = event_multiplier(
        importance=1.0,
        distance_km=1,
    )

    far = event_multiplier(
        importance=1.0,
        distance_km=50,
    )

    assert near > far


def test_external_multiplier():
    value = external_multiplier(
        temperature=24,
        precipitation=1,
        humidity=50,
        weather_condition="clear",
        tourism_index=0.8,
        business_index=0.7,
        student_index=0.5,
        population_density=10000,
        discount=15,
        promotion_type="discount",
        event_importance=0.8,
        event_distance_km=2,
    )

    assert value > 1.0