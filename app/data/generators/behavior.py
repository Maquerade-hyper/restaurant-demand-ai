import math
import random
from datetime import date
from app.data.generators.calendar_behavior import calendar_multiplier
from app.data.generators.external_behavior import external_multiplier


WEEKDAY_MULTIPLIERS = {
    0: 1.00,  # Monday
    1: 1.02,
    2: 1.03,
    3: 1.05,
    4: 1.12,  # Friday
    5: 1.25,  # Saturday
    6: 1.15,  # Sunday
}


OUTLET_MULTIPLIERS = {
    "restaurant": 1.00,
    "bar": 0.85,
    "restaurant_bar": 1.15,
    "cloud_kitchen": 1.10,
    "delivery_kitchen": 1.05,
}


PRODUCT_MULTIPLIERS = {
    "dairy": 1.00,
    "meat": 1.20,
    "grain": 1.05,
    "oil": 0.90,
    "protein": 1.10,
    "bakery": 1.15,
    "beverage": 1.25,
    "vegetable": 1.00,
    "packaging": 1.10,
}


def weekday_multiplier(day: date) -> float:
    return WEEKDAY_MULTIPLIERS[day.weekday()]


def outlet_multiplier(outlet_type: str) -> float:
    return OUTLET_MULTIPLIERS.get(outlet_type, 1.0)


def product_multiplier(category: str) -> float:
    return PRODUCT_MULTIPLIERS.get(category, 1.0)


def seasonal_multiplier(day: date) -> float:
    """
    Smooth annual seasonality.
    Produces realistic gradual changes rather than random jumps.
    """

    day_of_year = day.timetuple().tm_yday

    return 1.0 + 0.12 * math.sin(
        2 * math.pi * day_of_year / 365.25
    )


def location_multiplier(
    location_type: str,
    tourism_level: float,
    business_area: float,
    residential_area: float,
    student_area: float,
) -> float:

    multiplier = 1.0

    if location_type == "tourist":
        multiplier *= 1.0 + tourism_level * 0.30

    elif location_type == "business":
        multiplier *= 1.0 + business_area * 0.25

    elif location_type == "residential":
        multiplier *= 1.0 + residential_area * 0.15

    elif location_type == "university":
        multiplier *= 1.0 + student_area * 0.20

    elif location_type == "mixed":
        multiplier *= (
            1.0
            + tourism_level * 0.10
            + business_area * 0.10
            + residential_area * 0.10
            + student_area * 0.10
        )

    return multiplier


def demand_multiplier(
    day: date,
    outlet_type: str,
    category: str,
    location_type: str = "mixed",
    tourism_level: float = 0.0,
    business_area: float = 0.0,
    residential_area: float = 0.0,
    student_area: float = 0.0,
    holiday_type: str | None = None,
    holiday_importance: float = 0.0,
    religious_period: str | None = None,
    temperature: float | None = None,
    precipitation: float = 0.0,
    humidity: float = 50.0,
    weather_condition: str | None = None,
    tourism_index: float = 0.0,
    business_index: float = 0.0,
    student_index: float = 0.0,
    population_density: float = 0.0,
    discount: float = 0.0,
    promotion_type: str | None = None,
    event_importance: float = 0.0,
    event_distance_km: float = 100.0,
) -> float:

    multiplier = 1.0

    multiplier *= weekday_multiplier(day)
    multiplier *= outlet_multiplier(outlet_type)
    multiplier *= product_multiplier(category)
    multiplier *= seasonal_multiplier(day)
    multiplier *= calendar_multiplier(
        day=day,
        holiday_type=holiday_type,
        holiday_importance=holiday_importance,
        religious_period=religious_period,
    )

    multiplier *= location_multiplier(
        location_type=location_type,
        tourism_level=tourism_level,
        business_area=business_area,
        residential_area=residential_area,
        student_area=student_area,
    )

    multiplier *= external_multiplier(
        temperature=temperature,
        precipitation=precipitation,
        humidity=humidity,
        weather_condition=weather_condition,
        tourism_index=tourism_index,
        business_index=business_index,
        student_index=student_index,
        population_density=population_density,
        discount=discount,
        promotion_type=promotion_type,
        event_importance=event_importance,
        event_distance_km=event_distance_km,
    )

    return max(multiplier, 0.1)


def random_demand_noise(
    rng: random.Random,
    volatility: float = 0.08,
) -> float:
    """
    Multiplicative noise.
    Keeps demand positive while avoiding unrealistic independent noise.
    """

    return max(
        0.5,
        rng.lognormvariate(
            mean=0.0,
            sigma=volatility,
        ),
    )
