import math


def weather_multiplier(
    temperature: float,
    precipitation: float,
    humidity: float,
    weather_condition: str | None = None,
) -> float:
    """
    Models broad weather effects on food-service demand.
    """

    multiplier = 1.0

    # Comfortable temperatures generally support demand.
    temperature_effect = 1.0 - min(
        abs(temperature - 22.0) / 100.0,
        0.15,
    )

    multiplier *= temperature_effect

    # Heavy rain can reduce physical visits.
    rain_penalty = min(precipitation / 50.0, 0.20)
    multiplier *= 1.0 - rain_penalty

    # Humidity has a smaller effect.
    humidity_penalty = max(humidity - 75.0, 0.0) / 500.0
    multiplier *= 1.0 - min(humidity_penalty, 0.08)

    condition_effects = {
        "clear": 1.05,
        "sunny": 1.05,
        "cloudy": 1.00,
        "rain": 0.90,
        "storm": 0.78,
        "snow": 0.82,
        "heatwave": 0.88,
    }

    if weather_condition:
        multiplier *= condition_effects.get(
            weather_condition.lower(),
            1.0,
        )

    return max(multiplier, 0.5)


def tourism_multiplier(tourism_index: float) -> float:
    tourism_index = max(0.0, min(tourism_index, 1.0))
    return 1.0 + tourism_index * 0.25


def demographic_multiplier(
    business_index: float = 0.0,
    student_index: float = 0.0,
    population_density: float = 0.0,
) -> float:

    business_index = max(0.0, min(business_index, 1.0))
    student_index = max(0.0, min(student_index, 1.0))

    # Log scaling prevents extreme population-density effects.
    density_effect = min(
        math.log1p(max(population_density, 0.0)) / 20.0,
        0.20,
    )

    return (
        1.0
        + business_index * 0.12
        + student_index * 0.10
        + density_effect
    )


def promotion_multiplier(
    discount: float = 0.0,
    promotion_type: str | None = None,
) -> float:

    discount = max(0.0, min(discount, 100.0))

    promotion_effects = {
        "discount": 1.20,
        "bundle": 1.15,
        "bogo": 1.30,
        "happy_hour": 1.25,
        "delivery": 1.15,
        "campaign": 1.10,
    }

    base = promotion_effects.get(
        (promotion_type or "").lower(),
        1.05,
    )

    discount_effect = 1.0 + (discount / 100.0) * 0.40

    return base * discount_effect


def event_multiplier(
    importance: float = 0.0,
    distance_km: float = 0.0,
) -> float:

    importance = max(0.0, min(importance, 1.0))
    distance_km = max(distance_km, 0.0)

    # Event influence decays with distance.
    distance_decay = math.exp(-distance_km / 10.0)

    return 1.0 + (
        importance
        * distance_decay
        * 0.35
    )


def external_multiplier(
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

    if temperature is not None:
        multiplier *= weather_multiplier(
            temperature=temperature,
            precipitation=precipitation,
            humidity=humidity,
            weather_condition=weather_condition,
        )

    multiplier *= tourism_multiplier(tourism_index)

    multiplier *= demographic_multiplier(
        business_index=business_index,
        student_index=student_index,
        population_density=population_density,
    )

    if discount > 0 or promotion_type:
        multiplier *= promotion_multiplier(
            discount=discount,
            promotion_type=promotion_type,
        )

    if event_importance > 0:
        multiplier *= event_multiplier(
            importance=event_importance,
            distance_km=event_distance_km,
        )

    return max(multiplier, 0.1)