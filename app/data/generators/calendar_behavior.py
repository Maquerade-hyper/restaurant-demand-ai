from datetime import date


# Generic global calendar effects.
# Country/region-specific calendars will be added later.
HOLIDAY_MULTIPLIERS = {
    "national": 1.20,
    "religious": 1.25,
    "cultural": 1.18,
    "public": 1.15,
    "local": 1.10,
}


RELIGIOUS_PERIOD_MULTIPLIERS = {
    "ramadan": 0.90,
    "eid": 1.45,
    "christmas": 1.35,
    "easter": 1.20,
    "diwali": 1.40,
    "thanksgiving": 1.35,
    "lunar_new_year": 1.30,
}


def holiday_multiplier(
    holiday_type: str | None,
    importance: float = 0.0,
) -> float:
    """
    Converts holiday metadata into a demand multiplier.
    Importance is expected to be in the range 0-1.
    """

    if not holiday_type:
        return 1.0

    base = HOLIDAY_MULTIPLIERS.get(
        holiday_type.lower(),
        1.05,
    )

    importance = max(0.0, min(float(importance), 1.0))

    return 1.0 + (base - 1.0) * importance


def religious_period_multiplier(
    period: str | None,
) -> float:
    if not period:
        return 1.0

    return RELIGIOUS_PERIOD_MULTIPLIERS.get(
        period.lower(),
        1.0,
    )


def month_multiplier(day: date) -> float:
    """
    Broad seasonal effect.
    Avoids hard-coded country assumptions.
    """

    month_effects = {
        1: 0.98,
        2: 0.99,
        3: 1.00,
        4: 1.02,
        5: 1.04,
        6: 1.03,
        7: 1.01,
        8: 1.00,
        9: 1.02,
        10: 1.05,
        11: 1.08,
        12: 1.15,
    }

    return month_effects.get(day.month, 1.0)


def calendar_multiplier(
    day: date,
    holiday_type: str | None = None,
    holiday_importance: float = 0.0,
    religious_period: str | None = None,
) -> float:

    multiplier = month_multiplier(day)

    multiplier *= holiday_multiplier(
        holiday_type,
        holiday_importance,
    )

    multiplier *= religious_period_multiplier(
        religious_period,
    )

    return max(multiplier, 0.1)