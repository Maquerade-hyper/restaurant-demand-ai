from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "outlet_id",
    "date",
}


def validate_calendar(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        errors.append(
            f"calendar missing columns: {sorted(missing)}"
        )

    if df.empty:
        errors.append("calendar is empty")

    return errors


def prepare_calendar(
    df: pd.DataFrame,
) -> pd.DataFrame:

    errors = validate_calendar(df)

    if errors:
        raise ValueError("; ".join(errors))

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    if result["date"].isna().any():
        raise ValueError(
            "calendar contains invalid dates"
        )

    # ---------------------------------------------------------
    # Flexible column aliases.
    # This allows customer/API datasets to use different
    # naming conventions.
    # ---------------------------------------------------------

    aliases = {
        "holiday_importance": [
            "holiday_importance",
            "holiday_score",
            "holiday_strength",
        ],
        "religious_importance": [
            "religious_importance",
            "religious_score",
            "religious_strength",
        ],
        "event_importance": [
            "event_importance",
            "event_score",
            "event_strength",
        ],
    }

    for canonical, candidates in aliases.items():

        if canonical in result.columns:
            continue

        for candidate in candidates:
            if candidate in result.columns:
                result[canonical] = result[candidate]
                break

    for column in [
        "holiday_importance",
        "religious_importance",
        "event_importance",
    ]:

        if column not in result.columns:
            result[column] = 0.0

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if result[column].isna().any():
            raise ValueError(
                f"{column} contains invalid values"
            )

        if (
            (result[column] < 0.0)
            | (result[column] > 1.0)
        ).any():

            raise ValueError(
                f"{column} must be within [0, 1]"
            )

    result["holiday_active"] = (
        result["holiday_importance"] > 0
    ).astype(int)

    result["religious_period_active"] = (
        result["religious_importance"] > 0
    ).astype(int)

    result["event_active"] = (
        result["event_importance"] > 0
    ).astype(int)

    return result