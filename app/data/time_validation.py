import pandas as pd


def validate_dates(
    df: pd.DataFrame,
    date_column: str = "date",
) -> list[str]:

    errors = []

    if date_column not in df.columns:
        return [f"Missing date column: {date_column}"]

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    invalid = int(dates.isna().sum())

    if invalid:
        errors.append(
            f"{date_column}: {invalid} invalid dates"
        )

    if not dates.empty and dates.is_monotonic_decreasing:
        errors.append(
            f"{date_column}: dates are descending"
        )

    return errors


def validate_date_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    date_column: str = "date",
) -> list[str]:

    if date_column not in df.columns:
        return [f"Missing date column: {date_column}"]

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    invalid = int(
        ((dates < start) | (dates > end)).sum()
    )

    if invalid:
        return [
            f"{date_column}: {invalid} dates outside "
            f"{start_date} to {end_date}"
        ]

    return []