import pandas as pd


def add_date_features(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:

    result = df.copy()

    if date_column not in result.columns:
        return result

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result["year"] = result[date_column].dt.year
    result["month"] = result[date_column].dt.month
    result["day"] = result[date_column].dt.day
    result["day_of_week"] = (
        result[date_column].dt.dayofweek
    )
    result["week_of_year"] = (
        result[date_column].dt.isocalendar().week
    )
    result["is_weekend"] = (
        result[date_column].dt.dayofweek >= 5
    )

    return result