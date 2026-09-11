import pandas as pd


def create_time_features(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:

    result = df.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result["year"] = result[date_column].dt.year
    result["month"] = result[date_column].dt.month
    result["day"] = result[date_column].dt.day
    result["day_of_week"] = result[date_column].dt.dayofweek
    result["week_of_year"] = (
        result[date_column].dt.isocalendar().week.astype(int)
    )
    result["day_of_year"] = result[date_column].dt.dayofyear
    result["quarter"] = result[date_column].dt.quarter
    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    # Cyclic encoding
    result["dow_sin"] = (
        __import__("numpy").sin(
            2 * __import__("numpy").pi
            * result["day_of_week"] / 7
        )
    )

    result["dow_cos"] = (
        __import__("numpy").cos(
            2 * __import__("numpy").pi
            * result["day_of_week"] / 7
        )
    )

    result["month_sin"] = (
        __import__("numpy").sin(
            2 * __import__("numpy").pi
            * result["month"] / 12
        )
    )

    result["month_cos"] = (
        __import__("numpy").cos(
            2 * __import__("numpy").pi
            * result["month"] / 12
        )
    )

    return result