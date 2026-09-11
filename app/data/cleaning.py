import pandas as pd


def normalize_nulls(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result = result.replace(
        ["", "NA", "N/A", "null", "None"],
        pd.NA,
    )

    return result


def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result.columns = [
        column.strip().lower()
        for column in result.columns
    ]

    return result


def clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = normalize_columns(df)
    result = normalize_nulls(result)

    return result