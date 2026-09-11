import pandas as pd


def check_no_nulls(
    df: pd.DataFrame,
    columns: list[str],
) -> list[str]:

    errors = []

    for column in columns:
        if column not in df.columns:
            errors.append(
                f"Missing column: {column}"
            )
            continue

        count = int(df[column].isna().sum())

        if count > 0:
            errors.append(
                f"{column}: {count} null values"
            )

    return errors


def check_unique(
    df: pd.DataFrame,
    column: str,
) -> list[str]:

    if column not in df.columns:
        return [f"Missing column: {column}"]

    duplicates = int(
        df[column].duplicated().sum()
    )

    if duplicates:
        return [
            f"{column}: {duplicates} duplicates"
        ]

    return []


def check_non_negative(
    df: pd.DataFrame,
    columns: list[str],
) -> list[str]:

    errors = []

    for column in columns:
        if column not in df.columns:
            continue

        invalid = int(
            (df[column] < 0).sum()
        )

        if invalid:
            errors.append(
                f"{column}: {invalid} negative values"
            )

    return errors


def check_bounds(
    df: pd.DataFrame,
    column: str,
    minimum: float,
    maximum: float,
) -> list[str]:

    if column not in df.columns:
        return [f"Missing column: {column}"]

    invalid = int(
        (
            (df[column] < minimum)
            | (df[column] > maximum)
        ).sum()
    )

    if invalid:
        return [
            f"{column}: {invalid} values outside "
            f"[{minimum}, {maximum}]"
        ]

    return []