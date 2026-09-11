from __future__ import annotations

import pandas as pd


# Columns that must never be used as prediction features.
FORBIDDEN_FEATURES = {
    "quantity_sold",
    "future_quantity_sold",
    "future_demand",
    "prediction",
    "actual",
}


def find_forbidden_features(
    feature_columns: list[str],
) -> list[str]:

    violations = []

    for column in feature_columns:

        if column.lower() in FORBIDDEN_FEATURES:
            violations.append(column)

    return violations


def audit_feature_columns(
    feature_columns: list[str],
) -> dict:

    violations = find_forbidden_features(
        feature_columns
    )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
    }

def audit_temporal_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    date_column: str = "date",
) -> dict:

    train_dates = pd.to_datetime(
        train[date_column]
    )

    test_dates = pd.to_datetime(
        test[date_column]
    )

    train_max = train_dates.max()
    test_min = test_dates.min()

    passed = train_max < test_min

    return {
        "passed": bool(passed),
        "train_min": str(train_dates.min()),
        "train_max": str(train_max),
        "test_min": str(test_min),
        "test_max": str(test_dates.max()),
    }