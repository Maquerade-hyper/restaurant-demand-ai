from __future__ import annotations


FUTURE_ONLY_COLUMNS = {
    "quantity_sold",
    "revenue",
    "closing_stock",
    "wastage",
    "stockout",
    "actual_demand",
    "future_demand",
}


def audit_forecast_time_features(
    feature_columns: list[str],
) -> dict:

    violations = []

    for column in feature_columns:

        if column.lower() in FUTURE_ONLY_COLUMNS:
            violations.append(column)

    return {
        "passed": len(violations) == 0,
        "violations": violations,
    }