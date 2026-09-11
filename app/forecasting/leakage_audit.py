
from __future__ import annotations

import pandas as pd

from app.data.pipeline import load_synthetic_dataset
from app.forecasting.xgboost_features import (
    prepare_xgboost_dataset,
)
from app.forecasting.xgboost_service import (
    XGBoostForecastService,
)
from app.forecasting.leakage import (
    audit_feature_columns,
    audit_temporal_split,
)
from app.forecasting.feature_audit import (
    audit_lag_features,
    audit_rolling_features,
)
from app.forecasting.forecast_time_audit import (
    audit_forecast_time_features,
)


def main():

    print(
        "\n=============================="
    )
    print(
        "LEAKAGE & FORECAST-TIME AUDIT"
    )
    print(
        "=============================="
    )

    # ==========================================
    # 1. LOAD DATA
    # ==========================================

    sales = load_synthetic_dataset(
        "sales.csv"
    )

    # ==========================================
    # 2. BUILD FEATURES
    # ==========================================

    dataset = prepare_xgboost_dataset(
        sales
    )

    dataset = dataset.sort_values(
        "date"
    ).reset_index(drop=True)

    # ==========================================
    # 3. TEMPORAL TRAIN / TEST SPLIT
    # ==========================================

    cutoff = (
        dataset["date"].max()
        - pd.Timedelta(days=28)
    )

    train = dataset[
        dataset["date"] < cutoff
    ].copy()

    test = dataset[
        dataset["date"] >= cutoff
    ].copy()

    # ==========================================
    # 4. USE THE SAME FEATURE RULES AS XGBOOST
    # ==========================================

    feature_columns = (
        XGBoostForecastService()
        ._feature_columns(dataset)
    )

    # ==========================================
    # 5. FEATURE LEAKAGE AUDIT
    # ==========================================

    feature_result = audit_feature_columns(
        feature_columns
    )

    # ==========================================
    # 6. TEMPORAL SPLIT AUDIT
    # ==========================================

    temporal_result = audit_temporal_split(
        train,
        test,
        date_column="date",
    )

    # ==========================================
    # 7. LAG AUDIT
    # ==========================================

    lag_result = audit_lag_features(
        dataset
    )

    # ==========================================
    # 8. ROLLING AUDIT
    # ==========================================

    rolling_result = audit_rolling_features(
        dataset
    )

    # ==========================================
    # 9. FORECAST-TIME AUDIT
    # ==========================================

    forecast_result = (
        audit_forecast_time_features(
            feature_columns
        )
    )

    # ==========================================
    # 10. COLLECT RESULTS
    # ==========================================

    checks = {
        "FEATURE_COLUMNS": feature_result,
        "TEMPORAL_SPLIT": temporal_result,
        "LAG_FEATURES": lag_result,
        "ROLLING_FEATURES": rolling_result,
        "FORECAST_TIME": forecast_result,
    }

    failed = False

    # ==========================================
    # 11. PRINT RESULTS
    # ==========================================

    for name, result in checks.items():

        print(
            f"\n{name}"
        )

        print(
            "-" * len(name)
        )

        print(
            "PASS:",
            result["passed"],
        )

        if not result["passed"]:

            print(
                "VIOLATIONS:",
                result.get(
                    "violations",
                    [],
                ),
            )

            failed = True

    # ==========================================
    # 12. FINAL STATUS
    # ==========================================

    print(
        "\n=============================="
    )

    if failed:

        print(
            "LEAKAGE AUDIT: FAILED"
        )

        raise SystemExit(1)

    print(
        "LEAKAGE AUDIT: PASSED"
    )


if __name__ == "__main__":
    main()

