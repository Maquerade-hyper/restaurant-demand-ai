
from __future__ import annotations

import pandas as pd

from app.data.pipeline import load_synthetic_dataset
from app.forecasting.xgboost_features import (
    prepare_xgboost_dataset,
)
from app.forecasting.xgboost_service import (
    XGBoostForecastService,
)
from app.forecasting.diagnostic_report import (
    create_diagnostic_report,
)


def main():

    # ==========================================
    # 1. LOAD SALES DATA
    # ==========================================

    sales = load_synthetic_dataset(
        "sales.csv"
    )

    # ==========================================
    # 2. BUILD LEAKAGE-SAFE FEATURES
    # ==========================================

    dataset = prepare_xgboost_dataset(
        sales
    )

    dataset = dataset.sort_values(
        "date"
    ).reset_index(drop=True)

    # ==========================================
    # 3. CHRONOLOGICAL TRAIN / TEST SPLIT
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
    # 4. TRAIN XGBOOST
    # ==========================================

    service = XGBoostForecastService()

    service.train(
        train
    )

    # ==========================================
    # 5. PREDICT TEST PERIOD
    # ==========================================

    predictions = service.predict(
        test
    )

    # ==========================================
    # 6. CREATE DIAGNOSTIC DATASET
    # ==========================================

    diagnostic_df = test.copy()

    diagnostic_df["prediction"] = (
        predictions
    )

    # ==========================================
    # 7. GENERATE DIAGNOSTIC REPORT
    # ==========================================

    report = create_diagnostic_report(
        diagnostic_df
    )

    # ==========================================
    # 8. PRINT OVERALL RESULTS
    # ==========================================

    print(
        "\n=============================="
    )

    print(
        "XGBOOST DIAGNOSTIC REPORT"
    )

    print(
        "=============================="
    )

    print("\nDATA")
    print("----")

    print(
        f"TRAIN ROWS: {len(train)}"
    )

    print(
        f"TEST ROWS: {len(test)}"
    )

    print(
        f"TRAIN END: "
        f"{train['date'].max()}"
    )

    print(
        f"TEST START: "
        f"{test['date'].min()}"
    )

    # ==========================================
    # 9. OVERALL METRICS
    # ==========================================

    print("\nOVERALL")
    print("--------")

    for key, value in report[
        "overall"
    ].items():

        print(
            f"{key.upper()}: "
            f"{value:.4f}"
        )

    # ==========================================
    # 10. BIAS
    # ==========================================

    print("\nBIAS")
    print("----")

    for key, value in report[
        "bias"
    ].items():

        print(
            f"{key}: "
            f"{value:.4f}"
        )

    # ==========================================
    # 11. HIGH-DEMAND PERFORMANCE
    # ==========================================

    print("\nHIGH DEMAND")
    print("-----------")

    for key, value in report[
        "high_demand"
    ].items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    # ==========================================
    # 12. SPIKE RECALL
    # ==========================================

    print(
        "\nSPIKE RECALL: "
        f"{report['spike_recall']:.4f}"
    )

    # ==========================================
    # 13. WORST OUTLETS
    # ==========================================

    print("\nWORST OUTLETS")
    print("-------------")

    print(
        report[
            "outlet_diagnostics"
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ==========================================
    # 14. WORST PRODUCTS
    # ==========================================

    print("\nWORST PRODUCTS")
    print("--------------")

    print(
        report[
            "product_diagnostics"
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ==========================================
    # 15. TOP FEATURES
    # ==========================================

    print("\nTOP FEATURES")
    print("------------")

    print(
        service.model
        .feature_importance()
        .head(15)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

