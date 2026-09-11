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

    sales = load_synthetic_dataset(
        "sales.csv"
    )

    dataset = prepare_xgboost_dataset(
        sales
    )

    dataset = dataset.sort_values(
        "date"
    )

    cutoff = (
        dataset["date"].max()
        - pd.Timedelta(days=28)
    )

    train = dataset[
        dataset["date"] < cutoff
    ]

    test = dataset[
        dataset["date"] >= cutoff
    ]

    service = XGBoostForecastService()

    service.train(train)

    predictions = service.predict(
        test
    )

    diagnostic_df = test.copy()

    diagnostic_df["prediction"] = (
        predictions
    )

    report = create_diagnostic_report(
        diagnostic_df
    )

    print("\n==============================")
    print("XGBOOST DIAGNOSTIC REPORT")
    print("==============================")

    print("\nOVERALL")
    print("--------")

    for key, value in report[
        "overall"
    ].items():

        print(
            f"{key.upper()}: "
            f"{value:.4f}"
        )

    print("\nBIAS")
    print("----")

    for key, value in report[
        "bias"
    ].items():

        print(
            f"{key}: "
            f"{value:.4f}"
        )

    print("\nHIGH DEMAND")
    print("-----------")

    for key, value in report[
        "high_demand"
    ].items():

        if isinstance(value, float):
            print(
                f"{key}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print(
        "\nSPIKE RECALL: "
        f"{report['spike_recall']:.4f}"
    )

    print("\nWORST OUTLETS")
    print("-------------")

    print(
        report[
            "outlet_diagnostics"
        ].head(10).to_string(index=False)
    )

    print("\nWORST PRODUCTS")
    print("--------------")

    print(
        report[
            "product_diagnostics"
        ].head(10).to_string(index=False)
    )

    print("\nTOP FEATURES")
    print("------------")

    print(
        service.model
        .feature_importance()
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()