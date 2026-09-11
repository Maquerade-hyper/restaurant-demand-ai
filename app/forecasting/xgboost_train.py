from __future__ import annotations

import pandas as pd

from app.data.pipeline import load_synthetic_dataset
from app.forecasting.xgboost_features import (
    prepare_xgboost_dataset,
)
from app.forecasting.xgboost_service import (
    XGBoostForecastService,
)
from app.forecasting.metrics import evaluate_forecast


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

    cutoff = dataset["date"].max() - pd.Timedelta(
        days=28
    )

    train = dataset[
        dataset["date"] < cutoff
    ]

    test = dataset[
        dataset["date"] >= cutoff
    ]

    service = XGBoostForecastService()

    service.train(train)

    predictions = service.predict(test)

    actual = test[
        "quantity_sold"
    ].to_numpy()

    metrics = evaluate_forecast(
        actual,
        predictions,
    )

    print("\nXGBOOST RESULTS")
    print("================")

    for name, value in metrics.items():
        print(
            f"{name.upper()}: "
            f"{value:.4f}"
        )

    importance = (
        service.model
        .feature_importance()
        .head(15)
    )

    print("\nTOP FEATURES")
    print("============")

    print(
        importance.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()