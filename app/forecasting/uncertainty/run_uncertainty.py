from __future__ import annotations

import pandas as pd

from app.data.pipeline import (
    load_synthetic_dataset,
)

from app.forecasting.xgboost_features import (
    prepare_xgboost_dataset,
)

from app.forecasting.xgboost_service import (
    XGBoostForecastService,
)

from app.forecasting.uncertainty.service import (
    UncertaintyService,
)


def main():

    sales = load_synthetic_dataset(
        "sales.csv"
    )

    dataset = prepare_xgboost_dataset(
        sales
    )

    dataset = (
        dataset
        .sort_values("date")
        .reset_index(drop=True)
    )

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

    model = XGBoostForecastService()

    model.train(train)

    predictions = model.predict(
        test
    )

    actual = test[
        "quantity_sold"
    ].reset_index(drop=True)

    predictions = pd.Series(
        predictions
    ).reset_index(drop=True)

    uncertainty = UncertaintyService()

    profile = uncertainty.fit(
        actual,
        predictions,
    )

    intervals = uncertainty.predict(
        predictions
    )

    reliability = uncertainty.evaluate(
        actual,
        intervals,
    )

    print(
        "\n=============================="
    )
    print(
        "UNCERTAINTY INTELLIGENCE"
    )
    print(
        "=============================="
    )

    print("\nERROR PROFILE")
    print("-------------")

    for key, value in profile.items():
        print(
            f"{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    print("\nRELIABILITY")
    print("-----------")

    for key, value in reliability.items():
        print(
            f"{key}: {value:.4f}"
        )

    print("\nSAMPLE FORECASTS")
    print("----------------")

    print(
        intervals.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()