from __future__ import annotations

import pandas as pd

from app.forecasting.metrics import evaluate_forecast


def evaluate_horizons(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    required_actual = {
        "outlet_id",
        "product_id",
        "date",
        "quantity_sold",
    }

    required_predictions = {
        "outlet_id",
        "product_id",
        "date",
        "horizon",
        "prediction",
    }

    missing_actual = (
        required_actual
        - set(actual.columns)
    )

    missing_predictions = (
        required_predictions
        - set(predictions.columns)
    )

    if missing_actual:
        raise ValueError(
            f"Actual data missing: "
            f"{sorted(missing_actual)}"
        )

    if missing_predictions:
        raise ValueError(
            f"Prediction data missing: "
            f"{sorted(missing_predictions)}"
        )

    actual_data = actual.copy()
    prediction_data = predictions.copy()

    actual_data["date"] = pd.to_datetime(
        actual_data["date"]
    )

    prediction_data["date"] = pd.to_datetime(
        prediction_data["date"]
    )

    merged = prediction_data.merge(
        actual_data[
            [
                "outlet_id",
                "product_id",
                "date",
                "quantity_sold",
            ]
        ],
        on=[
            "outlet_id",
            "product_id",
            "date",
        ],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No matching actual/prediction rows."
        )

    results = []

    for horizon, group in (
        merged.groupby("horizon")
    ):

        metrics = evaluate_forecast(
            group["quantity_sold"],
            group["prediction"],
        )

        results.append(
            {
                "horizon": int(horizon),
                "rows": len(group),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "smape": metrics["smape"],
                "bias": metrics["bias"],
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("horizon")
        .reset_index(drop=True)
    )