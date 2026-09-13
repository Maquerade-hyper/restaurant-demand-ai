from __future__ import annotations

import pandas as pd

from app.forecasting.multihorizon.forecast_features import (
    build_forecast_features,
    append_forecast_to_history,
)
from app.forecasting.xgboost_features import (
    prepare_xgboost_dataset,
)
from app.forecasting.xgboost_service import (
    XGBoostForecastService,
)


class MultiHorizonForecastService:

    SUPPORTED_HORIZONS = (1, 3, 7)

    def __init__(self):
        self.model_service = XGBoostForecastService()
        self.trained = False

    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        sales: pd.DataFrame,
        cutoff_date: pd.Timestamp | str,
    ) -> None:

        dataset = sales.copy()

        dataset["date"] = pd.to_datetime(
            dataset["date"]
        )

        cutoff_date = pd.Timestamp(
            cutoff_date
        )

        train_sales = dataset[
            dataset["date"] < cutoff_date
        ].copy()

        if train_sales.empty:
            raise ValueError(
                "No training data exists before cutoff date."
            )

        # IMPORTANT:
        # Use the exact same feature preparation used
        # by the normal XGBoost training pipeline.
        train_dataset = prepare_xgboost_dataset(
            train_sales
        )

        self.model_service.train(
            train_dataset
        )

        if not self.model_service.model.feature_columns:
            raise RuntimeError(
                "XGBoost training produced no feature columns."
            )

        self.trained = True

    # =========================================================
    # FORECAST ONE OUTLET / PRODUCT
    # =========================================================

    def forecast_series(
        self,
        history: pd.DataFrame,
        outlet_id: str,
        product_id: str,
        start_date: pd.Timestamp | str,
        horizon: int,
    ) -> pd.DataFrame:

        if not self.trained:
            raise RuntimeError(
                "Model must be trained before forecasting."
            )

        if horizon <= 0:
            raise ValueError(
                "Horizon must be greater than zero."
            )

        if horizon > 7:
            raise ValueError(
                "Part 15 currently supports maximum D+7."
            )

        start_date = pd.Timestamp(
            start_date
        )

        working_history = history.copy()

        working_history["date"] = pd.to_datetime(
            working_history["date"]
        )

        working_history = working_history[
            (
                working_history["outlet_id"]
                == outlet_id
            )
            &
            (
                working_history["product_id"]
                == product_id
            )
        ].copy()

        if working_history.empty:
            raise ValueError(
                "No history found for outlet/product."
            )

        working_history = (
            working_history
            .sort_values("date")
            .reset_index(drop=True)
        )

        results = []

        for step in range(1, horizon + 1):

            forecast_date = (
                start_date
                + pd.Timedelta(days=step - 1)
            )

            # -------------------------------------------------
            # Future row
            # -------------------------------------------------

            future_row = pd.DataFrame(
                [
                    {
                        "outlet_id": outlet_id,
                        "product_id": product_id,
                        "date": forecast_date,
                        "unit": (
                            working_history[
                                "unit"
                            ].iloc[-1]
                            if "unit"
                            in working_history.columns
                            else None
                        ),
                        "revenue": float("nan"),
                    }
                ]
            )

            # -------------------------------------------------
            # Build features using only:
            #
            # actual observations
            # +
            # previous predictions
            #
            # NEVER the target of forecast_date.
            # -------------------------------------------------

            features = build_forecast_features(
                history=working_history,
                future_row=future_row,
            )

            model_features = (
                self.model_service
                .model
                .feature_columns
            )

            missing_features = [
                column
                for column in model_features
                if column not in features.columns
            ]

            if missing_features:
                raise RuntimeError(
                    "Forecast feature mismatch. "
                    f"Missing: {missing_features}"
                )

            X = features[
                model_features
            ].copy()

            # -------------------------------------------------
            # Check for invalid model input
            # -------------------------------------------------

            if X.isnull().all(axis=1).any():
                raise RuntimeError(
                    "All forecast features are null."
                )

            # XGBoost cannot safely receive object/string
            # feature columns.
            non_numeric = X.select_dtypes(
                exclude="number"
            ).columns.tolist()

            if non_numeric:
                raise RuntimeError(
                    "Non-numeric forecast features: "
                    f"{non_numeric}"
                )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            prediction = float(
                self.model_service
                .model
                .predict(X)[0]
            )

            prediction = max(
                prediction,
                0.0,
            )

            # -------------------------------------------------
            # Store result
            # -------------------------------------------------

            results.append(
                {
                    "outlet_id": outlet_id,
                    "product_id": product_id,
                    "date": forecast_date,
                    "horizon": step,
                    "prediction": prediction,
                }
            )

            # -------------------------------------------------
            # Recursive state update
            # -------------------------------------------------

            working_history = (
                append_forecast_to_history(
                    history=working_history,
                    forecast_row=future_row,
                    prediction=prediction,
                )
            )

        return pd.DataFrame(results)

    # =========================================================
    # FORECAST ALL SERIES
    # =========================================================

    def forecast(
        self,
        history: pd.DataFrame,
        start_date: pd.Timestamp | str,
        horizon: int,
    ) -> pd.DataFrame:

        if horizon not in self.SUPPORTED_HORIZONS:
            raise ValueError(
                "Supported horizons are "
                f"{self.SUPPORTED_HORIZONS}"
            )

        pairs = (
            history[
                [
                    "outlet_id",
                    "product_id",
                ]
            ]
            .drop_duplicates()
        )

        forecasts = []

        for _, pair in pairs.iterrows():

            result = self.forecast_series(
                history=history,
                outlet_id=pair["outlet_id"],
                product_id=pair["product_id"],
                start_date=start_date,
                horizon=horizon,
            )

            forecasts.append(result)

        if not forecasts:
            return pd.DataFrame(
                columns=[
                    "outlet_id",
                    "product_id",
                    "date",
                    "horizon",
                    "prediction",
                ]
            )

        return pd.concat(
            forecasts,
            ignore_index=True,
        )