
from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.multihorizon.optimized_forecast import (
    MODEL_FEATURES,
    build_batch_feature_matrix,
)
from app.forecasting.xgboost_service import XGBoostForecastService


class OptimizedMultiHorizonForecastService:

    SUPPORTED_HORIZONS = (1, 3, 7)

    def __init__(self):
        self.model_service = XGBoostForecastService()
        self.trained = False

    def train(
        self,
        sales: pd.DataFrame,
        cutoff_date,
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
                "Training dataset is empty."
            )

        from app.forecasting.xgboost_features import (
            prepare_xgboost_dataset,
        )

        train_dataset = prepare_xgboost_dataset(
            train_sales
        )

        self.model_service.train(
            train_dataset
        )

        if (
            self.model_service.model.feature_columns
            != MODEL_FEATURES
        ):
            raise RuntimeError(
                "Trained model feature contract does not "
                "match optimized 51-feature contract."
            )

        self.trained = True

    def forecast(
        self,
        history: pd.DataFrame,
        start_date,
        horizon: int,
    ) -> pd.DataFrame:

        if not self.trained:
            raise RuntimeError(
                "Model must be trained before forecasting."
            )

        if horizon not in self.SUPPORTED_HORIZONS:
            raise ValueError(
                f"Unsupported horizon: {horizon}. "
                f"Supported horizons: {self.SUPPORTED_HORIZONS}"
            )

        if history.empty:
            raise ValueError(
                "History dataset is empty."
            )

        required_columns = {
            "outlet_id",
            "product_id",
            "date",
            "quantity_sold",
        }

        missing_columns = (
            required_columns
            - set(history.columns)
        )

        if missing_columns:
            raise ValueError(
                "History is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        data = history.copy()

        data["date"] = pd.to_datetime(
            data["date"]
        )

        # ---------------------------------------------------------
        # SORT ONCE
        # ---------------------------------------------------------

        ordered = data.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        ).reset_index(drop=True)

        # ---------------------------------------------------------
        # IDENTIFY SERIES
        # ---------------------------------------------------------

        pairs = (
            ordered[
                [
                    "outlet_id",
                    "product_id",
                ]
            ]
            .drop_duplicates(
                ignore_index=True
            )
        )

        n_series = len(pairs)

        if n_series == 0:
            return pd.DataFrame(
                columns=[
                    "outlet_id",
                    "product_id",
                    "date",
                    "horizon",
                    "prediction",
                ]
            )

        # ---------------------------------------------------------
        # RECTANGULAR DATA CHECK
        # ---------------------------------------------------------

        n_rows = len(ordered)

        if n_rows % n_series != 0:
            raise RuntimeError(
                "Historical data is not rectangular. "
                "Every outlet/product series must contain "
                "the same number of observations for the "
                "current optimized forecasting engine."
            )

        historical_days = (
            n_rows // n_series
        )

        if historical_days < 28:
            raise RuntimeError(
                "At least 28 historical observations are "
                "required for the current feature contract."
            )

        # ---------------------------------------------------------
        # DIRECT NUMPY STATE MATRIX
        #
        # Instead of filtering the 871,000-row DataFrame for
        # every series, reshape the already sorted target data.
        #
        # Result:
        #
        #     n_series × historical_days
        #
        # Only the last 28 observations are retained because
        # lag_28 and rolling_28 are the largest historical
        # requirements of the current model.
        # ---------------------------------------------------------

        values = (
            ordered[
                "quantity_sold"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        state_matrix = values.reshape(
            n_series,
            historical_days,
        )

        state_matrix = state_matrix[
            :,
            -28:,
        ].copy()

        metadata = list(
            pairs.itertuples(
                index=False,
                name=None,
            )
        )

        # ---------------------------------------------------------
        # MODEL CONTRACT
        # ---------------------------------------------------------

        model_features = (
            self.model_service
            .model
            .feature_columns
        )

        if model_features != MODEL_FEATURES:
            raise RuntimeError(
                "Model feature columns do not match "
                "optimized 51-feature contract."
            )

        # ---------------------------------------------------------
        # FORECAST
        # ---------------------------------------------------------

        all_results = []

        start_date = pd.Timestamp(
            start_date
        )

        for step in range(
            1,
            horizon + 1,
        ):

            forecast_date = (
                start_date
                + pd.Timedelta(
                    days=step - 1
                )
            )

            # Convert the state matrix into the representation
            # expected by the existing vectorized feature builder.
            states = [
                state_matrix[index]
                for index in range(n_series)
            ]

            X = build_batch_feature_matrix(
                states=states,
                forecast_date=forecast_date,
            )

            # -----------------------------------------------------
            # FEATURE CONTRACT CHECK
            # -----------------------------------------------------

            if list(X.columns) != MODEL_FEATURES:
                raise RuntimeError(
                    "Optimized feature matrix does not "
                    "match the 51-feature contract."
                )

            # -----------------------------------------------------
            # BATCH XGBOOST PREDICTION
            # -----------------------------------------------------

            predictions = (
                self.model_service
                .model
                .predict(X)
            )

            predictions = np.asarray(
                predictions,
                dtype=np.float64,
            )

            predictions = np.maximum(
                predictions,
                0.0,
            )

            if len(predictions) != n_series:
                raise RuntimeError(
                    "Prediction count does not match "
                    "number of outlet/product series."
                )

            if not np.isfinite(
                predictions
            ).all():
                raise RuntimeError(
                    "NaN or infinite predictions detected."
                )

            # -----------------------------------------------------
            # RESULT ROWS
            # -----------------------------------------------------

            for index, prediction in enumerate(
                predictions
            ):

                outlet_id, product_id = (
                    metadata[index]
                )

                all_results.append(
                    {
                        "outlet_id": outlet_id,
                        "product_id": product_id,
                        "date": forecast_date,
                        "horizon": step,
                        "prediction": float(
                            prediction
                        ),
                    }
                )

            # -----------------------------------------------------
            # RECURSIVE UPDATE
            #
            # Today's predictions become the history used for
            # tomorrow.
            #
            # Original sales DataFrame is NEVER modified.
            # -----------------------------------------------------

            state_matrix[:, :-1] = (
                state_matrix[:, 1:]
            )

            state_matrix[:, -1] = (
                predictions
            )

        # ---------------------------------------------------------
        # FINAL DATAFRAME
        #
        # THIS RETURN WAS MISSING IN THE PREVIOUS VERSION.
        # ---------------------------------------------------------

        return pd.DataFrame(
            all_results,
            columns=[
                "outlet_id",
                "product_id",
                "date",
                "horizon",
                "prediction",
            ],
        )

