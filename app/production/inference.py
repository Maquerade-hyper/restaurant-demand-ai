from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pandas as pd

from .model_loader import ProductionModelLoader


class ProductionInferenceEngine:
    """
    CPU-first production inference engine.

    This is the API-facing inference layer for Part 29.

    Required history columns:
        date
        outlet_id
        product_id
        deconstrained_demand

    The feature contract matches the Part 28 candidate model.
    """

    FEATURE_COLUMNS = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
    ]

    REQUIRED_COLUMNS = {
        "date",
        "outlet_id",
        "product_id",
        "deconstrained_demand",
    }

    def __init__(
        self,
        loader: ProductionModelLoader,
    ):
        self.loader = loader

    # =========================================================
    # HISTORY PREPARATION
    # =========================================================

    def _prepare_history(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            history,
            pd.DataFrame,
        ):
            raise TypeError(
                "history must be a pandas DataFrame."
            )

        missing = (
            self.REQUIRED_COLUMNS
            - set(history.columns)
        )

        if missing:
            raise ValueError(
                "Missing required history columns: "
                f"{sorted(missing)}"
            )

        frame = history.copy()

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        )

        if frame["date"].isna().any():
            raise ValueError(
                "History contains invalid dates."
            )

        frame[
            "deconstrained_demand"
        ] = pd.to_numeric(
            frame[
                "deconstrained_demand"
            ],
            errors="coerce",
        )

        if (
            frame[
                "deconstrained_demand"
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                "deconstrained_demand contains "
                "non-numeric or missing values."
            )

        frame = frame.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        ).reset_index(
            drop=True
        )

        return frame

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================

    def build_features(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:

        frame = self._prepare_history(
            history
        )

        group_keys = [
            "outlet_id",
            "product_id",
        ]

        grouped = frame.groupby(
            group_keys,
            sort=False,
        )[
            "deconstrained_demand"
        ]

        # -----------------------------------------------------
        # LAGS
        # -----------------------------------------------------

        for lag in [
            1,
            2,
            3,
            7,
            14,
            28,
        ]:
            frame[
                f"lag_{lag}"
            ] = grouped.shift(lag)

        # -----------------------------------------------------
        # SHIFTED HISTORY
        #
        # IMPORTANT:
        # Rolling features are calculated from t-1 and earlier.
        # This prevents the current target from entering
        # its own feature row.
        # -----------------------------------------------------

        frame[
            "_shifted_demand"
        ] = grouped.shift(1)

        rolling_group = frame.groupby(
            group_keys,
            sort=False,
        )[
            "_shifted_demand"
        ]

        # -----------------------------------------------------
        # ROLLING MEANS
        # -----------------------------------------------------

        for window in [
            3,
            7,
            14,
            28,
        ]:

            frame[
                f"rolling_mean_{window}"
            ] = rolling_group.transform(
                lambda series,
                w=window:
                series.rolling(
                    window=w,
                    min_periods=1,
                ).mean()
            )

        # -----------------------------------------------------
        # ROLLING STANDARD DEVIATIONS
        # -----------------------------------------------------

        for window in [
            7,
            28,
        ]:

            frame[
                f"rolling_std_{window}"
            ] = rolling_group.transform(
                lambda series,
                w=window:
                series.rolling(
                    window=w,
                    min_periods=2,
                ).std()
            )

        # -----------------------------------------------------
        # CALENDAR
        # -----------------------------------------------------

        frame[
            "day_of_week"
        ] = (
            frame["date"]
            .dt.dayofweek
            .astype(float)
        )

        frame[
            "month"
        ] = (
            frame["date"]
            .dt.month
            .astype(float)
        )

        frame[
            "day_of_year"
        ] = (
            frame["date"]
            .dt.dayofyear
            .astype(float)
        )

        frame[
            "is_weekend"
        ] = (
            frame["day_of_week"]
            >= 5
        ).astype(float)

        # -----------------------------------------------------
        # CLEAN INTERNAL COLUMN
        # -----------------------------------------------------

        frame.drop(
            columns=[
                "_shifted_demand"
            ],
            inplace=True,
        )

        return frame

    # =========================================================
    # FEATURE MATRIX
    # =========================================================

    def build_feature_matrix(
        self,
        history: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
    ]:

        features = self.build_features(
            history
        )

        X = features[
            self.FEATURE_COLUMNS
        ].copy()

        valid = X.notna().all(
            axis=1
        )

        X = X.loc[
            valid
        ].astype(float)

        return (
            X,
            valid,
        )

    # =========================================================
    # PREDICTION
    # =========================================================

    def predict_history(
        self,
        history: pd.DataFrame,
    ) -> Tuple[
        np.ndarray,
        float,
    ]:

        model = self.loader.load()

        features = self.build_features(
            history
        )

        X = features[
            self.FEATURE_COLUMNS
        ].copy()

        valid = X.notna().all(
            axis=1
        )

        predictions = np.full(
            len(features),
            np.nan,
            dtype=float,
        )

        start = (
            time.perf_counter()
        )

        if valid.any():

            X_valid = (
                X.loc[
                    valid
                ]
                .astype(float)
            )

            predicted = model.predict(
                X_valid
            )

            predicted = np.asarray(
                predicted,
                dtype=float,
            )

            predictions[
                valid.to_numpy()
            ] = predicted

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # Demand cannot be negative.
        finite = np.isfinite(
            predictions
        )

        predictions[
            finite
        ] = np.maximum(
            predictions[
                finite
            ],
            0.0,
        )

        return (
            predictions,
            float(elapsed_ms),
        )

    # =========================================================
    # LATEST FORECAST
    # =========================================================

    def forecast_latest(
        self,
        history: pd.DataFrame,
    ) -> Tuple[
        float,
        float,
    ]:

        predictions, inference_ms = (
            self.predict_history(
                history
            )
        )

        finite_indices = np.flatnonzero(
            np.isfinite(
                predictions
            )
        )

        if len(
            finite_indices
        ) == 0:

            raise ValueError(
                "No valid production prediction "
                "could be generated. At least 29 "
                "historical observations are normally "
                "required for the current feature contract."
            )

        # Because history is sorted chronologically,
        # the final valid prediction represents the
        # latest forecastable observation.
        latest_index = (
            finite_indices[-1]
        )

        prediction = float(
            predictions[
                latest_index
            ]
        )

        return (
            prediction,
            float(inference_ms),
        )

    # =========================================================
    # FORECAST VALIDATION
    # =========================================================

    def validate_feature_contract(
        self,
    ) -> dict:

        errors = []

        forbidden = {
            "quantity_sold",
            "future_quantity_sold",
            "future_demand",
            "true_demand",
            "lost_demand",
            "actual_demand",
            "prediction",
            "actual",
            "target",
            "revenue",
            "closing_stock",
            "wastage",
            "stockout",
        }

        leakage = (
            forbidden
            & set(
                self.FEATURE_COLUMNS
            )
        )

        if leakage:
            errors.append(
                "forbidden_features:"
                + ",".join(
                    sorted(leakage)
                )
            )

        if len(
            self.FEATURE_COLUMNS
        ) != 16:

            errors.append(
                "unexpected_feature_count"
            )

        return {
            "passed": (
                len(errors) == 0
            ),
            "errors": errors,
            "feature_count": len(
                self.FEATURE_COLUMNS
            ),
            "features": list(
                self.FEATURE_COLUMNS
            ),
        }