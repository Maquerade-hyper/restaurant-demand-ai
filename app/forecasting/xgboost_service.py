from __future__ import annotations

import pandas as pd

from app.models.xgboost_model import XGBoostDemandModel


class XGBoostForecastService:

    EXCLUDED_FEATURES = {
        "quantity_sold",
        "revenue",
        "date",
        "outlet_id",
        "product_id",
        "unit",
    }

    def __init__(self):
        self.model = XGBoostDemandModel()

    def _feature_columns(
        self,
        dataset: pd.DataFrame,
    ) -> list[str]:

        return [
            column
            for column in dataset.columns
            if column not in self.EXCLUDED_FEATURES
        ]

    def train(
        self,
        dataset: pd.DataFrame,
        target_column: str = "quantity_sold",
    ):

        feature_columns = self._feature_columns(
            dataset
        )

        train_df = dataset.dropna(
            subset=feature_columns
            + [target_column]
        )

        X = train_df[feature_columns]
        y = train_df[target_column]

        self.model.fit(
            X,
            y,
        )

        return self.model

    def predict(
        self,
        dataset: pd.DataFrame,
    ):

        X = dataset[
            self.model.feature_columns
        ]

        return self.model.predict(X)