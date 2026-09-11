from __future__ import annotations

import pandas as pd

from app.models.xgboost_model import XGBoostDemandModel


class XGBoostForecastService:

    def __init__(self):
        self.model = XGBoostDemandModel()

    def train(
        self,
        dataset: pd.DataFrame,
        target_column: str = "quantity_sold",
    ):

        excluded = {
            target_column,
            "date",
            "outlet_id",
            "product_id",
            "unit",
        }

        feature_columns = [
            column
            for column in dataset.columns
            if column not in excluded
        ]

        train_df = dataset.dropna(
            subset=feature_columns
            + [target_column]
        )

        X = train_df[feature_columns]
        y = train_df[target_column]

        self.model.fit(X, y)

        return self.model

    def predict(
        self,
        dataset: pd.DataFrame,
    ):

        excluded = {
            "quantity_sold",
            "date",
            "outlet_id",
            "product_id",
            "unit",
        }

        feature_columns = [
            column
            for column in dataset.columns
            if column not in excluded
        ]

        X = dataset[
            self.model.feature_columns
        ]

        return self.model.predict(X)