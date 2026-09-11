from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostDemandModel:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_columns: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "XGBoostDemandModel":

        X_clean = X.copy()

        self.feature_columns = X_clean.columns.tolist()

        self.model.fit(
            X_clean,
            np.asarray(y),
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        X_clean = X[self.feature_columns]

        predictions = self.model.predict(X_clean)

        return np.maximum(predictions, 0.0)

    def feature_importance(self) -> pd.DataFrame:

        importance = self.model.feature_importances_

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": importance,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )