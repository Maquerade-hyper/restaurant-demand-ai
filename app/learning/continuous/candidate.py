from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .schemas import CandidateEvaluation


class CandidateTrainer:
    """
    CPU-first XGBoost candidate trainer.

    Candidate training is isolated from the incumbent champion.

    All candidate decisions are based on a common unseen evaluation
    period.
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

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 150,
    ):
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)

    def build_features(
        self,
        df: pd.DataFrame,
        target: str,
    ) -> pd.DataFrame:

        required = {
            "date",
            "outlet_id",
            "product_id",
            target,
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        frame = df.copy()

        frame["date"] = pd.to_datetime(
            frame["date"]
        )

        frame = frame.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        ).reset_index(drop=True)

        keys = [
            "outlet_id",
            "product_id",
        ]

        grouped = frame.groupby(
            keys,
            sort=False,
        )[target]

        for lag in [
            1,
            2,
            3,
            7,
            14,
            28,
        ]:
            frame[f"lag_{lag}"] = (
                grouped.shift(lag)
            )

        # All rolling features are explicitly shifted first.
        # Therefore the current target is never used.
        shifted = (
            frame.groupby(
                keys,
                sort=False,
            )[target]
            .shift(1)
        )

        frame["_shifted_target"] = shifted

        rolling_group = frame.groupby(
            keys,
            sort=False,
        )["_shifted_target"]

        frame["rolling_mean_3"] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=1,
                ).mean()
            )
        )

        frame["rolling_mean_7"] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(
                    7,
                    min_periods=1,
                ).mean()
            )
        )

        frame["rolling_mean_14"] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(
                    14,
                    min_periods=1,
                ).mean()
            )
        )

        frame["rolling_mean_28"] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(
                    28,
                    min_periods=1,
                ).mean()
            )
        )

        frame["rolling_std_7"] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(
                    7,
                    min_periods=2,
                ).std()
            )
        )

        frame["rolling_std_28"] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(
                    28,
                    min_periods=2,
                ).std()
            )
        )

        frame["day_of_week"] = (
            frame["date"].dt.dayofweek
        )

        frame["month"] = (
            frame["date"].dt.month
        )

        frame["day_of_year"] = (
            frame["date"].dt.dayofyear
        )

        frame["is_weekend"] = (
            frame["day_of_week"] >= 5
        ).astype(int)

        frame.drop(
            columns=["_shifted_target"],
            inplace=True,
        )

        return frame

    def _prepare(
        self,
        frame: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:

        clean = frame.dropna(
            subset=self.FEATURE_COLUMNS
            + ["target"]
        )

        X = clean[
            self.FEATURE_COLUMNS
        ].astype(float)

        y = clean["target"].astype(float)

        return X, y

    def train(
        self,
        train_df: pd.DataFrame,
        target: str,
    ):

        frame = self.build_features(
            train_df,
            target,
        )

        frame = frame.rename(
            columns={
                target: "target"
            }
        )

        X, y = self._prepare(
            frame
        )

        if len(X) < 100:
            raise ValueError(
                "Not enough training rows for candidate model."
            )

        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=1,
        )

        model.fit(
            X,
            y,
        )

        return model

    def predict(
        self,
        model,
        frame: pd.DataFrame,
        target: str,
    ):

        features = self.build_features(
            frame,
            target,
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

        if valid.any():
            predictions[valid] = (
                model.predict(
                    X.loc[valid].astype(
                        float
                    )
                )
            )

        return np.maximum(
            predictions,
            0.0,
        )

    @staticmethod
    def metrics(
        actual,
        prediction,
    ) -> Dict[str, float]:

        actual = np.asarray(
            actual,
            dtype=float,
        )

        prediction = np.asarray(
            prediction,
            dtype=float,
        )

        mask = (
            np.isfinite(actual)
            & np.isfinite(prediction)
        )

        actual = actual[mask]
        prediction = prediction[mask]

        if len(actual) == 0:
            raise ValueError(
                "No valid evaluation rows."
            )

        mae = mean_absolute_error(
            actual,
            prediction,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    prediction,
                )
            )
        )

        bias = float(
            np.mean(
                prediction - actual
            )
        )

        return {
            "mae": float(mae),
            "rmse": rmse,
            "bias": bias,
        }

    def evaluate_candidate(
        self,
        champion_predictions,
        candidate_predictions,
        actual,
        candidate_name: str = (
            "xgboost_candidate"
        ),
        min_improvement_pct: float = 2.0,
        max_bias_worsening: float = 1.0,
    ) -> CandidateEvaluation:

        champion_metrics = self.metrics(
            actual,
            champion_predictions,
        )

        candidate_metrics = self.metrics(
            actual,
            candidate_predictions,
        )

        champion_mae = (
            champion_metrics["mae"]
        )

        candidate_mae = (
            candidate_metrics["mae"]
        )

        improvement_pct = (
            (
                champion_mae
                - candidate_mae
            )
            / max(
                champion_mae,
                1e-12,
            )
            * 100.0
        )

        bias_change = (
            abs(
                candidate_metrics["bias"]
            )
            - abs(
                champion_metrics["bias"]
            )
        )

        stable = bool(
            np.isfinite(
                candidate_mae
            )
            and np.isfinite(
                candidate_metrics[
                    "rmse"
                ]
            )
            and np.isfinite(
                candidate_metrics[
                    "bias"
                ]
            )
        )

        reasons = []

        if (
            improvement_pct
            < min_improvement_pct
        ):
            reasons.append(
                "insufficient_improvement:"
                f"{improvement_pct:.4f}%"
            )

        if (
            bias_change
            > max_bias_worsening
        ):
            reasons.append(
                "bias_worsened:"
                f"{bias_change:.4f}"
            )

        if not stable:
            reasons.append(
                "candidate_not_finite"
            )

        passed = bool(
            stable
            and improvement_pct
            >= min_improvement_pct
            and bias_change
            <= max_bias_worsening
        )

        return CandidateEvaluation(
            model_name=candidate_name,
            champion_mae=champion_mae,
            candidate_mae=candidate_mae,
            champion_rmse=champion_metrics[
                "rmse"
            ],
            candidate_rmse=candidate_metrics[
                "rmse"
            ],
            champion_bias=champion_metrics[
                "bias"
            ],
            candidate_bias=candidate_metrics[
                "bias"
            ],
            improvement_pct=float(
                improvement_pct
            ),
            bias_change=float(
                bias_change
            ),
            candidate_stable=stable,
            passes=passed,
            reasons=reasons,
        )