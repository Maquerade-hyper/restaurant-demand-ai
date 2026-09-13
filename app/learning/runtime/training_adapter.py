from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.xgboost_features import prepare_xgboost_dataset
from app.models.xgboost_model import XGBoostDemandModel
from app.learning.continuous.registry import ModelRegistry


class ProductionTrainingAdapter:
    """
    Production learning adapter for Part B.

    Architecture:

        Client canonical data
                ↓
        Part 7 feature pipeline
                ↓
        Part 9 XGBoost model
                ↓
        Candidate evaluation
                ↓
        Governance / promotion

    IMPORTANT:
    - Does NOT use the old 16-feature trainer.
    - Uses the established production feature pipeline.
    - Uses XGBoostDemandModel.
    - Uses chronological train/validation/evaluation periods.
    - Evaluation predictions retain historical context.
    """

    TARGET_COLUMN = "quantity_sold"

    MIN_TRAIN_ROWS = 1000
    MIN_EVALUATION_ROWS = 100

    MIN_IMPROVEMENT_PCT = 2.0
    MAX_BIAS_WORSENING = 1.0

    HISTORY_DAYS = 28

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        models_root: str = "models",
    ):
        self.registry = registry or ModelRegistry(
            root=models_root
        )

    # ============================================================
    # DATA VALIDATION
    # ============================================================

    @classmethod
    def validate_input(
        cls,
        df: pd.DataFrame,
    ) -> None:

        required = {
            "date",
            "outlet_id",
            "product_id",
            cls.TARGET_COLUMN,
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                "Missing required production training columns: "
                f"{sorted(missing)}"
            )

        if df.empty:
            raise ValueError(
                "Training dataset is empty."
            )

        if df["date"].isna().any():
            raise ValueError(
                "Training dataset contains null dates."
            )

        if df[cls.TARGET_COLUMN].isna().any():
            raise ValueError(
                "Training dataset contains null target values."
            )

    # ============================================================
    # CANONICAL PREPARATION
    # ============================================================

    @classmethod
    def prepare(
        cls,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        cls.validate_input(df)

        frame = df.copy()

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        )

        frame[cls.TARGET_COLUMN] = pd.to_numeric(
            frame[cls.TARGET_COLUMN],
            errors="coerce",
        )

        frame = (
            frame
            .dropna(
                subset=[
                    "date",
                    "outlet_id",
                    "product_id",
                    cls.TARGET_COLUMN,
                ]
            )
            .sort_values(
                [
                    "outlet_id",
                    "product_id",
                    "date",
                ]
            )
            .reset_index(drop=True)
        )

        if frame.empty:
            raise ValueError(
                "No valid rows remain after production-data preparation."
            )

        return frame

    # ============================================================
    # FEATURE BUILDING
    # ============================================================

    @classmethod
    def build_features(
        cls,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        frame = cls.prepare(df)

        featured = prepare_xgboost_dataset(
            frame
        )

        if featured.empty:
            raise ValueError(
                "Production feature pipeline returned no rows."
            )

        return featured

    # ============================================================
    # TEMPORAL SPLITTING
    # ============================================================

    @classmethod
    def chronological_split(
        cls,
        df: pd.DataFrame,
        validation_days: int = 30,
        evaluation_days: int = 30,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:

        frame = cls.prepare(df)

        dates = (
            pd.Series(
                frame["date"].dropna().unique()
            )
            .sort_values()
            .reset_index(drop=True)
        )

        if len(dates) <= (
            validation_days
            + evaluation_days
            + cls.HISTORY_DAYS
        ):
            raise ValueError(
                "Insufficient chronological history for "
                "train/validation/evaluation split."
            )

        evaluation_dates = dates.iloc[
            -evaluation_days:
        ]

        validation_dates = dates.iloc[
            -(evaluation_days + validation_days):
            -evaluation_days
        ]

        evaluation_start = evaluation_dates.iloc[0]
        validation_start = validation_dates.iloc[0]

        train_df = frame[
            frame["date"] < validation_start
        ].copy()

        validation_df = frame[
            (frame["date"] >= validation_start)
            & (frame["date"] < evaluation_start)
        ].copy()

        evaluation_df = frame[
            frame["date"] >= evaluation_start
        ].copy()

        if train_df.empty:
            raise ValueError(
                "Chronological split produced empty training data."
            )

        if validation_df.empty:
            raise ValueError(
                "Chronological split produced empty validation data."
            )

        if evaluation_df.empty:
            raise ValueError(
                "Chronological split produced empty evaluation data."
            )

        return (
            train_df.reset_index(drop=True),
            validation_df.reset_index(drop=True),
            evaluation_df.reset_index(drop=True),
        )

    # ============================================================
    # MODEL TRAINING
    # ============================================================

    @classmethod
    def train_model(
        cls,
        train_df: pd.DataFrame,
    ) -> XGBoostDemandModel:

        if len(train_df) < cls.MIN_TRAIN_ROWS:
            raise ValueError(
                f"Not enough training rows: "
                f"{len(train_df)} < {cls.MIN_TRAIN_ROWS}"
            )

        featured = cls.build_features(
            train_df
        )

        if cls.TARGET_COLUMN not in featured.columns:
            raise ValueError(
                "Production feature pipeline did not retain "
                f"target column '{cls.TARGET_COLUMN}'."
            )

        excluded = {
            "quantity_sold",
            "date",
            "outlet_id",
            "product_id",
            "unit",
            "revenue",
            "actual_demand",
            "true_demand",
            "lost_demand",
            "future_demand",
            "closing_stock",
            "stockout",
            "demand_truth",
            "lost_demand_truth",
            "prediction",
            "target",
        }

        feature_columns = [
            column
            for column in featured.columns
            if column not in excluded
        ]

        if not feature_columns:
            raise ValueError(
                "No production feature columns available."
            )

        X = featured[
            feature_columns
        ].copy()

        y = pd.to_numeric(
            featured[
                cls.TARGET_COLUMN
            ],
            errors="coerce",
        )

        numeric_columns = [
            column
            for column in X.columns
            if pd.api.types.is_numeric_dtype(
                X[column]
            )
        ]

        X = X[
            numeric_columns
        ].copy()

        if X.empty:
            raise ValueError(
                "No numeric production features available."
            )

        valid = (
            X.notna().all(axis=1)
            & np.isfinite(
                X.to_numpy(
                    dtype=float
                )
            ).all(axis=1)
            & y.notna().to_numpy()
            & np.isfinite(
                y.to_numpy(
                    dtype=float
                )
            )
        )

        X = X.loc[valid].astype(float)
        y = y.loc[valid].astype(float)

        if len(X) < cls.MIN_TRAIN_ROWS:
            raise ValueError(
                "Not enough valid feature rows after "
                f"feature construction: {len(X)} "
                f"< {cls.MIN_TRAIN_ROWS}"
            )

        model = XGBoostDemandModel()

        model.fit(
            X,
            y,
        )

        if len(model.feature_columns) != 51:
            raise ValueError(
                "Production model feature contract mismatch. "
                f"Expected 51 Part 9 features, got "
                f"{len(model.feature_columns)}."
            )

        return model

    # ============================================================
    # FEATURE ALIGNMENT WITH HISTORICAL CONTEXT
    # ============================================================

    @classmethod
    def _prediction_frame(
        cls,
        model: XGBoostDemandModel,
        history_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
    ]:

        history = cls.prepare(
            history_df
        )

        prediction = cls.prepare(
            prediction_df
        )

        combined = pd.concat(
            [
                history,
                prediction,
            ],
            ignore_index=True,
        )

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "date",
                    "outlet_id",
                    "product_id",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "outlet_id",
                    "product_id",
                    "date",
                ]
            )
            .reset_index(drop=True)
        )

        featured = cls.build_features(
            combined
        )

        prediction_keys = prediction[
            [
                "date",
                "outlet_id",
                "product_id",
            ]
        ].copy()

        featured = featured.merge(
            prediction_keys,
            on=[
                "date",
                "outlet_id",
                "product_id",
            ],
            how="inner",
        )

        missing = (
            set(model.feature_columns)
            - set(featured.columns)
        )

        if missing:
            raise ValueError(
                "Production inference is missing model features: "
                f"{sorted(missing)}"
            )

        X = featured[
            model.feature_columns
        ].copy()

        valid = (
            X.notna().all(axis=1)
            & np.isfinite(
                X.to_numpy(
                    dtype=float
                )
            ).all(axis=1)
        )

        return (
            X.astype(float),
            valid,
        )

    # ============================================================
    # PREDICTION
    # ============================================================

    @classmethod
    def predict(
        cls,
        model: XGBoostDemandModel,
        df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:

        if history_df is None:
            history_df = df

        X, valid = cls._prediction_frame(
            model,
            history_df,
            df,
        )

        predictions = np.full(
            len(X),
            np.nan,
            dtype=float,
        )

        if valid.any():
            predictions[
                valid.to_numpy()
            ] = model.predict(
                X.loc[valid]
            )

        return np.maximum(
            predictions,
            0.0,
        )

    # ============================================================
    # METRICS
    # ============================================================

    @staticmethod
    def metrics(
        actual,
        prediction,
    ) -> dict[str, float]:

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
                "No valid rows available for evaluation."
            )

        error = prediction - actual

        mae = float(
            np.mean(
                np.abs(error)
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        bias = float(
            np.mean(error)
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "rows": int(len(actual)),
        }

    # ============================================================
    # CANDIDATE VS CHAMPION
    # ============================================================

    @classmethod
    def evaluate(
        cls,
        champion_model: XGBoostDemandModel,
        candidate_model: XGBoostDemandModel,
        evaluation_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> dict:

        if history_df is None:
            history_df = evaluation_df

        actual = pd.to_numeric(
            evaluation_df[
                cls.TARGET_COLUMN
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        champion_prediction = cls.predict(
            champion_model,
            evaluation_df,
            history_df=history_df,
        )

        candidate_prediction = cls.predict(
            candidate_model,
            evaluation_df,
            history_df=history_df,
        )

        valid = (
            np.isfinite(actual)
            & np.isfinite(champion_prediction)
            & np.isfinite(candidate_prediction)
        )

        if valid.sum() < cls.MIN_EVALUATION_ROWS:
            raise ValueError(
                "Insufficient common unseen evaluation rows: "
                f"{valid.sum()} < {cls.MIN_EVALUATION_ROWS}"
            )

        champion_metrics = cls.metrics(
            actual[valid],
            champion_prediction[valid],
        )

        candidate_metrics = cls.metrics(
            actual[valid],
            candidate_prediction[valid],
        )

        champion_mae = champion_metrics[
            "mae"
        ]

        candidate_mae = candidate_metrics[
            "mae"
        ]

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
                candidate_metrics[
                    "bias"
                ]
            )
            - abs(
                champion_metrics[
                    "bias"
                ]
            )
        )

        passes = bool(
            np.isfinite(
                candidate_mae
            )
            and improvement_pct
            >= cls.MIN_IMPROVEMENT_PCT
            and bias_change
            <= cls.MAX_BIAS_WORSENING
        )

        reasons = []

        if improvement_pct < cls.MIN_IMPROVEMENT_PCT:
            reasons.append(
                f"insufficient_improvement:{improvement_pct:.4f}%"
            )

        if bias_change > cls.MAX_BIAS_WORSENING:
            reasons.append(
                f"bias_worsened:{bias_change:.4f}"
            )

        return {
            "champion": champion_metrics,
            "candidate": candidate_metrics,
            "improvement_pct": float(
                improvement_pct
            ),
            "bias_change": float(
                bias_change
            ),
            "passes": passes,
            "reasons": reasons,
            "evaluation_rows": int(
                valid.sum()
            ),
        }

    # ============================================================
    # INITIAL CHAMPION
    # ============================================================

    @classmethod
    def create_initial_champion(
        cls,
        training_df: pd.DataFrame,
        registry: ModelRegistry,
    ) -> tuple[
        XGBoostDemandModel,
        str,
    ]:

        model = cls.train_model(
            training_df
        )

        run_id = (
            "initial_"
            + uuid.uuid4().hex[:12]
        )

        manifest = {
            "run_id": run_id,
            "model_name": "initial_production_xgboost",
            "model_type": "XGBoostDemandModel",
            "feature_count": len(
                model.feature_columns
            ),
            "feature_columns": model.feature_columns,
            "training_rows": int(
                len(training_df)
            ),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "initial_champion",
        }

        candidate_path = registry.save_candidate(
            model,
            run_id,
            manifest,
        )

        registry.promote(
            candidate_path,
            manifest,
        )

        return (
            model,
            run_id,
        )

    # ============================================================
    # CANDIDATE ARTIFACT
    # ============================================================

    @classmethod
    def save_candidate(
        cls,
        model: XGBoostDemandModel,
        registry: ModelRegistry,
        run_id: str,
        evaluation: dict,
        training_rows: int,
    ) -> Path:

        manifest = {
            "run_id": run_id,
            "model_name": "production_xgboost_candidate",
            "model_type": "XGBoostDemandModel",
            "feature_count": len(
                model.feature_columns
            ),
            "feature_columns": model.feature_columns,
            "training_rows": int(
                training_rows
            ),
            "candidate_mae": evaluation[
                "candidate"
            ]["mae"],
            "candidate_rmse": evaluation[
                "candidate"
            ]["rmse"],
            "candidate_bias": evaluation[
                "candidate"
            ]["bias"],
            "champion_mae": evaluation[
                "champion"
            ]["mae"],
            "champion_rmse": evaluation[
                "champion"
            ]["rmse"],
            "champion_bias": evaluation[
                "champion"
            ]["bias"],
            "improvement_pct": evaluation[
                "improvement_pct"
            ],
            "bias_change": evaluation[
                "bias_change"
            ],
            "promotion_passed": evaluation[
                "passes"
            ],
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        return registry.save_candidate(
            model,
            run_id,
            manifest,
        )