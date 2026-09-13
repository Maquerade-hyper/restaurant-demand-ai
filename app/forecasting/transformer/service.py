from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.transformer.benchmark import (
    TransformerBenchmark,
)


class TransformerBenchmarkService:

    """
    Part 25E - Unified multi-series Transformer benchmark.
    """

    def __init__(
        self,
        sequence_length: int = 28,
        test_size: int = 30,
        validation_size: int = 30,
        improvement_threshold: float = 0.02,
        minimum_series_win_rate: float = 0.50,
        epochs: int = 12,
        batch_size: int = 32,
    ) -> None:

        self.sequence_length = sequence_length
        self.test_size = test_size
        self.validation_size = validation_size
        self.improvement_threshold = (
            improvement_threshold
        )
        self.minimum_series_win_rate = (
            minimum_series_win_rate
        )
        self.epochs = epochs
        self.batch_size = batch_size

    def benchmark(
        self,
        frame: pd.DataFrame,
        max_series: int = 12,
    ):

        benchmark = TransformerBenchmark(
            sequence_length=self.sequence_length,
            test_size=self.test_size,
            validation_size=self.validation_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

        return benchmark.run(
            frame,
            max_series=max_series,
        )

    def production_decision(
        self,
        aggregate: pd.DataFrame,
        series_detail: pd.DataFrame,
    ) -> dict:

        required_models = {
            "xgboost",
            "transformer",
        }

        models = set(
            aggregate["model_name"]
        )

        missing = (
            required_models
            -
            models
        )

        if missing:

            raise ValueError(
                "Missing models: "
                +
                ", ".join(
                    sorted(missing)
                )
            )

        xgb = aggregate[
            aggregate["model_name"]
            ==
            "xgboost"
        ].iloc[0]

        transformer = aggregate[
            aggregate["model_name"]
            ==
            "transformer"
        ].iloc[0]

        xgb_mae = float(
            xgb["mae"]
        )

        transformer_mae = float(
            transformer["mae"]
        )

        improvement = (
            xgb_mae
            -
            transformer_mae
        ) / max(
            xgb_mae,
            1e-9,
        )

        transformer_wins = (
            series_detail[
                "transformer_mae"
            ]
            <
            series_detail[
                "xgboost_mae"
            ]
        )

        series_win_rate = float(
            transformer_wins.mean()
        )

        meaningful_accuracy = (
            improvement
            >=
            self.improvement_threshold
        )

        sufficient_series_wins = (
            series_win_rate
            >=
            self.minimum_series_win_rate
        )

        transformer_candidate = (
            meaningful_accuracy
            and
            sufficient_series_wins
        )

        if transformer_candidate:

            recommendation = (
                "TRANSFORMER_CANDIDATE"
            )

        else:

            recommendation = (
                "KEEP_XGBOOST"
            )

        return {
            "recommendation": recommendation,
            "xgboost_mae": xgb_mae,
            "transformer_mae": transformer_mae,
            "transformer_mae_improvement": float(
                improvement
            ),
            "transformer_series_win_rate": (
                series_win_rate
            ),
            "minimum_series_win_rate": (
                self.minimum_series_win_rate
            ),
            "meaningful_improvement": bool(
                meaningful_accuracy
            ),
            "sufficient_series_wins": bool(
                sufficient_series_wins
            ),
            "transformer_prediction_cost": float(
                transformer[
                    "prediction_cost"
                ]
            ),
            "xgboost_prediction_cost": float(
                xgb[
                    "prediction_cost"
                ]
            ),
        }

    def validate(
        self,
        aggregate: pd.DataFrame,
        series_detail: pd.DataFrame,
    ) -> dict:

        errors = []

        required_models = {
            "naive",
            "xgboost",
            "transformer",
        }

        if not isinstance(
            aggregate,
            pd.DataFrame,
        ):

            errors.append(
                "aggregate must be a DataFrame"
            )

            return {
                "passed": False,
                "errors": errors,
            }

        if not isinstance(
            series_detail,
            pd.DataFrame,
        ):

            errors.append(
                "series_detail must be a DataFrame"
            )

        models = set(
            aggregate[
                "model_name"
            ]
        )

        missing_models = (
            required_models
            -
            models
        )

        if missing_models:

            errors.append(
                "Missing models: "
                +
                ", ".join(
                    sorted(
                        missing_models
                    )
                )
            )

        required_metrics = [
            "mae",
            "rmse",
            "smape",
            "bias",
            "high_demand_mae",
            "high_demand_bias",
            "spike_recall",
            "prediction_seconds",
            "prediction_cost",
        ]

        for column in required_metrics:

            if column not in aggregate.columns:

                errors.append(
                    f"Missing metric: {column}"
                )

                continue

            values = pd.to_numeric(
                aggregate[column],
                errors="coerce",
            )

            if not np.isfinite(
                values
            ).all():

                errors.append(
                    f"Non-finite values: {column}"
                )

        if series_detail.empty:

            errors.append(
                "No series-level benchmark results"
            )

        else:

            required_detail = {
                "outlet_id",
                "product_id",
                "naive_mae",
                "xgboost_mae",
                "transformer_mae",
                "best_model",
            }

            missing_detail = (
                required_detail
                -
                set(series_detail.columns)
            )

            if missing_detail:

                errors.append(
                    "Missing series columns: "
                    +
                    ", ".join(
                        sorted(
                            missing_detail
                        )
                    )
                )

            if (
                "best_model"
                in series_detail.columns
            ):

                allowed = {
                    "naive",
                    "xgboost",
                    "transformer",
                }

                invalid = set(
                    series_detail[
                        "best_model"
                    ]
                ) - allowed

                if invalid:

                    errors.append(
                        "Invalid best_model values: "
                        +
                        ", ".join(
                            sorted(
                                invalid
                            )
                        )
                    )

        if "rank" in aggregate.columns:

            expected = list(
                range(
                    1,
                    len(aggregate) + 1,
                )
            )

            actual = (
                aggregate[
                    "rank"
                ]
                .astype(int)
                .tolist()
            )

            if actual != expected:

                errors.append(
                    "Invalid ranking sequence"
                )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "models": sorted(
                models
            ),
            "aggregate_rows": int(
                len(aggregate)
            ),
            "series_count": int(
                len(series_detail)
            ),
        }