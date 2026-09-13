from __future__ import annotations

import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.forecasting.transformer.trainer import (
    TransformerTrainer,
)


def mae(actual, prediction) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    return float(
        np.mean(
            np.abs(
                actual - prediction
            )
        )
    )


def rmse(actual, prediction) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (actual - prediction) ** 2
            )
        )
    )


def smape(actual, prediction) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    denominator = (
        np.abs(actual)
        + np.abs(prediction)
        + 1e-6
    )

    return float(
        np.mean(
            2.0
            * np.abs(actual - prediction)
            / denominator
        )
        * 100.0
    )


def bias(actual, prediction) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    return float(
        np.mean(
            prediction - actual
        )
    )


def high_demand_metrics(
    actual,
    prediction,
    threshold,
) -> tuple[float, float]:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    mask = actual >= threshold

    if not np.any(mask):
        return 0.0, 0.0

    return (
        float(
            np.mean(
                np.abs(
                    actual[mask]
                    -
                    prediction[mask]
                )
            )
        ),
        float(
            np.mean(
                prediction[mask]
                -
                actual[mask]
            )
        ),
    )


def spike_recall(
    actual,
    prediction,
    threshold,
) -> float:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    actual_spikes = (
        actual >= threshold
    )

    if not np.any(actual_spikes):
        return 0.0

    predicted_spikes = (
        prediction >= threshold
    )

    return float(
        np.sum(
            actual_spikes
            &
            predicted_spikes
        )
        /
        np.sum(
            actual_spikes
        )
    )


def build_xgb_features(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:

    values = np.asarray(
        series,
        dtype=float,
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "quantity_sold": values,
        }
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["dow_sin"] = np.sin(
        2.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["dow_cos"] = np.cos(
        2.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["month_sin"] = np.sin(
        2.0
        * np.pi
        * (df["month"] - 1)
        / 12.0
    )

    df["month_cos"] = np.cos(
        2.0
        * np.pi
        * (df["month"] - 1)
        / 12.0
    )

    group = df["quantity_sold"]

    for lag in [1, 2, 3, 7, 14, 28]:

        df[f"lag_{lag}"] = (
            group.shift(lag)
        )

    for window in [3, 7, 14, 28]:

        shifted = group.shift(1)

        df[
            f"rolling_mean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        df[
            f"rolling_std_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )

    df["trend_1d"] = (
        df["lag_1"]
        -
        df["lag_2"]
    )

    df["trend_7d"] = (
        df["rolling_mean_7"]
        -
        df["rolling_mean_14"]
    )

    df["trend_14d"] = (
        df["rolling_mean_14"]
        -
        df["rolling_mean_28"]
    )

    eps = 1e-6

    df["trend_ratio_7d"] = (
        df["rolling_mean_7"]
        /
        (
            df["rolling_mean_14"]
            + eps
        )
    )

    df["trend_ratio_14d"] = (
        df["rolling_mean_14"]
        /
        (
            df["rolling_mean_28"]
            + eps
        )
    )

    df["trend_acceleration"] = (
        df["trend_7d"]
        -
        df["trend_14d"]
    )

    df["volatility_3"] = (
        df["rolling_std_3"]
    )

    df["coefficient_variation_3"] = (
        df["rolling_std_3"]
        /
        (
            df["rolling_mean_3"].abs()
            + eps
        )
    )

    df["volatility_7"] = (
        df["rolling_std_7"]
    )

    df["coefficient_variation_7"] = (
        df["rolling_std_7"]
        /
        (
            df["rolling_mean_7"].abs()
            + eps
        )
    )

    df["volatility_14"] = (
        df["rolling_std_14"]
    )

    df["coefficient_variation_14"] = (
        df["rolling_std_14"]
        /
        (
            df["rolling_mean_14"].abs()
            + eps
        )
    )

    df["volatility_28"] = (
        df["rolling_std_28"]
    )

    df["coefficient_variation_28"] = (
        df["rolling_std_28"]
        /
        (
            df["rolling_mean_28"].abs()
            + eps
        )
    )

    df["order_velocity_3d"] = (
        df["rolling_mean_3"]
    )

    df["order_velocity_7d"] = (
        df["rolling_mean_7"]
    )

    df["order_velocity_14d"] = (
        df["rolling_mean_14"]
    )

    df["order_velocity_change"] = (
        df["order_velocity_3d"]
        -
        df["order_velocity_14d"]
    )

    df["dow_sin_2"] = np.sin(
        4.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["dow_cos_2"] = np.cos(
        4.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["month_sin_2"] = np.sin(
        4.0
        * np.pi
        * (df["month"] - 1)
        / 12.0
    )

    df["month_cos_2"] = np.cos(
        4.0
        * np.pi
        * (df["month"] - 1)
        / 12.0
    )

    df["quarter_sin"] = np.sin(
        2.0
        * np.pi
        * (df["quarter"] - 1)
        / 4.0
    )

    df["quarter_cos"] = np.cos(
        2.0
        * np.pi
        * (df["quarter"] - 1)
        / 4.0
    )

    df["weekend_month_interaction"] = (
        df["is_weekend"]
        *
        df["month"]
    )

    feature_columns = [
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "day_of_year",
        "quarter",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
        "rolling_mean_28",
        "rolling_std_28",
        "trend_1d",
        "trend_7d",
        "trend_14d",
        "trend_ratio_7d",
        "trend_ratio_14d",
        "trend_acceleration",
        "volatility_3",
        "coefficient_variation_3",
        "volatility_7",
        "coefficient_variation_7",
        "volatility_14",
        "coefficient_variation_14",
        "volatility_28",
        "coefficient_variation_28",
        "order_velocity_3d",
        "order_velocity_7d",
        "order_velocity_14d",
        "order_velocity_change",
        "dow_sin_2",
        "dow_cos_2",
        "month_sin_2",
        "month_cos_2",
        "quarter_sin",
        "quarter_cos",
        "weekend_month_interaction",
    ]

    return df[feature_columns]


class TransformerBenchmark:

    """
    Part 25D

    Multi-series benchmark.

    Every model receives exactly the same:

        training period
        validation period
        unseen test period

    No test target is used during training.
    """

    def __init__(
        self,
        sequence_length: int = 28,
        test_size: int = 30,
        validation_size: int = 30,
        seed: int = 42,
        epochs: int = 12,
        batch_size: int = 32,
    ) -> None:

        self.sequence_length = sequence_length
        self.test_size = test_size
        self.validation_size = validation_size
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size

    def run(
        self,
        frame: pd.DataFrame,
        max_series: int = 12,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        required = {
            "outlet_id",
            "product_id",
            "date",
            "deconstrained_demand",
        }

        missing = (
            required
            -
            set(frame.columns)
        )

        if missing:

            raise ValueError(
                "Missing columns: "
                +
                ", ".join(
                    sorted(missing)
                )
            )

        data = frame.copy()

        data["date"] = pd.to_datetime(
            data["date"]
        )

        data[
            "deconstrained_demand"
        ] = pd.to_numeric(
            data[
                "deconstrained_demand"
            ],
            errors="coerce",
        )

        data = data.dropna(
            subset=[
                "date",
                "deconstrained_demand",
            ]
        )

        series_keys = (
            data[
                [
                    "outlet_id",
                    "product_id",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "outlet_id",
                    "product_id",
                ]
            )
            .reset_index(drop=True)
        )

        if max_series is not None:

            series_keys = series_keys.head(
                max_series
            )

        results = []
        series_results = []

        for _, key in series_keys.iterrows():

            outlet_id = key[
                "outlet_id"
            ]

            product_id = key[
                "product_id"
            ]

            series = data[
                (
                    data["outlet_id"]
                    ==
                    outlet_id
                )
                &
                (
                    data["product_id"]
                    ==
                    product_id
                )
            ][
                [
                    "date",
                    "deconstrained_demand",
                ]
            ].sort_values(
                "date"
            )

            if len(series) < (
                self.sequence_length
                +
                self.validation_size
                +
                self.test_size
                +
                5
            ):
                continue

            dates = pd.DatetimeIndex(
                series["date"]
            )

            values = series[
                "deconstrained_demand"
            ].to_numpy(
                dtype=float
            )

            result, detail = (
                self._run_series(
                    dates,
                    values,
                    outlet_id,
                    product_id,
                )
            )

            results.extend(result)
            series_results.append(
                detail
            )

        if not results:

            raise ValueError(
                "No valid series available "
                "for Transformer benchmark"
            )

        aggregate = (
            self._aggregate(
                results
            )
        )

        series_detail = pd.DataFrame(
            series_results
        )

        return (
            aggregate,
            series_detail,
        )

    def _run_series(
        self,
        dates,
        values,
        outlet_id,
        product_id,
    ):

        test_start = (
            len(values)
            -
            self.test_size
        )

        validation_start = (
            test_start
            -
            self.validation_size
        )

        train_values = values[
            :validation_start
        ]

        validation_values = values[
            validation_start:test_start
        ]

        test_values = values[
            test_start:
        ]

        # ========================================================
        # NAIVE
        # ========================================================

        naive_prediction = np.repeat(
            train_values[-1],
            self.test_size,
        )

        naive_result = (
            self._metric_row(
                "naive",
                test_values,
                naive_prediction,
                train_values,
                0.0,
                outlet_id,
                product_id,
            )
        )

        # ========================================================
        # XGBOOST
        # ========================================================

        # IMPORTANT:
        # XGBoost trains only on the same training period.
        # Validation is NOT added to XGBoost training.
        xgb_dates = dates[
            :validation_start
        ]

        xgb_features = (
            build_xgb_features(
                train_values,
                xgb_dates,
            )
        )

        xgb_train = xgb_features.copy()

        xgb_train["target"] = (
            train_values
        )

        xgb_train = xgb_train.dropna()

        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=self.seed,
            n_jobs=1,
        )

        xgb_start = time.perf_counter()

        xgb_model.fit(
            xgb_train.drop(
                columns=["target"]
            ),
            xgb_train["target"],
            verbose=False,
        )

        working = list(
            train_values
        )

        xgb_predictions = []

        for step in range(
            self.test_size
        ):

            working_dates = pd.date_range(
                start=dates[0],
                periods=len(working),
                freq="D",
            )

            features = (
                build_xgb_features(
                    np.asarray(
                        working,
                        dtype=float,
                    ),
                    working_dates,
                )
            )

            prediction = float(
                xgb_model.predict(
                    features.iloc[[-1]]
                )[0]
            )

            prediction = max(
                0.0,
                prediction,
            )

            xgb_predictions.append(
                prediction
            )

            working.append(
                prediction
            )

        xgb_seconds = (
            time.perf_counter()
            -
            xgb_start
        )

        xgb_result = (
            self._metric_row(
                "xgboost",
                test_values,
                np.asarray(
                    xgb_predictions
                ),
                train_values,
                xgb_seconds,
                outlet_id,
                product_id,
            )
        )

        # ========================================================
        # TRANSFORMER
        # ========================================================

        trainer = TransformerTrainer(
            sequence_length=self.sequence_length,
            d_model=32,
            n_heads=4,
            n_layers=2,
            feedforward_dim=64,
            dropout=0.10,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=4,
            seed=self.seed,
            device="cpu",
        )

        training_info = trainer.fit(
            train_values,
            validation_values=validation_values,
        )

        transformer_start = (
            time.perf_counter()
        )

        transformer_predictions = (
            trainer.predict(
                train_values,
                horizon=self.test_size,
            )
        )

        transformer_seconds = (
            time.perf_counter()
            -
            transformer_start
        )

        transformer_result = (
            self._metric_row(
                "transformer",
                test_values,
                transformer_predictions,
                train_values,
                transformer_seconds,
                outlet_id,
                product_id,
            )
        )

        transformer_result[
            "training_seconds"
        ] = training_info[
            "training_seconds"
        ]

        transformer_result[
            "epochs_completed"
        ] = training_info[
            "epochs_completed"
        ]

        # ========================================================
        # SERIES WINNER
        # ========================================================

        model_rows = {
            row["model_name"]: row
            for row in [
                naive_result,
                xgb_result,
                transformer_result,
            ]
        }

        best_model = min(
            model_rows,
            key=lambda name:
                model_rows[name]["mae"],
        )

        detail = {
            "outlet_id": outlet_id,
            "product_id": product_id,
            "test_start": str(
                dates[test_start].date()
            ),
            "test_end": str(
                dates[-1].date()
            ),
            "naive_mae": naive_result[
                "mae"
            ],
            "xgboost_mae": xgb_result[
                "mae"
            ],
            "transformer_mae": transformer_result[
                "mae"
            ],
            "transformer_vs_xgb_improvement": (
                (
                    xgb_result["mae"]
                    -
                    transformer_result["mae"]
                )
                /
                max(
                    xgb_result["mae"],
                    1e-9,
                )
            ),
            "best_model": best_model,
        }

        return (
            [
                naive_result,
                xgb_result,
                transformer_result,
            ],
            detail,
        )

    def _metric_row(
        self,
        model_name,
        actual,
        prediction,
        training_values,
        prediction_seconds,
        outlet_id,
        product_id,
    ):

        actual = np.asarray(
            actual,
            dtype=float,
        )

        prediction = np.asarray(
            prediction,
            dtype=float,
        )

        # High-demand threshold is calculated from
        # training history only.
        threshold = float(
            np.quantile(
                training_values,
                0.90,
            )
        )

        high_mae, high_bias = (
            high_demand_metrics(
                actual,
                prediction,
                threshold,
            )
        )

        # Spike definition is also based on training
        # distribution rather than test information.
        spike_threshold = float(
            np.quantile(
                training_values,
                0.95,
            )
        )

        return {
            "outlet_id": outlet_id,
            "product_id": product_id,
            "model_name": model_name,
            "n_test": len(actual),
            "mae": mae(
                actual,
                prediction,
            ),
            "rmse": rmse(
                actual,
                prediction,
            ),
            "smape": smape(
                actual,
                prediction,
            ),
            "bias": bias(
                actual,
                prediction,
            ),
            "high_demand_mae": high_mae,
            "high_demand_bias": high_bias,
            "spike_recall": spike_recall(
                actual,
                prediction,
                spike_threshold,
            ),
            "prediction_seconds": float(
                prediction_seconds
            ),
            "prediction_cost": float(
                prediction_seconds
                /
                max(
                    len(actual),
                    1,
                )
            ),
        }

    def _aggregate(
        self,
        rows,
    ) -> pd.DataFrame:

        frame = pd.DataFrame(
            rows
        )

        output = []

        for model_name, group in frame.groupby(
            "model_name"
        ):

            output.append(
                {
                    "model_name": model_name,
                    "series_count": int(
                        group[
                            [
                                "outlet_id",
                                "product_id",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "test_observations": int(
                        group["n_test"].sum()
                    ),
                    "mae": float(
                        np.average(
                            group["mae"],
                            weights=group["n_test"],
                        )
                    ),
                    "rmse": float(
                        np.average(
                            group["rmse"],
                            weights=group["n_test"],
                        )
                    ),
                    "smape": float(
                        np.average(
                            group["smape"],
                            weights=group["n_test"],
                        )
                    ),
                    "bias": float(
                        np.average(
                            group["bias"],
                            weights=group["n_test"],
                        )
                    ),
                    "high_demand_mae": float(
                        np.average(
                            group["high_demand_mae"],
                            weights=group["n_test"],
                        )
                    ),
                    "high_demand_bias": float(
                        np.average(
                            group["high_demand_bias"],
                            weights=group["n_test"],
                        )
                    ),
                    "spike_recall": float(
                        np.average(
                            group["spike_recall"],
                            weights=group["n_test"],
                        )
                    ),
                    "prediction_seconds": float(
                        group[
                            "prediction_seconds"
                        ].sum()
                    ),
                    "prediction_cost": float(
                        group[
                            "prediction_seconds"
                        ].sum()
                        /
                        max(
                            group[
                                "n_test"
                            ].sum(),
                            1,
                        )
                    ),
                }
            )

        result = pd.DataFrame(
            output
        )

        result = result.sort_values(
            [
                "mae",
                "rmse",
            ]
        ).reset_index(
            drop=True
        )

        result["rank"] = (
            np.arange(
                len(result)
            )
            +
            1
        )

        result["winner"] = (
            result["rank"] == 1
        )

        return result