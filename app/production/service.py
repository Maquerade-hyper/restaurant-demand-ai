from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from .batch import BatchInferenceEngine
from .inference import ProductionInferenceEngine
from .model_loader import ProductionModelLoader
from .performance import PerformanceMonitor
from .schemas import (
    PerformanceReport,
    ProductionForecastResponse,
)


class ProductionMLService:
    """
    Unified production ML service.

    Model is loaded once and reused.
    """

    def __init__(
        self,
        model_root: str = "models/champion",
    ):

        self.loader = ProductionModelLoader(
            model_root=model_root
        )

        self.engine = (
            ProductionInferenceEngine(
                self.loader
            )
        )

        self.batch = (
            BatchInferenceEngine(
                self.engine
            )
        )

        self.performance = (
            PerformanceMonitor(
                cpu_only=True
            )
        )

    def load_model(self):

        model = self.loader.load()

        return {
            "model_name": (
                self.loader.model_name()
            ),
            "model_version": (
                self.loader.model_version()
            ),
            "load_ms": (
                self.loader.load_time_ms()
            ),
            "model_loaded": (
                model is not None
            ),
        }

    def forecast(
        self,
        outlet_id: str,
        product_id: str,
        history: pd.DataFrame,
        horizon: int = 1,
    ) -> ProductionForecastResponse:

        if horizon < 1:
            raise ValueError(
                "horizon must be >= 1"
            )

        if horizon != 1:
            raise ValueError(
                "Part 29 currently exposes "
                "the validated D+1 production path. "
                "Multi-horizon production routing "
                "is retained for the next integration stage."
            )

        history = history[
            (
                history["outlet_id"]
                == outlet_id
            )
            & (
                history["product_id"]
                == product_id
            )
        ].copy()

        if history.empty:
            raise ValueError(
                "No history found for requested "
                "outlet/product."
            )

        start = (
            time.perf_counter()
        )

        prediction, _ = (
            self.engine.forecast_latest(
                history
            )
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return ProductionForecastResponse(
            outlet_id=outlet_id,
            product_id=product_id,
            horizon=horizon,
            predictions=[
                prediction
            ],
            model_name=(
                self.loader.model_name()
            ),
            model_version=(
                self.loader.model_version()
            ),
            inference_ms=float(
                elapsed_ms
            ),
        )

    def batch_forecast(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:

        return self.batch.predict(
            history
        )

    def benchmark(
        self,
        history: pd.DataFrame,
        iterations: int = 20,
    ) -> PerformanceReport:

        if history.empty:
            raise ValueError(
                "History cannot be empty."
            )

        # Warm model cache first.
        self.loader.load()

        series = (
            history[
                [
                    "outlet_id",
                    "product_id",
                ]
            ]
            .drop_duplicates()
            .iloc[0]
        )

        outlet_id = series[
            "outlet_id"
        ]

        product_id = series[
            "product_id"
        ]

        series_history = history[
            (
                history["outlet_id"]
                == outlet_id
            )
            & (
                history["product_id"]
                == product_id
            )
        ].copy()

        report = (
            self.performance.benchmark(
                lambda: self.forecast(
                    outlet_id,
                    product_id,
                    series_history,
                    horizon=1,
                ),
                iterations=iterations,
                rows_per_request=1,
            )
        )

        report.model_load_ms = (
            self.loader.load_time_ms()
        )

        return report

    def validate(self) -> dict:

        errors = []

        if not self.loader.exists():
            errors.append(
                "champion_model_missing"
            )

        try:
            self.loader.load()

        except Exception as exc:
            errors.append(
                f"model_load_failed:{exc}"
            )

        if not self.loader.model_name():
            errors.append(
                "missing_model_name"
            )

        if not self.loader.model_version():
            errors.append(
                "missing_model_version"
            )

        return {
            "passed": (
                len(errors) == 0
            ),
            "errors": errors,
        }