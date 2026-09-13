from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .inference import ProductionInferenceEngine


class BatchInferenceEngine:
    """
    CPU-first batch inference.

    Groups requests by outlet/product and performs one feature
    construction pass per series.
    """

    def __init__(
        self,
        engine: ProductionInferenceEngine,
    ):
        self.engine = engine

    def predict(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:

        required = {
            "date",
            "outlet_id",
            "product_id",
            "deconstrained_demand",
        }

        missing = required - set(
            history.columns
        )

        if missing:
            raise ValueError(
                f"Missing columns: "
                f"{sorted(missing)}"
            )

        outputs = []

        for (
            outlet_id,
            product_id,
        ), group in history.groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        ):

            prediction, inference_ms = (
                self.engine.forecast_latest(
                    group
                )
            )

            outputs.append(
                {
                    "outlet_id": outlet_id,
                    "product_id": product_id,
                    "prediction": prediction,
                    "inference_ms": inference_ms,
                }
            )

        if not outputs:
            return pd.DataFrame(
                columns=[
                    "outlet_id",
                    "product_id",
                    "prediction",
                    "inference_ms",
                ]
            )

        return pd.DataFrame(
            outputs
        )