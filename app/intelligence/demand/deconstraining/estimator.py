from __future__ import annotations

import pandas as pd

from .reconstruction import (
    build_clean_reference,
    reconstruct_demand,
)
from .stockout import (
    prepare_stockout_signals,
)


class DeconstrainedDemandEstimator:
    """
    Availability-aware latent demand estimator.

    This estimator uses only information that can exist in
    operational sales/inventory data.

    It MUST NOT receive:
        true_demand
        lost_demand_truth
        demand_truth.csv
    """

    def __init__(
        self,
        lookback_days: int = 56,
        minimum_clean_days: int = 7,
    ):

        if lookback_days < 7:
            raise ValueError(
                "lookback_days must be >= 7"
            )

        if minimum_clean_days < 1:
            raise ValueError(
                "minimum_clean_days must be >= 1"
            )

        if minimum_clean_days > lookback_days:
            raise ValueError(
                "minimum_clean_days cannot exceed lookback_days"
            )

        self.lookback_days = lookback_days
        self.minimum_clean_days = (
            minimum_clean_days
        )

    def fit_transform(
        self,
        sales: pd.DataFrame,
        inventory: pd.DataFrame,
    ) -> pd.DataFrame:

        prepared = prepare_stockout_signals(
            sales=sales,
            inventory=inventory,
        )

        prepared = build_clean_reference(
            prepared,
            lookback_days=self.lookback_days,
            minimum_clean_days=self.minimum_clean_days,
        )

        prepared = reconstruct_demand(
            prepared
        )

        return prepared

    def transform(
        self,
        sales: pd.DataFrame,
        inventory: pd.DataFrame,
    ) -> pd.DataFrame:

        return self.fit_transform(
            sales=sales,
            inventory=inventory,
        )