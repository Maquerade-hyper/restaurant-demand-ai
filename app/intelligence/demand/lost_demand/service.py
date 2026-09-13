from __future__ import annotations

import pandas as pd

from .metrics import (
    add_lost_demand_metrics,
    aggregate_lost_demand,
)
from .patterns import (
    add_loss_patterns,
    classify_loss_severity,
)
from .prioritization import (
    calculate_priority_score,
)


class LostDemandIntelligenceService:
    """
    Main Part 16C service.

    Input:
        Output from Part 16B.

    Output:
        Daily lost-demand intelligence and
        operational prioritization.
    """

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = add_lost_demand_metrics(
            df
        )

        result = add_loss_patterns(
            result
        )

        result = classify_loss_severity(
            result
        )

        result = calculate_priority_score(
            result
        )

        return result

    def summarize_outlets(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        analyzed = self.analyze(
            df
        )

        return aggregate_lost_demand(
            analyzed,
            ["outlet_id"],
        )

    def summarize_products(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        analyzed = self.analyze(
            df
        )

        return aggregate_lost_demand(
            analyzed,
            ["product_id"],
        )

    def summarize_outlet_products(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        analyzed = self.analyze(
            df
        )

        return aggregate_lost_demand(
            analyzed,
            [
                "outlet_id",
                "product_id",
            ],
        )

    def top_opportunities(
        self,
        df: pd.DataFrame,
        limit: int = 20,
    ) -> pd.DataFrame:

        analyzed = self.analyze(
            df
        )

        return (
            analyzed.sort_values(
                "priority_score",
                ascending=False,
            )
            .head(limit)
            .reset_index(drop=True)
        )