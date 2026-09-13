from __future__ import annotations

import pandas as pd

from .behavior import build_outlet_demand_behavior
from .performance import build_outlet_performance
from .profiling import build_outlet_profiles

from .clustering import build_outlet_segments

from .validation import validate_outlet_intelligence


class OutletProfilingService:
    def analyze(
        self,
        outlets: pd.DataFrame,
        demand: pd.DataFrame,
    ) -> pd.DataFrame:
        return build_outlet_profiles(
            outlets=outlets,
            demand=demand,
        )

    def top_risk_outlets(
        self,
        profiles: pd.DataFrame,
        limit: int = 20,
    ) -> pd.DataFrame:
        return profiles.sort_values(
            [
                "operational_risk",
                "lost_demand_rate",
                "stockout_rate",
            ],
            ascending=False,
        ).head(limit)

    def analyze_behavior(
        self,
        demand: pd.DataFrame,
    ) -> pd.DataFrame:
        return build_outlet_demand_behavior(demand)

    def top_behavioral_outlets(
        self,
        behavior: pd.DataFrame,
        limit: int = 20,
    ) -> pd.DataFrame:
        return behavior.sort_values(
            [
                "peak_intensity",
                "coefficient_variation",
            ],
            ascending=False,
        ).head(limit)

    def analyze_performance(
        self,
        demand: pd.DataFrame,
    ) -> pd.DataFrame:
        return build_outlet_performance(demand)

    def top_performing_outlets(
        self,
        performance: pd.DataFrame,
        limit: int = 20,
    ) -> pd.DataFrame:
        return performance.sort_values(
            "performance_score",
            ascending=False,
        ).head(limit)

    def top_opportunity_outlets(
        self,
        performance: pd.DataFrame,
        limit: int = 20,
    ) -> pd.DataFrame:
        return performance.sort_values(
            "opportunity_index",
            ascending=False,
        ).head(limit)

    def top_operational_risk_outlets(
        self,
        performance: pd.DataFrame,
        limit: int = 20,
    ) -> pd.DataFrame:
        return performance.sort_values(
            "risk_index",
            ascending=False,
        ).head(limit)

    def analyze_segments(
        self,
        performance: pd.DataFrame,
        min_clusters: int = 2,
        max_clusters: int = 8,
    ):
        return build_outlet_segments(
            performance,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

    def segment_outlets(
        self,
        performance: pd.DataFrame,
    ):
        clustered, profiles, evaluation, metadata = (
            self.analyze_segments(performance)
        )

        return clustered

    def validate_intelligence(
        self,
        profiles: pd.DataFrame,
        behavior: pd.DataFrame,
        performance: pd.DataFrame,
        segments: pd.DataFrame,
    ):
        return validate_outlet_intelligence(
            profiles=profiles,
            behavior=behavior,
            performance=performance,
            segments=segments,
        )