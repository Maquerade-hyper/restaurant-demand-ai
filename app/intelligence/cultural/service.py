from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import prepare_calendar
from .cultural_effects import build_cultural_effects
from .demographics import prepare_demographics


class CulturalDemographicIntelligenceService:

    def prepare(
        self,
        demand: pd.DataFrame,
        calendar: pd.DataFrame,
        demographics: pd.DataFrame,
    ) -> pd.DataFrame:

        prepared_calendar = prepare_calendar(
            calendar
        )

        prepared_demographics = prepare_demographics(
            demographics
        )

        return build_cultural_effects(
            demand=demand,
            calendar=prepared_calendar,
            demographics=prepared_demographics,
        )

    def validate(
        self,
        result: pd.DataFrame,
    ) -> dict:

        errors: list[str] = []

        required = {
            "outlet_id",
            "product_id",
            "date",
            "demand",
            "holiday_importance",
            "religious_importance",
            "event_importance",
            "religious_population_share",
            "cultural_context_score",
            "cultural_demand_pressure",
            "historical_demand_reference",
        }

        missing = required - set(result.columns)

        if missing:
            errors.append(
                f"result missing columns: {sorted(missing)}"
            )

            return {
                "passed": False,
                "errors": errors,
                "rows": int(len(result)),
            }

        if result.empty:
            errors.append(
                "cultural intelligence result is empty"
            )

            return {
                "passed": False,
                "errors": errors,
                "rows": 0,
            }

        # ------------------------------------------------------
        # Numeric validity
        # ------------------------------------------------------

        numeric_columns = [
            "demand",
            "holiday_importance",
            "religious_importance",
            "event_importance",
            "population_density",
            "tourism_index",
            "student_index",
            "business_index",
            "residential_index",
            "religious_population_share",
            "young_population_share",
            "working_population_share",
            "religious_context_exposure",
            "tourism_context_exposure",
            "student_context_exposure",
            "business_context_exposure",
            "working_population_exposure",
            "cultural_context_score",
            "historical_demand_reference",
            "cultural_demand_pressure",
        ]

        for column in numeric_columns:

            if column not in result.columns:
                continue

            values = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            if values.isna().any():
                errors.append(
                    f"{column} contains invalid values"
                )
                continue

            if not np.isfinite(values).all():
                errors.append(
                    f"{column} contains non-finite values"
                )

            if (values < 0).any():
                errors.append(
                    f"{column} contains negative values"
                )

        # ------------------------------------------------------
        # Bounded context signals
        # ------------------------------------------------------

        bounded_columns = [
            "holiday_importance",
            "religious_importance",
            "event_importance",
            "religious_population_share",
            "cultural_context_score",
        ]

        for column in bounded_columns:

            values = result[column]

            if (values < 0).any() or (
                values > 1
            ).any():
                errors.append(
                    f"{column} outside [0, 1]"
                )

        # ------------------------------------------------------
        # Identity
        # ------------------------------------------------------

        if result["outlet_id"].isna().any():
            errors.append(
                "outlet_id contains null values"
            )

        if result["product_id"].isna().any():
            errors.append(
                "product_id contains null values"
            )

        if result["date"].isna().any():
            errors.append(
                "date contains null values"
            )

        # ------------------------------------------------------
        # Truth leakage protection
        # ------------------------------------------------------

        forbidden = {
            "true_demand",
            "lost_demand_truth",
            "demand_truth",
            "actual_demand",
        }

        leakage = forbidden.intersection(
            result.columns
        )

        if leakage:
            errors.append(
                f"truth columns present: {sorted(leakage)}"
            )

        # ------------------------------------------------------
        # Duplicate protection
        # ------------------------------------------------------

        if result.duplicated(
            subset=[
                "outlet_id",
                "product_id",
                "date",
            ]
        ).any():

            errors.append(
                "duplicate outlet/product/date records"
            )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "rows": int(len(result)),
        }

    def analyze(
        self,
        demand: pd.DataFrame,
        calendar: pd.DataFrame,
        demographics: pd.DataFrame,
    ) -> pd.DataFrame:

        result = self.prepare(
            demand=demand,
            calendar=calendar,
            demographics=demographics,
        )

        validation = self.validate(
            result
        )

        if not validation["passed"]:
            raise ValueError(
                "; ".join(
                    validation["errors"]
                )
            )

        return result