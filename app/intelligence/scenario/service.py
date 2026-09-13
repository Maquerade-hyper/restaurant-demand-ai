from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import ScenarioEngine
from .schemas import ScenarioDefinition


class ScenarioEngineService:

    def __init__(self):

        self.engine = ScenarioEngine()

    # ---------------------------------------------------------
    # Public scenario creation
    # ---------------------------------------------------------

    def create_scenario(
        self,
        name: str,
        **kwargs,
    ) -> ScenarioDefinition:

        return self.engine.create_scenario(
            name=name,
            **kwargs,
        )

    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------

    def analyze(
        self,
        forecast: pd.DataFrame,
        scenarios: list[ScenarioDefinition],
    ) -> pd.DataFrame:

        if forecast.empty:
            raise ValueError(
                "forecast cannot be empty"
            )

        if not scenarios:
            raise ValueError(
                "scenarios cannot be empty"
            )

        if "baseline_demand" not in forecast.columns:
            raise ValueError(
                "forecast must contain "
                "baseline_demand"
            )

        result = self.engine.compare(
            forecast=forecast,
            scenarios=scenarios,
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

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summarize(
        self,
        result: pd.DataFrame,
    ) -> pd.DataFrame:

        return self.engine.summarize(
            result
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate(
        self,
        result: pd.DataFrame,
    ) -> dict:

        errors: list[str] = []

        required = {
            "scenario_name",
            "baseline_demand",
            "scenario_demand",
            "scenario_multiplier",
            "absolute_change",
            "percentage_change",
        }

        missing = required - set(
            result.columns
        )

        if missing:
            errors.append(
                f"missing columns: "
                f"{sorted(missing)}"
            )

            return {
                "passed": False,
                "errors": errors,
            }

        if result.empty:
            errors.append(
                "scenario result is empty"
            )

        # ------------------------------------------------------
        # Numerical validity
        # ------------------------------------------------------

        numeric_columns = [
            "baseline_demand",
            "scenario_demand",
            "scenario_multiplier",
            "absolute_change",
            "percentage_change",
        ]

        for column in numeric_columns:

            values = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            if values.isna().any():
                errors.append(
                    f"{column} contains invalid values"
                )
                continue

            if not np.isfinite(
                values
            ).all():

                errors.append(
                    f"{column} contains non-finite values"
                )

        # ------------------------------------------------------
        # Demand cannot become negative.
        # ------------------------------------------------------

        if (
            result["baseline_demand"]
            < 0
        ).any():

            errors.append(
                "baseline demand contains negative values"
            )

        if (
            result["scenario_demand"]
            < 0
        ).any():

            errors.append(
                "scenario demand contains negative values"
            )

        # ------------------------------------------------------
        # Multiplier validity.
        # ------------------------------------------------------

        if (
            result["scenario_multiplier"]
            <= 0
        ).any():

            errors.append(
                "scenario multiplier must be positive"
            )

        # ------------------------------------------------------
        # Arithmetic reconciliation.
        # ------------------------------------------------------

        expected_change = (
            result["scenario_demand"]
            - result["baseline_demand"]
        )

        if not np.allclose(
            result["absolute_change"],
            expected_change,
            atol=1e-8,
        ):

            errors.append(
                "absolute change reconciliation failed"
            )

        # ------------------------------------------------------
        # Truth leakage protection.
        # ------------------------------------------------------

        forbidden = {
            "true_demand",
            "lost_demand_truth",
            "demand_truth",
            "future_demand",
            "actual_demand",
        }

        leakage = forbidden.intersection(
            result.columns
        )

        if leakage:
            errors.append(
                f"truth columns present: "
                f"{sorted(leakage)}"
            )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
        }