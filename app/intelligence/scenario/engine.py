from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .adjustments import (
    apply_scenario,
    combined_multiplier,
)
from .schemas import ScenarioDefinition


class ScenarioEngine:

    def __init__(
        self,
        min_multiplier: float = 0.50,
        max_multiplier: float = 2.00,
    ):

        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    # ---------------------------------------------------------
    # Scenario definitions
    # ---------------------------------------------------------

    def create_scenario(
        self,
        name: str,
        **kwargs,
    ) -> ScenarioDefinition:

        kwargs.setdefault(
            "min_multiplier",
            self.min_multiplier,
        )

        kwargs.setdefault(
            "max_multiplier",
            self.max_multiplier,
        )

        scenario = ScenarioDefinition(
            name=name,
            **kwargs,
        )

        # Validate every supplied multiplier.
        multiplier_values = {
            key: value
            for key, value in asdict(
                scenario
            ).items()
            if key.endswith("_multiplier")
        }

        combined_multiplier(
            multiplier_values
        )

        return scenario

    # ---------------------------------------------------------
    # Apply one scenario
    # ---------------------------------------------------------

    def run(
        self,
        forecast: pd.DataFrame,
        scenario: ScenarioDefinition,
        baseline_column: str = "baseline_demand",
    ) -> pd.DataFrame:

        result = forecast.copy()

        for key in [
            "holiday_multiplier",
            "promotion_multiplier",
            "event_multiplier",
            "weather_multiplier",
            "tourism_multiplier",
            "demographic_multiplier",
            "cultural_multiplier",
        ]:

            result[key] = getattr(
                scenario,
                key,
            )

        result["scenario_name"] = (
            scenario.name
        )

        result = apply_scenario(
            result,
            baseline_column=baseline_column,
        )

        return result

    # ---------------------------------------------------------
    # Run multiple scenarios
    # ---------------------------------------------------------

    def compare(
        self,
        forecast: pd.DataFrame,
        scenarios: list[ScenarioDefinition],
        baseline_column: str = "baseline_demand",
    ) -> pd.DataFrame:

        outputs = []

        for scenario in scenarios:

            output = self.run(
                forecast=forecast,
                scenario=scenario,
                baseline_column=baseline_column,
            )

            outputs.append(
                output
            )

        if not outputs:
            raise ValueError(
                "at least one scenario is required"
            )

        return pd.concat(
            outputs,
            ignore_index=True,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summarize(
        self,
        result: pd.DataFrame,
    ) -> pd.DataFrame:

        required = {
            "scenario_name",
            "baseline_demand",
            "scenario_demand",
            "absolute_change",
            "percentage_change",
        }

        missing = (
            required
            - set(result.columns)
        )

        if missing:
            raise ValueError(
                f"scenario result missing: "
                f"{sorted(missing)}"
            )

        return (
            result
            .groupby(
                "scenario_name",
                as_index=False,
            )
            .agg(
                baseline_demand=(
                    "baseline_demand",
                    "sum",
                ),
                scenario_demand=(
                    "scenario_demand",
                    "sum",
                ),
                absolute_change=(
                    "absolute_change",
                    "sum",
                ),
            )
        ).assign(
            percentage_change=lambda x:
                (
                    x["absolute_change"]
                    / x["baseline_demand"]
                    .replace(0, pd.NA)
                    * 100.0
                )
                .fillna(0.0)
            )