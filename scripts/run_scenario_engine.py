from __future__ import annotations

import sys
from pathlib import Path


# ==============================================================
# PROJECT ROOT
# ==============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from app.intelligence.scenario import (
    ScenarioEngineService,
)


# ==============================================================
# INPUT
# ==============================================================

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "cultural_demographic_intelligence.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "scenario_intelligence.csv"
)


# ==============================================================
# LOAD BASELINE
# ==============================================================

def load_baseline():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required input not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    if "historical_demand_reference" in df.columns:

        baseline = df[
            [
                "outlet_id",
                "product_id",
                "date",
                "historical_demand_reference",
            ]
        ].rename(
            columns={
                "historical_demand_reference":
                    "baseline_demand"
            }
        )

    elif "demand" in df.columns:

        baseline = df[
            [
                "outlet_id",
                "product_id",
                "date",
                "demand",
            ]
        ].rename(
            columns={
                "demand":
                    "baseline_demand"
            }
        )

    else:

        raise ValueError(
            "No usable baseline demand column found"
        )

    baseline["baseline_demand"] = (
        pd.to_numeric(
            baseline["baseline_demand"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    return baseline


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print(
        "PART 20 - SCENARIO ENGINE"
    )
    print("=" * 70)

    forecast = load_baseline()

    print(
        f"Baseline rows : {len(forecast):,}"
    )

    print(
        f"Outlets       : "
        f"{forecast['outlet_id'].nunique():,}"
    )

    print(
        f"Products      : "
        f"{forecast['product_id'].nunique():,}"
    )

    print()

    service = (
        ScenarioEngineService()
    )

    # ----------------------------------------------------------
    # Baseline
    # ----------------------------------------------------------

    baseline = service.create_scenario(
        name="baseline"
    )

    # ----------------------------------------------------------
    # Holiday
    # ----------------------------------------------------------

    holiday = service.create_scenario(
        name="holiday",
        holiday_multiplier=1.15,
    )

    # ----------------------------------------------------------
    # Promotion
    # ----------------------------------------------------------

    promotion = service.create_scenario(
        name="promotion",
        promotion_multiplier=1.20,
    )

    # ----------------------------------------------------------
    # Event
    # ----------------------------------------------------------

    event = service.create_scenario(
        name="major_event",
        event_multiplier=1.25,
    )

    # ----------------------------------------------------------
    # Weather
    #
    # Scenario assumption:
    # demand is 10% higher under the hypothetical weather
    # condition.
    # ----------------------------------------------------------

    weather = service.create_scenario(
        name="favorable_weather",
        weather_multiplier=1.10,
    )

    # ----------------------------------------------------------
    # Combined
    # ----------------------------------------------------------

    combined = service.create_scenario(
        name="holiday_promotion_event",
        holiday_multiplier=1.15,
        promotion_multiplier=1.20,
        event_multiplier=1.25,
    )

    scenarios = [
        baseline,
        holiday,
        promotion,
        event,
        weather,
        combined,
    ]

    result = service.analyze(
        forecast=forecast,
        scenarios=scenarios,
    )

    summary = service.summarize(
        result
    )

    validation = service.validate(
        result
    )

    print(
        "SCENARIO RESULTS"
    )

    print("-" * 70)

    for _, row in summary.iterrows():

        print(
            f"{row['scenario_name']:<30}"
            f" baseline="
            f"{row['baseline_demand']:,.2f}"
            f" scenario="
            f"{row['scenario_demand']:,.2f}"
            f" change="
            f"{row['absolute_change']:,.2f}"
            f" ({row['percentage_change']:.2f}%)"
        )

    print()

    print(
        "VALIDATION"
    )

    print("-" * 70)

    print(
        f"PASSED       : "
        f"{validation['passed']}"
    )

    print(
        f"ERRORS       : "
        f"{len(validation['errors'])}"
    )

    if validation["errors"]:

        for error in validation["errors"]:
            print(
                f"  ERROR: {error}"
            )

        raise SystemExit(
            "PART 20 VALIDATION FAILED"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        f"Output saved : {OUTPUT_PATH}"
    )

    print()

    print("=" * 70)
    print(
        "PART 20 SCENARIO ENGINE : PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()