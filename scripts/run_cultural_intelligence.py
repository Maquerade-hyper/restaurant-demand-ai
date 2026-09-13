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

from app.intelligence.cultural import (
    CulturalDemographicIntelligenceService,
)


# ==============================================================
# PATHS
# ==============================================================

SALES_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "sales.csv"
)

OUTLET_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "outlets.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "cultural_demographic_intelligence.csv"
)


# ==============================================================
# DEMAND
# ==============================================================

def load_demand():

    demand_path = (
        ROOT
        / "data"
        / "interim"
        / "demand_censoring_intelligence.csv"
    )

    if demand_path.exists():

        df = pd.read_csv(
            demand_path,
            parse_dates=["date"],
        )

        if "deconstrained_demand" in df.columns:

            return df[
                [
                    "outlet_id",
                    "product_id",
                    "date",
                    "deconstrained_demand",
                ]
            ].rename(
                columns={
                    "deconstrained_demand": "demand"
                }
            )

    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"],
    )

    return sales[
        [
            "outlet_id",
            "product_id",
            "date",
            "quantity_sold",
        ]
    ].rename(
        columns={
            "quantity_sold": "demand"
        }
    )


# ==============================================================
# CALENDAR
# ==============================================================

def build_calendar(demand):

    calendar = demand[
        [
            "outlet_id",
            "date",
        ]
    ].drop_duplicates()

    # ----------------------------------------------------------
    # We derive basic calendar signals from date only.
    #
    # Actual customer deployments should replace these with
    # authoritative holiday/religious/event feeds.
    # ----------------------------------------------------------

    calendar["holiday_importance"] = (
        (
            calendar["date"].dt.month == 12
        )
        & (
            calendar["date"].dt.day == 25
        )
    ).astype(float)

    calendar["religious_importance"] = 0.0

    calendar["event_importance"] = 0.0

    return calendar


# ==============================================================
# DEMOGRAPHICS
# ==============================================================

def build_demographics():

    if not OUTLET_PATH.exists():
        raise FileNotFoundError(
            f"Outlet file not found: {OUTLET_PATH}"
        )

    outlets = pd.read_csv(
        OUTLET_PATH
    )

    result = pd.DataFrame(
        {
            "outlet_id": outlets["outlet_id"],
        }
    )

    # ----------------------------------------------------------
    # Use available outlet context.
    #
    # These are context signals, not demographic truth.
    # ----------------------------------------------------------

    result["population_density"] = 0.0
    result["tourism_index"] = (
        pd.to_numeric(
            outlets.get(
                "tourism_level",
                0,
            ),
            errors="coerce",
        )
        .fillna(0.0)
    )

    result["student_index"] = (
        pd.to_numeric(
            outlets.get(
                "student_area",
                0,
            ),
            errors="coerce",
        )
        .fillna(0.0)
    )

    result["business_index"] = (
        pd.to_numeric(
            outlets.get(
                "business_area",
                0,
            ),
            errors="coerce",
        )
        .fillna(0.0)
    )

    result["residential_index"] = (
        pd.to_numeric(
            outlets.get(
                "residential_area",
                0,
            ),
            errors="coerce",
        )
        .fillna(0.0)
    )

    result["religious_population_share"] = 0.0
    result["young_population_share"] = 0.0
    result["working_population_share"] = 0.0

    result["dominant_religious_context"] = (
        "unknown"
    )

    result["demographic_context"] = (
        outlets.get(
            "location_type",
            "unknown",
        )
        .fillna("unknown")
        .astype(str)
    )

    result["city_class"] = "unknown"

    return result


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print(
        "PART 19 - CULTURAL & DEMOGRAPHIC INTELLIGENCE"
    )
    print("=" * 70)

    demand = load_demand()

    calendar = build_calendar(
        demand
    )

    demographics = build_demographics()

    print(
        f"Demand rows       : {len(demand):,}"
    )

    print(
        f"Calendar rows     : {len(calendar):,}"
    )

    print(
        f"Demographic rows  : {len(demographics):,}"
    )

    print()

    service = (
        CulturalDemographicIntelligenceService()
    )

    result = service.analyze(
        demand=demand,
        calendar=calendar,
        demographics=demographics,
    )

    validation = service.validate(
        result
    )

    print("CULTURAL INTELLIGENCE RESULTS")
    print("-" * 70)

    print(
        f"Output rows               : "
        f"{len(result):,}"
    )

    print(
        f"Outlets covered           : "
        f"{result['outlet_id'].nunique():,}"
    )

    print(
        f"Products covered          : "
        f"{result['product_id'].nunique():,}"
    )

    print(
        f"Holiday-active rows       : "
        f"{int((result['holiday_importance'] > 0).sum()):,}"
    )

    print(
        f"Religious-context rows    : "
        f"{int((result['religious_importance'] > 0).sum()):,}"
    )

    print(
        f"Event-active rows         : "
        f"{int((result['event_importance'] > 0).sum()):,}"
    )

    print(
        f"Average cultural score   : "
        f"{result['cultural_context_score'].mean():.6f}"
    )

    print(
        f"Average cultural pressure: "
        f"{result['cultural_demand_pressure'].mean():.4f}"
    )

    print()

    print("VALIDATION")
    print("-" * 70)

    print(
        f"PASSED                    : "
        f"{validation['passed']}"
    )

    print(
        f"ERRORS                    : "
        f"{len(validation['errors'])}"
    )

    if validation["errors"]:

        for error in validation["errors"]:
            print(
                f"  ERROR: {error}"
            )

        raise SystemExit(
            "PART 19 VALIDATION FAILED"
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
        f"Output saved              : "
        f"{OUTPUT_PATH}"
    )

    print()

    print("=" * 70)
    print(
        "PART 19 CULTURAL & DEMOGRAPHIC INTELLIGENCE : PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()