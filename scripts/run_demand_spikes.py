from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from app.intelligence.spikes.service import (
    DemandSpikeIntelligenceService,
)


# ============================================================
# PATHS
# ============================================================

CENSORED_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)

SALES_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "sales.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_spike_intelligence.csv"
)


# ============================================================
# DATA LOADER
# ============================================================

def load_input() -> tuple[pd.DataFrame, str]:

    if CENSORED_PATH.exists():

        df = pd.read_csv(
            CENSORED_PATH,
            parse_dates=["date"],
        )

        if "deconstrained_demand" in df.columns:

            return (
                df,
                "deconstrained_demand",
            )

    if not SALES_PATH.exists():

        raise FileNotFoundError(
            "Neither Part 16 demand intelligence "
            "nor synthetic sales data was found."
        )

    df = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"],
    )

    return (
        df,
        "quantity_sold",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PART 23 - DEMAND SPIKE INTELLIGENCE")
    print("=" * 70)

    df, demand_column = load_input()

    print(
        f"Input demand column: {demand_column}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Outlets: "
        f"{df['outlet_id'].nunique():,}"
    )

    print(
        f"Products: "
        f"{df['product_id'].nunique():,}"
    )

    service = DemandSpikeIntelligenceService(
        demand_column=demand_column,
        baseline_window=28,
        minimum_history=7,
    )

    result = service.transform(
        df
    )

    validation = service.validate(
        result
    )

    summary = service.summary(
        result
    )

    print()
    print("SPIKE SUMMARY")
    print("-" * 70)

    print(
        f"Rows: {summary['rows']:,}"
    )

    print(
        f"Spikes: {summary['spikes']:,}"
    )

    print(
        f"Spike rate: "
        f"{summary['spike_rate']:.4f}"
    )

    print(
        f"Moderate: "
        f"{summary['moderate_spikes']:,}"
    )

    print(
        f"Major: "
        f"{summary['major_spikes']:,}"
    )

    print(
        f"Extreme: "
        f"{summary['extreme_spikes']:,}"
    )

    print(
        f"Persistent: "
        f"{summary['persistent_spikes']:,}"
    )

    print(
        f"Isolated: "
        f"{summary['isolated_spikes']:,}"
    )

    print()
    print("VALIDATION")
    print("-" * 70)

    print(
        f"PASSED: {validation['passed']}"
    )

    print(
        f"ERRORS: "
        f"{len(validation['errors'])}"
    )

    for error in validation["errors"][:20]:

        print(
            f" - {error}"
        )

    if not validation["passed"]:

        raise RuntimeError(
            "PART 23 VALIDATION FAILED"
        )

    # ========================================================
    # SAVE
    # ========================================================

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
        f"Output saved: {OUTPUT_PATH}"
    )

    print()
    print(
        "PART 23 PASSED"
    )


if __name__ == "__main__":
    main()