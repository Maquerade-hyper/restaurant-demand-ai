from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.demo.formatter import format_client_report
from app.demo.schemas import ClientDemoRequest
from app.demo.service import ClientDemoService


# ============================================================================
# PATHS
# ============================================================================

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

DATASET = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)


# ============================================================================
# CONSTANTS
# ============================================================================

DEMAND_COLUMN = "deconstrained_demand"

MIN_HISTORY_DAYS = 35

RANDOM_SEED = 42


# ============================================================================
# DATA PREPARATION
# ============================================================================


def load_demo_data() -> pd.DataFrame:
    """
    Load the synthetic demonstration dataset and normalize the minimum
    columns required for scenario discovery.
    """

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Demo dataset not found:\n{DATASET}"
        )

    df = pd.read_csv(DATASET)

    required = {
        "date",
        "outlet_id",
        "product_id",
        DEMAND_COLUMN,
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Demo dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df[DEMAND_COLUMN] = pd.to_numeric(
        df[DEMAND_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "outlet_id",
            "product_id",
            DEMAND_COLUMN,
        ]
    ).copy()

    df = df.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================================
# SERIES SUMMARY
# ============================================================================


def build_series_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one summary row per outlet/product series.

    Scenario selection is based on actual historical data.
    """

    grouped = (
        df.groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        )
    )

    summary = grouped[DEMAND_COLUMN].agg(
        mean_demand="mean",
        std_demand="std",
        max_demand="max",
        min_demand="min",
        observations="count",
    ).reset_index()

    summary["std_demand"] = summary[
        "std_demand"
    ].fillna(0.0)

    summary["cv"] = (
        summary["std_demand"]
        / summary["mean_demand"].clip(lower=1e-6)
    )

    # ------------------------------------------------------------------------
    # Recent vs historical demand
    # ------------------------------------------------------------------------

    recent = (
        df.sort_values("date")
        .groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        )
        .tail(28)
        .groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        )[DEMAND_COLUMN]
        .mean()
        .rename("recent_mean")
        .reset_index()
    )

    historical = (
        df.sort_values("date")
        .groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        )
        .head(-28)
        .groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        )[DEMAND_COLUMN]
        .mean()
        .rename("historical_mean")
        .reset_index()
    )

    summary = summary.merge(
        recent,
        on=[
            "outlet_id",
            "product_id",
        ],
        how="left",
    )

    summary = summary.merge(
        historical,
        on=[
            "outlet_id",
            "product_id",
        ],
        how="left",
    )

    summary["recent_mean"] = summary[
        "recent_mean"
    ].fillna(summary["mean_demand"])

    summary["historical_mean"] = summary[
        "historical_mean"
    ].fillna(summary["mean_demand"])

    summary["recent_growth"] = (
        (
            summary["recent_mean"]
            - summary["historical_mean"]
        )
        / summary["historical_mean"].clip(lower=1e-6)
    )

    # ------------------------------------------------------------------------
    # Recent spike score
    # ------------------------------------------------------------------------

    recent_max = (
        df.sort_values("date")
        .groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        )
        .tail(28)
        .groupby(
            [
                "outlet_id",
                "product_id",
            ],
            sort=False,
        )[DEMAND_COLUMN]
        .max()
        .rename("recent_max")
        .reset_index()
    )

    summary = summary.merge(
        recent_max,
        on=[
            "outlet_id",
            "product_id",
        ],
        how="left",
    )

    summary["spike_ratio"] = (
        summary["recent_max"]
        / summary["historical_mean"].clip(lower=1e-6)
    )

    # ------------------------------------------------------------------------
    # Optional inventory information
    # ------------------------------------------------------------------------

    inventory_columns = [
        column
        for column in [
            "closing_stock",
            "inventory_position",
            "stockout",
            "stockout_event",
            "censor",
            "lost_demand",
        ]
        if column in df.columns
    ]

    if inventory_columns:
        inventory = (
            df.sort_values("date")
            .groupby(
                [
                    "outlet_id",
                    "product_id",
                ],
                sort=False,
            )
            .tail(28)
            .groupby(
                [
                    "outlet_id",
                    "product_id",
                ],
                sort=False,
            )
            .agg(
                recent_stockout=(
                    "stockout"
                    if "stockout" in inventory_columns
                    else inventory_columns[0],
                    "mean",
                ),
            )
            .reset_index()
        )

        summary = summary.merge(
            inventory,
            on=[
                "outlet_id",
                "product_id",
            ],
            how="left",
        )

    else:
        summary["recent_stockout"] = 0.0

    summary["recent_stockout"] = (
        pd.to_numeric(
            summary["recent_stockout"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    return summary


# ============================================================================
# SCENARIO SELECTION
# ============================================================================


def select_normal_scenario(
    summary: pd.DataFrame,
) -> pd.Series:
    """
    Select a stable, representative demand series.
    """

    candidates = summary[
        (summary["observations"] >= MIN_HISTORY_DAYS)
        & (summary["cv"] >= 0.05)
        & (summary["cv"] <= 0.35)
        & (summary["recent_growth"].abs() <= 0.10)
        & (summary["recent_stockout"] <= 0.05)
    ]

    if candidates.empty:
        candidates = summary[
            summary["observations"] >= MIN_HISTORY_DAYS
        ]

    candidates = candidates.copy()

    candidates["score"] = (
        candidates["cv"].sub(0.18).abs()
        + candidates["recent_growth"].abs()
    )

    return candidates.sort_values("score").iloc[0]


def select_spike_scenario(
    summary: pd.DataFrame,
) -> pd.Series:
    """
    Select a series with unusually high recent demand.
    """

    candidates = summary[
        (summary["observations"] >= MIN_HISTORY_DAYS)
        & (summary["spike_ratio"] >= 1.25)
    ]

    if candidates.empty:
        candidates = summary[
            summary["observations"] >= MIN_HISTORY_DAYS
        ]

    return candidates.sort_values(
        [
            "spike_ratio",
            "recent_max",
        ],
        ascending=False,
    ).iloc[0]


def select_shortage_scenario(
    summary: pd.DataFrame,
) -> pd.Series:
    """
    Select a series with historical stockout/inventory pressure.
    """

    candidates = summary[
        (summary["observations"] >= MIN_HISTORY_DAYS)
        & (summary["recent_stockout"] > 0)
    ]

    if candidates.empty:
        candidates = summary[
            summary["observations"] >= MIN_HISTORY_DAYS
        ].copy()

        # Fallback to volatile/high-demand series.
        candidates["fallback_score"] = (
            candidates["cv"]
            + candidates["recent_growth"].clip(lower=0)
        )

        return candidates.sort_values(
            "fallback_score",
            ascending=False,
        ).iloc[0]

    return candidates.sort_values(
        "recent_stockout",
        ascending=False,
    ).iloc[0]


def select_growth_scenario(
    summary: pd.DataFrame,
) -> pd.Series:
    """
    Select a series whose recent demand is materially above
    its historical level.
    """

    candidates = summary[
        (summary["observations"] >= MIN_HISTORY_DAYS)
        & (summary["recent_growth"] >= 0.15)
    ]

    if candidates.empty:
        candidates = summary[
            summary["observations"] >= MIN_HISTORY_DAYS
        ]

    return candidates.sort_values(
        "recent_growth",
        ascending=False,
    ).iloc[0]


def select_low_demand_scenario(
    summary: pd.DataFrame,
) -> pd.Series:
    """
    Select a relatively low-volume series.
    """

    candidates = summary[
        summary["observations"] >= MIN_HISTORY_DAYS
    ]

    return candidates.sort_values(
        "mean_demand",
        ascending=True,
    ).iloc[0]


def select_random_scenario(
    summary: pd.DataFrame,
) -> pd.Series:

    candidates = summary[
        summary["observations"] >= MIN_HISTORY_DAYS
    ]

    rng = np.random.default_rng(RANDOM_SEED)

    index = rng.integers(
        0,
        len(candidates),
    )

    return candidates.iloc[index]


# ============================================================================
# SCENARIO CATALOG
# ============================================================================


SCENARIOS = {
    "1": {
        "name": "NORMAL / STABLE DEMAND",
        "selector": select_normal_scenario,
        "scenario": "baseline",
    },
    "2": {
        "name": "HIGH DEMAND / DEMAND SPIKE",
        "selector": select_spike_scenario,
        "scenario": "baseline",
    },
    "3": {
        "name": "INVENTORY SHORTAGE / STOCKOUT RISK",
        "selector": select_shortage_scenario,
        "scenario": "baseline",
    },
    "4": {
        "name": "GROWING DEMAND",
        "selector": select_growth_scenario,
        "scenario": "baseline",
    },
    "5": {
        "name": "LOW DEMAND",
        "selector": select_low_demand_scenario,
        "scenario": "baseline",
    },
    "6": {
        "name": "RANDOM OUTLET / PRODUCT",
        "selector": select_random_scenario,
        "scenario": "baseline",
    },
}


# ============================================================================
# FORECAST DATE
# ============================================================================


def get_forecast_date(
    df: pd.DataFrame,
    outlet_id: str,
    product_id: str,
) -> pd.Timestamp:

    series = df[
        (df["outlet_id"] == outlet_id)
        & (df["product_id"] == product_id)
    ].sort_values("date")

    if series.empty:
        raise ValueError(
            f"No data found for {outlet_id}/{product_id}"
        )

    last_date = series["date"].max()

    return last_date + pd.Timedelta(days=1)


# ============================================================================

# RUN ONE SCENARIO
# ============================================================================


def run_one_scenario(
    service: ClientDemoService,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    scenario_key: str,
    scenario_number: int,
    total_scenarios: int,
) -> dict:

    config = SCENARIOS[scenario_key]
    selected = config["selector"](summary)

    outlet_id = str(selected["outlet_id"])
    product_id = str(selected["product_id"])

    forecast_date = get_forecast_date(
        df,
        outlet_id,
        product_id,
    )

    request = ClientDemoRequest(
        outlet_id=outlet_id,
        product_id=product_id,
        forecast_date=forecast_date,
        horizon_days=7,
        scenario=config["scenario"],
        lead_time_days=3,
    )

    print()
    print("=" * 76)
    print(
        f"SCENARIO {scenario_number}/{total_scenarios}: "
        f"{config['name']}"
    )
    print("=" * 76)
    print(f"Outlet       : {outlet_id}")
    print(f"Product      : {product_id}")
    print(
        f"Forecast Date: {forecast_date.strftime('%Y-%m-%d')}"
    )

    print()
    print("Running intelligence pipeline:")

    steps = [
        "Historical demand analysis",
        "D+1 / D+3 / D+7 forecasting",
        "Demand intelligence",
        "Inventory intelligence",
        "Supply recommendation",
        "Autonomous commercial decision",
    ]

    start = time.perf_counter()
    result = service.run(request)
    runtime = time.perf_counter() - start

    for index, step in enumerate(steps, start=1):
        print(f"  [{index}/6] {step}")

    validation = service.validate_result(result)

    if not validation["passed"]:
        raise RuntimeError(
            "Scenario validation failed:\n"
            + json.dumps(validation, indent=2, default=str)
        )

    payload = result.to_dict()
    payload["scenario_name"] = config["name"]
    payload["scenario_key"] = scenario_key
    payload["runtime_seconds"] = runtime
    payload["validation"] = validation

    print()
    print(format_client_report(payload))
    print()
    print(f"Scenario runtime: {runtime:.3f} sec")

    return payload


# ============================================================================
# SAVE OUTPUT
# ============================================================================


def save_output(payload, filename: str) -> Path:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / filename

    output_file.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    return output_file


# ============================================================================
# MENU
# ============================================================================


def print_menu() -> None:

    print()
    print("=" * 76)
    print("RESTAURANT DEMAND AI - CLIENT SCENARIO DEMONSTRATION")
    print("=" * 76)
    print()
    print("Select a real data-driven business scenario:")
    print()

    for key, config in SCENARIOS.items():
        print(f"  [{key}] {config['name']}")

    print("  [7] RUN ALL SCENARIOS")
    print("  [0] EXIT")
    print()


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:

    print()
    print("Loading Restaurant Demand AI client demonstration...")

    try:
        df = load_demo_data()

        print(f"Dataset: {DATASET}")
        print(f"Rows loaded: {len(df):,}")
        print(f"Demand column: {DEMAND_COLUMN}")

        print("Building data-driven scenario catalog...")

        summary = build_series_summary(df)

        print(
            f"Available outlet/product series: "
            f"{len(summary):,}"
        )

        service = ClientDemoService(dataset_path=DATASET)

    except Exception as exc:

        print()
        print("DEMO INITIALIZATION FAILED")
        print(str(exc))
        return 1

    print_menu()

    choice = input("Enter scenario number: ").strip()

    if choice == "0":
        print()
        print("Demo cancelled.")
        return 0

    if choice == "7":

        results = []
        total = len(SCENARIOS)

        for index, key in enumerate(
            SCENARIOS.keys(),
            start=1,
        ):

            try:

                payload = run_one_scenario(
                    service=service,
                    df=df,
                    summary=summary,
                    scenario_key=key,
                    scenario_number=index,
                    total_scenarios=total,
                )

                results.append(payload)

            except Exception as exc:

                print()
                print(f"[FAILED] Scenario {key}: {exc}")

        output = {
            "demo_type": "scenario_based_client_demonstration",
            "dataset": str(DATASET),
            "source_status": "SYNTHETIC DEMONSTRATION DATA",
            "scenarios_requested": total,
            "scenarios_completed": len(results),
            "results": results,
        }

        output_file = save_output(
            output,
            "client_demo_scenarios.json",
        )

        print()
        print("=" * 76)
        print("SCENARIO DEMONSTRATION COMPLETE")
        print("=" * 76)
        print(f"Completed: {len(results)}/{total}")
        print(f"Output: {output_file}")

        if len(results) == total:
            print("\nPART A SCENARIO DEMONSTRATION: PASS")
            return 0

        print("\nPART A SCENARIO DEMONSTRATION: PARTIAL")
        return 1

    if choice not in SCENARIOS:

        print()
        print("Invalid selection.")
        return 1

    try:

        payload = run_one_scenario(
            service=service,
            df=df,
            summary=summary,
            scenario_key=choice,
            scenario_number=1,
            total_scenarios=1,
        )

        output_file = save_output(
            payload,
            "client_demo_output.json",
        )

        print()
        print("=" * 76)
        print("CLIENT DEMONSTRATION COMPLETE")
        print("=" * 76)
        print(f"Output saved: {output_file}")
        print("Source: SYNTHETIC DEMONSTRATION DATA")
        print("Forecast: DATA-DRIVEN")
        print("\nPART A CLIENT SCENARIO DEMONSTRATION: PASS")

        return 0

    except Exception as exc:

        print()
        print("=" * 76)
        print("CLIENT DEMONSTRATION FAILED")
        print("=" * 76)
        print(str(exc))
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    raise SystemExit(main())
