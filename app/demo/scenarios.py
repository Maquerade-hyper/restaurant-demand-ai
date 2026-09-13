from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)


@dataclass
class DemoScenario:
    outlet_id: str
    product_id: str
    forecast_date: str
    scenario: str = "baseline"


def discover_dataset(path: Optional[Path] = None) -> Path:
    """
    Locate the Part 16/23 synthetic intelligence dataset.
    """

    candidates = []

    if path is not None:
        candidates.append(Path(path))

    candidates.extend(
        [
            DEFAULT_DATASET,
            Path("data/interim/demand_censoring_intelligence.csv"),
            Path("data/interim/demand_spike_intelligence.csv"),
            Path("data/processed/sales.csv"),
            Path("data/synthetic/sales.csv"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No suitable synthetic dataset found. Expected "
        "data/interim/demand_censoring_intelligence.csv"
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Return the first available column from a candidate list.
    """

    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def discover_demo_scenario(
    dataset_path: Optional[Path] = None,
) -> DemoScenario:
    """
    Select a deterministic but data-driven demonstration series.

    Selection favors a sufficiently long series with relatively strong
    recent demand so that the client demonstration represents a useful
    commercial case rather than a nearly empty series.
    """

    path = discover_dataset(dataset_path)

    df = pd.read_csv(
        path,
        usecols=lambda c: c in {
            "outlet_id",
            "product_id",
            "date",
            "deconstrained_demand",
            "quantity_sold",
        },
    )

    if "date" not in df.columns:
        raise ValueError("Dataset must contain date")

    demand_col = (
        "deconstrained_demand"
        if "deconstrained_demand" in df.columns
        else "quantity_sold"
    )

    if demand_col not in df.columns:
        raise ValueError(
            "Dataset must contain deconstrained_demand or quantity_sold"
        )

    if "outlet_id" not in df.columns or "product_id" not in df.columns:
        raise ValueError("Dataset must contain outlet_id and product_id")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")

    df = df.dropna(
        subset=["date", "outlet_id", "product_id", demand_col]
    )

    grouped = (
        df.groupby(["outlet_id", "product_id"], as_index=False)
        .agg(
            rows=(demand_col, "size"),
            mean_demand=(demand_col, "mean"),
            recent_demand=(
                demand_col,
                lambda s: float(s.tail(28).mean()),
            ),
        )
    )

    grouped = grouped[grouped["rows"] >= 60].copy()

    if grouped.empty:
        raise ValueError("No sufficiently long series found")

    # Favor a commercially visible series while remaining deterministic.
    grouped["selection_score"] = (
        grouped["mean_demand"] * 0.5
        + grouped["recent_demand"] * 0.5
    )

    selected = grouped.sort_values(
        ["selection_score", "outlet_id", "product_id"],
        ascending=[False, True, True],
    ).iloc[0]

    series = df[
        (df["outlet_id"].astype(str) == str(selected["outlet_id"]))
        & (df["product_id"].astype(str) == str(selected["product_id"]))
    ].sort_values("date")

    latest_date = series["date"].max()

    # The demonstration forecasts immediately after the last historical
    # observation, so there is no leakage from future synthetic data.
    forecast_date = (
        latest_date + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    return DemoScenario(
        outlet_id=str(selected["outlet_id"]),
        product_id=str(selected["product_id"]),
        forecast_date=forecast_date,
        scenario="baseline",
    )