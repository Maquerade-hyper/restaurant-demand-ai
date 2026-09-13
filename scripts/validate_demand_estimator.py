from pathlib import Path

import numpy as np
import pandas as pd

from app.intelligence.demand import DemandIntelligenceService


SALES_PATH = Path("data/synthetic/sales.csv")
INVENTORY_PATH = Path("data/synthetic/inventory.csv")
TRUTH_PATH = Path("data/synthetic/demand_truth.csv")


def main():

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    sales = pd.read_csv(SALES_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    truth = pd.read_csv(TRUTH_PATH)

    # Normalize dates
    sales["date"] = pd.to_datetime(sales["date"])
    inventory["date"] = pd.to_datetime(inventory["date"])
    truth["date"] = pd.to_datetime(truth["date"])

    # ---------------------------------------------------------
    # Run demand estimator
    # ---------------------------------------------------------

    service = DemandIntelligenceService(
        lookback_days=28,
        min_history=7,
    )

    estimated = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    # ---------------------------------------------------------
    # Merge estimator output with causal ground truth
    # ---------------------------------------------------------

    comparison = estimated.merge(
        truth,
        on=["outlet_id", "product_id", "date"],
        how="inner",
        suffixes=("_estimated", "_truth"),
    )

    if comparison.empty:
        raise RuntimeError(
            "No matching rows between estimator output and demand truth."
        )

    # ---------------------------------------------------------
    # Estimator error
    # ---------------------------------------------------------

    error = (
        comparison["true_demand_estimated"]
        - comparison["true_demand_truth"]
    )

    absolute_error = error.abs()

    mae = float(
        absolute_error.mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error.to_numpy() ** 2
            )
        )
    )

    bias = float(
        error.mean()
    )

    # ---------------------------------------------------------
    # Stockout-only performance
    # ---------------------------------------------------------

    stockout_mask = (
        comparison["stockout_truth"].astype(bool)
    )

    stockout_count = int(
        stockout_mask.sum()
    )

    if stockout_count > 0:

        stockout_error = error[
            stockout_mask
        ]

        stockout_mae = float(
            stockout_error.abs().mean()
        )

        stockout_bias = float(
            stockout_error.mean()
        )

    else:

        stockout_mae = 0.0
        stockout_bias = 0.0

    # ---------------------------------------------------------
    # Lost-demand estimation
    # ---------------------------------------------------------

    lost_demand_error = (
        comparison["lost_demand_estimated"]
        - comparison["lost_demand_truth"]
    )

    lost_demand_mae = float(
        lost_demand_error.abs().mean()
    )

    estimated_lost_demand = float(
        comparison["lost_demand_estimated"].sum()
    )

    actual_lost_demand = float(
        comparison["lost_demand_truth"].sum()
    )

    # ---------------------------------------------------------
    # Constraint correctness
    # ---------------------------------------------------------

    true_constrained = (
        comparison["lost_demand_truth"] > 0
    )

    estimated_constrained = (
        comparison["lost_demand_estimated"] > 0
    )

    constraint_accuracy = float(
        (
            true_constrained
            == estimated_constrained
        ).mean()
    )

    # ---------------------------------------------------------
    # Demand recovery ratio
    # ---------------------------------------------------------

    true_demand_total = float(
        comparison["true_demand_truth"].sum()
    )

    estimated_demand_total = float(
        comparison["true_demand_estimated"].sum()
    )

    if true_demand_total > 0:

        demand_recovery_ratio = (
            estimated_demand_total
            / true_demand_total
        )

    else:

        demand_recovery_ratio = 0.0

    # ---------------------------------------------------------
    # Lost-demand recovery ratio
    # ---------------------------------------------------------

    if actual_lost_demand > 0:

        lost_demand_recovery_ratio = (
            estimated_lost_demand
            / actual_lost_demand
        )

    else:

        lost_demand_recovery_ratio = 0.0

    # ---------------------------------------------------------
    # Fulfillment comparison
    # ---------------------------------------------------------

    true_fulfillment = np.where(
        comparison["true_demand_truth"] > 0,
        comparison["observed_sales"]
        / comparison["true_demand_truth"],
        1.0,
    )

    estimated_fulfillment = (
        comparison["fulfillment_rate"]
        .to_numpy()
    )

    fulfillment_error = (
        estimated_fulfillment
        - true_fulfillment
    )

    fulfillment_mae = float(
        np.abs(fulfillment_error).mean()
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("=" * 70)
    print("DEMAND ESTIMATOR VALIDATION")
    print("=" * 70)

    print(
        f"Comparison rows          : "
        f"{len(comparison):,}"
    )

    print(
        f"True demand              : "
        f"{true_demand_total:,.2f}"
    )

    print(
        f"Estimated true demand    : "
        f"{estimated_demand_total:,.2f}"
    )

    print(
        f"Demand recovery ratio    : "
        f"{demand_recovery_ratio:.4f}"
    )

    print(
        f"Demand MAE               : "
        f"{mae:.4f}"
    )

    print(
        f"Demand RMSE              : "
        f"{rmse:.4f}"
    )

    print(
        f"Demand bias              : "
        f"{bias:.4f}"
    )

    print(
        f"Stockout records         : "
        f"{stockout_count:,}"
    )

    print(
        f"Stockout demand MAE      : "
        f"{stockout_mae:.4f}"
    )

    print(
        f"Stockout demand bias     : "
        f"{stockout_bias:.4f}"
    )

    print(
        f"Actual lost demand       : "
        f"{actual_lost_demand:,.2f}"
    )

    print(
        f"Estimated lost demand    : "
        f"{estimated_lost_demand:,.2f}"
    )

    print(
        f"Lost-demand recovery     : "
        f"{lost_demand_recovery_ratio:.4f}"
    )

    print(
        f"Lost demand MAE          : "
        f"{lost_demand_mae:.4f}"
    )

    print(
        f"Constraint accuracy      : "
        f"{constraint_accuracy:.4f}"
    )

    print(
        f"Fulfillment MAE          : "
        f"{fulfillment_mae:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()