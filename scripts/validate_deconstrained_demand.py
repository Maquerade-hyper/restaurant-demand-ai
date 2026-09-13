from pathlib import Path

import numpy as np
import pandas as pd

from app.intelligence.demand.deconstraining import (
    DemandDeconstrainingService,
)


SALES_PATH = Path(
    "data/synthetic/sales.csv"
)

INVENTORY_PATH = Path(
    "data/synthetic/inventory.csv"
)

TRUTH_PATH = Path(
    "data/synthetic/demand_truth.csv"
)


def mae(actual, predicted):

    return float(
        np.abs(
            np.asarray(predicted)
            - np.asarray(actual)
        ).mean()
    )


def rmse(actual, predicted):

    error = (
        np.asarray(predicted)
        - np.asarray(actual)
    )

    return float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )


def main():

    # ---------------------------------------------------------
    # Load operational data
    # ---------------------------------------------------------

    sales = pd.read_csv(
        SALES_PATH
    )

    inventory = pd.read_csv(
        INVENTORY_PATH
    )

    truth = pd.read_csv(
        TRUTH_PATH
    )

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    inventory["date"] = pd.to_datetime(
        inventory["date"]
    )

    truth["date"] = pd.to_datetime(
        truth["date"]
    )

    # ---------------------------------------------------------
    # Run estimator
    #
    # IMPORTANT:
    # truth is NOT passed to the estimator.
    # ---------------------------------------------------------

    service = DemandDeconstrainingService(
        lookback_days=56,
        minimum_clean_days=7,
    )

    estimated = service.analyze(
        sales=sales,
        inventory=inventory,
    )

    # ---------------------------------------------------------
    # Compare against truth
    #
    # Truth is validation-only.
    # ---------------------------------------------------------

    comparison = estimated.merge(
        truth[
            [
                "outlet_id",
                "product_id",
                "date",
                "true_demand",
                "observed_sales",
                "lost_demand",
                "stockout",
            ]
        ],
        on=[
            "outlet_id",
            "product_id",
            "date",
        ],
        how="inner",
        suffixes=(
            "_estimated",
            "_truth",
        ),
    )

    if comparison.empty:

        raise RuntimeError(
            "No matching validation rows."
        )

    # ---------------------------------------------------------
    # True demand error
    # ---------------------------------------------------------

    demand_error = (
        comparison[
            "deconstrained_demand"
        ]
        - comparison[
            "true_demand"
        ]
    )

    demand_mae = float(
        demand_error.abs().mean()
    )

    demand_rmse = float(
        np.sqrt(
            np.mean(
                demand_error.to_numpy()
                ** 2
            )
        )
    )

    demand_bias = float(
        demand_error.mean()
    )

    # ---------------------------------------------------------
    # Stockout-only demand error
    # ---------------------------------------------------------

    stockout_mask = (
        comparison[
            "stockout_truth"
        ].astype(bool)
    )

    stockout_count = int(
        stockout_mask.sum()
    )

    if stockout_count > 0:

        stockout_error = (
            demand_error[
                stockout_mask
            ]
        )

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
    # Lost demand
    # ---------------------------------------------------------

    lost_error = (
        comparison[
            "estimated_lost_demand"
        ]
        - comparison[
            "lost_demand"
        ]
    )

    lost_mae = float(
        lost_error.abs().mean()
    )

    actual_lost = float(
        comparison[
            "lost_demand"
        ].sum()
    )

    estimated_lost = float(
        comparison[
            "estimated_lost_demand"
        ].sum()
    )

    if actual_lost > 0:

        lost_recovery = (
            estimated_lost
            / actual_lost
        )

    else:

        lost_recovery = 0.0

    # ---------------------------------------------------------
    # Constraint detection
    # ---------------------------------------------------------

    true_constrained = (
        comparison[
            "lost_demand"
        ] > 0
    )

    estimated_constrained = (
        comparison[
            "estimated_lost_demand"
        ] > 0
    )

    constraint_accuracy = float(
        (
            true_constrained
            == estimated_constrained
        ).mean()
    )

    # ---------------------------------------------------------
    # Aggregate demand recovery
    # ---------------------------------------------------------

    actual_demand = float(
        comparison[
            "true_demand"
        ].sum()
    )

    estimated_demand = float(
        comparison[
            "deconstrained_demand"
        ].sum()
    )

    if actual_demand > 0:

        demand_recovery = (
            estimated_demand
            / actual_demand
        )

    else:

        demand_recovery = 0.0

    # ---------------------------------------------------------
    # Fulfillment
    # ---------------------------------------------------------

    true_fulfillment = np.where(
        comparison[
            "true_demand"
        ] > 0,
        comparison[
            "observed_sales"
        ]
        / comparison[
            "true_demand"
        ],
        1.0,
    )

    estimated_fulfillment = (
        comparison[
            "estimated_fulfillment_rate"
        ].to_numpy()
    )

    fulfillment_mae = float(
        np.abs(
            estimated_fulfillment
            - true_fulfillment
        ).mean()
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("=" * 72)
    print(
        "PART 16B - DECONSTRAINED DEMAND VALIDATION"
    )
    print("=" * 72)

    print(
        f"Comparison rows          : "
        f"{len(comparison):,}"
    )

    print(
        f"True demand              : "
        f"{actual_demand:,.2f}"
    )

    print(
        f"Estimated demand         : "
        f"{estimated_demand:,.2f}"
    )

    print(
        f"Demand recovery          : "
        f"{demand_recovery:.4f}"
    )

    print(
        f"Demand MAE               : "
        f"{demand_mae:.4f}"
    )

    print(
        f"Demand RMSE              : "
        f"{demand_rmse:.4f}"
    )

    print(
        f"Demand bias              : "
        f"{demand_bias:.4f}"
    )

    print("-" * 72)

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

    print("-" * 72)

    print(
        f"Actual lost demand       : "
        f"{actual_lost:,.2f}"
    )

    print(
        f"Estimated lost demand    : "
        f"{estimated_lost:,.2f}"
    )

    print(
        f"Lost-demand recovery     : "
        f"{lost_recovery:.4f}"
    )

    print(
        f"Lost-demand MAE          : "
        f"{lost_mae:.4f}"
    )

    print("-" * 72)

    print(
        f"Constraint accuracy      : "
        f"{constraint_accuracy:.4f}"
    )

    print(
        f"Fulfillment MAE          : "
        f"{fulfillment_mae:.4f}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()