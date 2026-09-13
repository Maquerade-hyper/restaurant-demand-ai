from pathlib import Path

import numpy as np
import pandas as pd


SALES_PATH = Path("data/synthetic/sales.csv")
INVENTORY_PATH = Path("data/synthetic/inventory.csv")
TRUTH_PATH = Path("data/synthetic/demand_truth.csv")


def main():

    sales = pd.read_csv(SALES_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    truth = pd.read_csv(TRUTH_PATH)

    print("=" * 70)
    print("CAUSAL DEMAND VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Basic row-count integrity
    # ---------------------------------------------------------

    assert len(sales) == len(inventory)
    assert len(sales) == len(truth)

    # ---------------------------------------------------------
    # TRUE DEMAND >= OBSERVED SALES
    # ---------------------------------------------------------

    demand_violation = (
        truth["observed_sales"]
        > truth["true_demand"] + 1e-6
    )

    print(
        "TRUE DEMAND >= OBSERVED SALES :",
        not demand_violation.any(),
    )

    assert not demand_violation.any()

    # ---------------------------------------------------------
    # LOST DEMAND >= 0
    # ---------------------------------------------------------

    lost_demand_violation = (
        truth["lost_demand"] < -1e-6
    )

    print(
        "LOST DEMAND >= 0              :",
        not lost_demand_violation.any(),
    )

    assert not lost_demand_violation.any()

    # ---------------------------------------------------------
    # Accounting identity
    #
    # opening + received - sales - wastage = closing
    # ---------------------------------------------------------

    calculated_closing = (
        inventory["opening_stock"]
        + inventory["received_stock"]
        - sales["quantity_sold"]
        - inventory["wastage"]
    )

    accounting_error = np.abs(
        calculated_closing
        - inventory["closing_stock"]
    )

    max_error = float(
        accounting_error.max()
    )

    print(
        f"MAX INVENTORY ACCOUNTING ERROR : "
        f"{max_error:.6f}"
    )

    assert max_error <= 1e-3

    # ---------------------------------------------------------
    # Stockout must correspond to constrained demand.
    # ---------------------------------------------------------

    stockout_without_lost_demand = (
        inventory["stockout"]
        & (
            truth["lost_demand"]
            <= 1e-6
        )
    )

    print(
        "STOCKOUT => LOST DEMAND         :",
        not stockout_without_lost_demand.any(),
    )

    assert not stockout_without_lost_demand.any()

    # ---------------------------------------------------------
    # Stockout rate
    # ---------------------------------------------------------

    stockout_rate = float(
        inventory["stockout"].mean()
    )

    print(
        f"STOCKOUT RATE                  : "
        f"{stockout_rate:.4f}"
    )

    # We want genuine but controlled constraints.
    assert 0.01 <= stockout_rate <= 0.20

    # ---------------------------------------------------------
    # Lost demand must actually exist.
    # ---------------------------------------------------------

    total_lost_demand = float(
        truth["lost_demand"].sum()
    )

    print(
        f"TOTAL LOST DEMAND              : "
        f"{total_lost_demand:,.2f}"
    )

    assert total_lost_demand > 0

    # ---------------------------------------------------------
    # Fulfillment
    # ---------------------------------------------------------

    total_true_demand = float(
        truth["true_demand"].sum()
    )

    total_observed_sales = float(
        truth["observed_sales"].sum()
    )

    fulfillment_rate = (
        total_observed_sales
        / total_true_demand
    )

    print(
        f"FULFILLMENT RATE               : "
        f"{fulfillment_rate:.4f}"
    )

    assert 0 < fulfillment_rate <= 1

    print("=" * 70)
    print("CAUSAL DEMAND VALIDATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()