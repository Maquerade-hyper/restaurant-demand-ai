from pathlib import Path

import pandas as pd

from app.intelligence.demand.deconstraining.service import (
    DemandDeconstrainingService,
)
from app.intelligence.demand.validation.service import (
    DemandIntelligenceValidationService,
)


ROOT = Path(__file__).resolve().parents[1]

SALES_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "sales.csv"
)

INVENTORY_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "inventory.csv"
)

TRUTH_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "demand_truth.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_intelligence_validation.csv"
)


def main() -> None:

    print("=" * 72)
    print("PART 16E - DEMAND INTELLIGENCE VALIDATION")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------------
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

    print(
        f"Sales rows              : {len(sales):,}"
    )

    print(
        f"Inventory rows          : "
        f"{len(inventory):,}"
    )

    print(
        f"Truth rows              : "
        f"{len(truth):,}"
    )

    # ---------------------------------------------------------------
    # 2. GENERATE DECONSTRAINED DEMAND
    #
    # Truth is deliberately NOT supplied to the estimator.
    # Truth is only used afterwards for validation.
    # ---------------------------------------------------------------
    deconstraining_service = (
        DemandDeconstrainingService()
    )

    intelligence = (
        deconstraining_service.analyze(
            sales=sales,
            inventory=inventory,
        )
    )

    print(
        f"Deconstrained rows      : "
        f"{len(intelligence):,}"
    )

    # ---------------------------------------------------------------
    # 3. LOAD 16D OUTPUT IF AVAILABLE
    #
    # 16D contains censoring/recovery intelligence.
    # ---------------------------------------------------------------
    censoring_path = (
        ROOT
        / "data"
        / "interim"
        / "demand_censoring_intelligence.csv"
    )

    if censoring_path.exists():

        censoring = pd.read_csv(
            censoring_path
        )

        censoring["date"] = pd.to_datetime(
            censoring["date"]
        )

        intelligence = intelligence.merge(
            censoring[
                [
                    "outlet_id",
                    "product_id",
                    "date",
                    "censoring_strength",
                    "censoring_class",
                    "censored_demand_flag",
                    "recovery_event",
                ]
            ],
            on=[
                "outlet_id",
                "product_id",
                "date",
            ],
            how="left",
            validate="one_to_one",
        )

    else:

        intelligence[
            "censoring_strength"
        ] = 0.0

        intelligence[
            "censoring_class"
        ] = "not_censored"

        intelligence[
            "censored_demand_flag"
        ] = False

        intelligence[
            "recovery_event"
        ] = False

    # ---------------------------------------------------------------
    # 4. VALIDATE
    # ---------------------------------------------------------------
    service = (
        DemandIntelligenceValidationService()
    )

    report = service.validate(
        intelligence=intelligence,
        truth=truth,
    )

    acceptance = (
        service.acceptance_check(
            report
        )
    )

    # ---------------------------------------------------------------
    # 5. PRINT RESULTS
    # ---------------------------------------------------------------
    print()
    print("PART 16E VALIDATION")
    print("-" * 72)

    overall = report["overall"]

    print(
        f"Rows compared           : "
        f"{overall['rows_compared']:,}"
    )

    print(
        f"Demand MAE              : "
        f"{overall['demand_mae']:.4f}"
    )

    print(
        f"Demand RMSE             : "
        f"{overall['demand_rmse']:.4f}"
    )

    print(
        f"Demand bias             : "
        f"{overall['demand_bias']:.4f}"
    )

    print(
        f"Demand recovery         : "
        f"{overall['demand_recovery']:.4f}"
    )

    print(
        f"Lost-demand MAE         : "
        f"{overall['lost_demand_mae']:.4f}"
    )

    print(
        f"Lost-demand RMSE        : "
        f"{overall['lost_demand_rmse']:.4f}"
    )

    print(
        f"Lost-demand bias        : "
        f"{overall['lost_demand_bias']:.4f}"
    )

    print(
        f"Lost-demand recovery    : "
        f"{overall['lost_demand_recovery']:.4f}"
    )

    print(
        f"Constraint accuracy     : "
        f"{overall['constraint_accuracy']:.4f}"
    )

    print(
        f"Fulfillment MAE         : "
        f"{overall['fulfillment_mae']:.4f}"
    )

    stockout = report["stockout"]

    print()
    print("STOCKOUT VALIDATION")
    print("-" * 72)

    print(
        f"Stockout rows           : "
        f"{stockout['stockout_rows']:,}"
    )

    print(
        f"Stockout demand MAE     : "
        f"{stockout['stockout_demand_mae']:.4f}"
    )

    print(
        f"Stockout demand RMSE    : "
        f"{stockout['stockout_demand_rmse']:.4f}"
    )

    print(
        f"Stockout demand bias    : "
        f"{stockout['stockout_demand_bias']:.4f}"
    )

    print(
        f"Stockout lost-demand MAE: "
        f"{stockout['stockout_lost_demand_mae']:.4f}"
    )

    print(
        f"Stockout lost recovery  : "
        f"{stockout['stockout_lost_demand_recovery']:.4f}"
    )

    # ---------------------------------------------------------------
    # 6. INTEGRITY
    # ---------------------------------------------------------------
    print()
    print("INTEGRITY")
    print("-" * 72)

    truth_check = report[
        "truth_validation"
    ]

    output_check = report[
        "output_validation"
    ]

    print(
        "TRUE DEMAND >= OBSERVED : "
        f"{truth_check['true_demand_ge_observed']}"
    )

    print(
        "LOST DEMAND >= 0        : "
        f"{truth_check['lost_demand_nonnegative']}"
    )

    print(
        "MAX TRUTH ACCOUNT ERROR : "
        f"{truth_check['max_accounting_error']:.6f}"
    )

    print(
        "OUTPUT DEMAND >= SALES  : "
        f"{output_check['demand_below_observed'] == 0}"
    )

    print(
        "OUTPUT LOST DEMAND >= 0 : "
        f"{output_check['negative_lost_demand'] == 0}"
    )

    print(
        "CENSORING STRENGTH VALID: "
        f"{output_check['invalid_censoring_strength'] == 0}"
    )

    # ---------------------------------------------------------------
    # 7. ACCEPTANCE
    # ---------------------------------------------------------------
    print()
    print("ACCEPTANCE CHECKS")
    print("-" * 72)

    for name, passed in acceptance[
        "checks"
    ].items():

        print(
            f"{name:<42}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # ---------------------------------------------------------------
    # 8. SAVE VALIDATION DATA
    # ---------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    intelligence.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Validation output saved : "
        f"{OUTPUT_PATH}"
    )

    print()

    if acceptance["passed"]:

        print(
            "PART 16E VALIDATION     : PASSED"
        )

    else:

        print(
            "PART 16E VALIDATION     : FAILED"
        )

        raise RuntimeError(
            "Part 16E acceptance checks failed."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()