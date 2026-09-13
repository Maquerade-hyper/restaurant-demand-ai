from __future__ import annotations

import sys
from pathlib import Path


# ==============================================================
# PROJECT ROOT
# ==============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


import numpy as np
import pandas as pd

from app.forecasting.benchmarking import (
    ModelBenchmarkingService,
)


# ==============================================================
# INPUT
# ==============================================================

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
    / "model_benchmarking.csv"
)


# ==============================================================
# LOAD ONE REPRESENTATIVE SERIES
# ==============================================================

def load_series():

    if not SALES_PATH.exists():

        raise FileNotFoundError(
            f"Sales file not found: "
            f"{SALES_PATH}"
        )

    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"],
    )

    required = {
        "outlet_id",
        "product_id",
        "date",
        "quantity_sold",
    }

    missing = (
        required
        - set(sales.columns)
    )

    if missing:

        raise ValueError(
            f"sales missing: "
            f"{sorted(missing)}"
        )

    # ----------------------------------------------------------
    # Select a deterministic representative series.
    # ----------------------------------------------------------

    first_outlet = (
        sales["outlet_id"]
        .sort_values()
        .iloc[0]
    )

    first_product = (
        sales[
            sales["outlet_id"]
            == first_outlet
        ]["product_id"]
        .sort_values()
        .iloc[0]
    )

    series = sales[
        (
            sales["outlet_id"]
            == first_outlet
        )
        & (
            sales["product_id"]
            == first_product
        )
    ].sort_values(
        "date"
    )

    values = (
        pd.to_numeric(
            series["quantity_sold"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(values) < 60:

        raise ValueError(
            "insufficient series history"
        )

    # ----------------------------------------------------------
    # Fixed temporal holdout.
    #
    # Last 14 days are unseen test data.
    # ----------------------------------------------------------

    history = values[:-14]
    test = values[-14:]

    return (
        pd.Series(history),
        pd.Series(test),
        first_outlet,
        first_product,
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print(
        "PART 21 - MODEL BENCHMARKING"
    )
    print("=" * 70)

    (
        history,
        test,
        outlet_id,
        product_id,
    ) = load_series()

    print(
        f"Outlet          : {outlet_id}"
    )

    print(
        f"Product         : {product_id}"
    )

    print(
        f"History rows    : {len(history):,}"
    )

    print(
        f"Test rows       : {len(test):,}"
    )

    print()

    service = (
        ModelBenchmarkingService()
    )

    result = service.analyze(
        history=history,
        test=test,
    )

    validation = service.validate(
        result
    )

    print(
        "MODEL BENCHMARK RESULTS"
    )

    print("-" * 70)

    display_columns = [
        "rank",
        "model_name",
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "high_demand_bias",
        "spike_recall",
        "stability",
        "prediction_cost",
        "score",
    ]

    print(
        result[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()

    winner = result.iloc[0]

    print(
        "WINNER"
    )

    print("-" * 70)

    print(
        f"Model           : "
        f"{winner['model_name']}"
    )

    print(
        f"MAE             : "
        f"{winner['mae']:.4f}"
    )

    print(
        f"RMSE            : "
        f"{winner['rmse']:.4f}"
    )

    print(
        f"sMAPE           : "
        f"{winner['smape']:.4f}"
    )

    print(
        f"Spike Recall    : "
        f"{winner['spike_recall']:.4f}"
    )

    print(
        f"Score           : "
        f"{winner['score']:.4f}"
    )

    print()

    print(
        "VALIDATION"
    )

    print("-" * 70)

    print(
        f"PASSED          : "
        f"{validation['passed']}"
    )

    print(
        f"ERRORS          : "
        f"{len(validation['errors'])}"
    )

    if validation["errors"]:

        for error in validation["errors"]:
            print(
                f"  ERROR: {error}"
            )

        raise SystemExit(
            "PART 21 VALIDATION FAILED"
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
        f"Output saved    : "
        f"{OUTPUT_PATH}"
    )

    print()

    print("=" * 70)
    print(
        "PART 21 MODEL BENCHMARKING : PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()