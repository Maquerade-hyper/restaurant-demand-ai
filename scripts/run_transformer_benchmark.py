from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT),
    )


from app.forecasting.transformer.service import (
    TransformerBenchmarkService,
)


INPUT_PATH = (
    ROOT
    /
    "data"
    /
    "interim"
    /
    "demand_censoring_intelligence.csv"
)

OUTPUT_PATH = (
    ROOT
    /
    "data"
    /
    "interim"
    /
    "transformer_benchmark.csv"
)

DETAIL_OUTPUT_PATH = (
    ROOT
    /
    "data"
    /
    "interim"
    /
    "transformer_benchmark_series.csv"
)


def main():

    print("=" * 70)
    print(
        "PART 25 - MULTI-SERIES TRANSFORMER BENCHMARK"
    )
    print("=" * 70)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        f"Outlets: "
        f"{df['outlet_id'].nunique():,}"
    )

    print(
        f"Products: "
        f"{df['product_id'].nunique():,}"
    )

    service = TransformerBenchmarkService(
        sequence_length=28,
        test_size=30,
        validation_size=30,
        improvement_threshold=0.02,
        minimum_series_win_rate=0.50,
        epochs=12,
        batch_size=32,
    )

    aggregate, detail = (
        service.benchmark(
            df,
            max_series=12,
        )
    )

    validation = service.validate(
        aggregate,
        detail,
    )

    if not validation["passed"]:

        print()
        print(
            "VALIDATION FAILED"
        )

        for error in validation[
            "errors"
        ]:

            print(
                f" - {error}"
            )

        raise RuntimeError(
            "PART 25 validation failed"
        )

    decision = (
        service.production_decision(
            aggregate,
            detail,
        )
    )

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================

    print()
    print(
        "AGGREGATE BENCHMARK"
    )
    print("-" * 70)

    columns = [
        "rank",
        "model_name",
        "series_count",
        "test_observations",
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "high_demand_bias",
        "spike_recall",
        "prediction_seconds",
        "prediction_cost",
    ]

    print(
        aggregate[
            columns
        ].to_string(
            index=False
        )
    )

    # ============================================================
    # SERIES RESULTS
    # ============================================================

    print()
    print(
        "SERIES-LEVEL RESULTS"
    )
    print("-" * 70)

    print(
        detail[
            [
                "outlet_id",
                "product_id",
                "naive_mae",
                "xgboost_mae",
                "transformer_mae",
                "transformer_vs_xgb_improvement",
                "best_model",
            ]
        ].to_string(
            index=False
        )
    )

    # ============================================================
    # DECISION
    # ============================================================

    print()
    print(
        "PRODUCTION DECISION"
    )
    print("-" * 70)

    print(
        f"Recommendation: "
        f"{decision['recommendation']}"
    )

    print(
        f"XGBoost MAE: "
        f"{decision['xgboost_mae']:.6f}"
    )

    print(
        f"Transformer MAE: "
        f"{decision['transformer_mae']:.6f}"
    )

    print(
        f"Transformer MAE improvement: "
        f"{decision['transformer_mae_improvement'] * 100:.2f}%"
    )

    print(
        f"Transformer series win rate: "
        f"{decision['transformer_series_win_rate'] * 100:.2f}%"
    )

    print(
        f"Required series win rate: "
        f"{decision['minimum_series_win_rate'] * 100:.2f}%"
    )

    print(
        f"Meaningful accuracy improvement: "
        f"{decision['meaningful_improvement']}"
    )

    print(
        f"Sufficient series wins: "
        f"{decision['sufficient_series_wins']}"
    )

    print(
        f"XGBoost prediction cost: "
        f"{decision['xgboost_prediction_cost']:.8f}"
    )

    print(
        f"Transformer prediction cost: "
        f"{decision['transformer_prediction_cost']:.8f}"
    )

    # ============================================================
    # SAVE
    # ============================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    detail.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Aggregate output: {OUTPUT_PATH}"
    )

    print(
        f"Series output: {DETAIL_OUTPUT_PATH}"
    )

    print()
    print(
        "VALIDATION"
    )
    print("-" * 70)

    print(
        f"PASSED: "
        f"{validation['passed']}"
    )

    print(
        f"ERRORS: "
        f"{len(validation['errors'])}"
    )

    print()
    print(
        "PART 25 PASSED"
    )


if __name__ == "__main__":
    main()