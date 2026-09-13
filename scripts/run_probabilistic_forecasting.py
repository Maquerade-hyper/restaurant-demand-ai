from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from app.forecasting.probabilistic.service import (
    ProbabilisticForecastingService,
)


INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_spike_intelligence.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "probabilistic_forecasting.csv"
)


def main() -> None:

    print("=" * 70)
    print("PART 24 - PROBABILISTIC FORECASTING")
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

    required = {
        "outlet_id",
        "product_id",
        "date",
        "deconstrained_demand",
        "spike_baseline",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing columns: "
            +
            ", ".join(
                sorted(missing)
            )
        )

    # ============================================================
    # SORT
    # ============================================================

    df = (
        df.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    # ============================================================
    # REMOVE UNREADY BASELINE ROWS
    # ============================================================

    df = df[
        df["spike_baseline"].notna()
    ].copy()

    df["actual"] = pd.to_numeric(
        df["deconstrained_demand"],
        errors="coerce",
    )

    df["point_prediction"] = pd.to_numeric(
        df["spike_baseline"],
        errors="coerce",
    )

    df = df[
        np.isfinite(
            df["actual"]
        )
        &
        np.isfinite(
            df["point_prediction"]
        )
    ].copy()

    # ============================================================
    # CHRONOLOGICAL SPLIT
    # ============================================================

    unique_dates = np.sort(
        df["date"]
        .drop_duplicates()
        .to_numpy()
    )

    n_dates = len(
        unique_dates
    )

    train_end = int(
        n_dates * 0.70
    )

    calibration_end = int(
        n_dates * 0.85
    )

    train_dates = unique_dates[
        :train_end
    ]

    calibration_dates = unique_dates[
        train_end:calibration_end
    ]

    evaluation_dates = unique_dates[
        calibration_end:
    ]

    calibration = df[
        df["date"].isin(
            calibration_dates
        )
    ].copy()

    evaluation = df[
        df["date"].isin(
            evaluation_dates
        )
    ].copy()

    if calibration.empty:
        raise RuntimeError(
            "Calibration set is empty"
        )

    if evaluation.empty:
        raise RuntimeError(
            "Evaluation set is empty"
        )

    print()
    print("CHRONOLOGICAL SPLIT")
    print("-" * 70)

    print(
        f"Calibration rows: "
        f"{len(calibration):,}"
    )

    print(
        f"Evaluation rows: "
        f"{len(evaluation):,}"
    )

    print(
        f"Calibration dates: "
        f"{calibration['date'].min().date()} "
        f"-> "
        f"{calibration['date'].max().date()}"
    )

    print(
        f"Evaluation dates: "
        f"{evaluation['date'].min().date()} "
        f"-> "
        f"{evaluation['date'].max().date()}"
    )

    # ============================================================
    # CALIBRATION RESIDUALS
    # ============================================================

    calibration_actual = (
        calibration["actual"]
        .to_numpy(
            dtype=float
        )
    )

    calibration_prediction = (
        calibration["point_prediction"]
        .to_numpy(
            dtype=float
        )
    )

    evaluation_actual = (
        evaluation["actual"]
        .to_numpy(
            dtype=float
        )
    )

    evaluation_prediction = (
        evaluation["point_prediction"]
        .to_numpy(
            dtype=float
        )
    )

    # ============================================================
    # SERVICE
    # ============================================================

    service = (
        ProbabilisticForecastingService(
            coverage=0.90
        )
    )

    # ============================================================
    # EVALUATION FORECAST
    # ============================================================

    forecast = service.forecast(
        evaluation_prediction,
        calibration_actual,
        calibration_prediction,
    )

    validation = (
        service.validate_forecast(
            forecast
        )
    )

    if not validation["passed"]:

        print(
            "VALIDATION ERRORS:"
        )

        for error in validation["errors"]:
            print(
                f" - {error}"
            )

        raise RuntimeError(
            "PROBABILISTIC FORECAST VALIDATION FAILED"
        )

    # ============================================================
    # CALIBRATION
    # ============================================================

    evaluation_report = (
        service.evaluate(
            evaluation_actual,
            evaluation_prediction,
            calibration_actual,
            calibration_prediction,
        )
    )

    # ============================================================
    # OUTPUT DATAFRAME
    # ============================================================

    result = evaluation[
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ].copy()

    result["actual"] = (
        evaluation_actual
    )

    result["point_forecast"] = (
        evaluation_prediction
    )

    quantiles = (
        forecast["quantiles"]
    )

    for name, values in quantiles.items():

        result[
            name
        ] = values

    result["conformal_lower"] = (
        forecast[
            "interval"
        ]["lower"]
    )

    result["conformal_upper"] = (
        forecast[
            "interval"
        ]["upper"]
    )

    result["conformal_radius"] = (
        forecast[
            "interval"
        ]["radius"]
    )

    result["asymmetric_lower"] = (
        forecast[
            "asymmetric_interval"
        ]["lower"]
    )

    result["asymmetric_upper"] = (
        forecast[
            "asymmetric_interval"
        ]["upper"]
    )

    result["covered_90"] = (
        (
            result["actual"]
            >=
            result["conformal_lower"]
        )
        &
        (
            result["actual"]
            <=
            result["conformal_upper"]
        )
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    interval_report = (
        evaluation_report[
            "intervals"
        ]
    )

    quantile_report = (
        evaluation_report[
            "quantiles"
        ]
    )

    print()
    print("PROBABILISTIC RESULTS")
    print("-" * 70)

    for name in [
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
    ]:

        report = (
            quantile_report[name]
        )

        print(
            f"{name}: "
            f"pinball="
            f"{report['pinball_loss']:.4f}"
        )

    conformal = (
        interval_report[
            "conformal_90"
        ]
    )

    print(
        f"Conformal 90% coverage: "
        f"{conformal['coverage']:.4f}"
    )

    print(
        f"Conformal mean width: "
        f"{conformal['mean_width']:.4f}"
    )

    asymmetric = (
        interval_report[
            "asymmetric_80"
        ]
    )

    print(
        f"Asymmetric interval coverage: "
        f"{asymmetric['coverage']:.4f}"
    )

    print(
        f"Asymmetric mean width: "
        f"{asymmetric['mean_width']:.4f}"
    )

    # ============================================================
    # SAVE
    # ============================================================

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
    print("VALIDATION")
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
        "PART 24 PASSED"
    )


if __name__ == "__main__":
    main()