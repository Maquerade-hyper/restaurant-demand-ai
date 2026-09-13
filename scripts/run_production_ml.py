"""
PART 29 - LOW-COST PRODUCTION ML ACCEPTANCE RUNNER

29A Production inference architecture
29B Model packaging + loading
29C CPU optimization + caching
29D Batch/API-facing inference
29E Performance validation

This runner:
    - uses the existing Part 28 champion if available
    - otherwise creates a temporary local demonstration model
    - measures actual local performance
    - does not invent latency/cost numbers
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor

from app.production import (
    BatchInferenceEngine,
    PerformanceMonitor,
    ProductionInferenceEngine,
    ProductionMLService,
    ProductionModelLoader,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)

CHAMPION_ROOT = (
    ROOT
    / "models"
    / "champion"
)

DEMO_ROOT = (
    ROOT
    / "models"
    / "production_demo"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "production_ml_result.json"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):

    print(
        f"\nERROR: {message}"
    )

    raise SystemExit(1)


def load_data():

    if not DATA_PATH.exists():
        fail(
            f"Dataset not found: "
            f"{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "date",
            "outlet_id",
            "product_id",
            "deconstrained_demand",
        ],
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    return df


def select_series(
    df,
    max_series=4,
):

    counts = (
        df.groupby(
            [
                "outlet_id",
                "product_id",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    selected = counts.head(
        max_series
    ).index

    index = df.set_index(
        [
            "outlet_id",
            "product_id",
        ]
    ).index

    mask = index.isin(
        selected
    )

    return df.loc[
        mask
    ].copy()


def train_demo_champion(
    history,
):

    print(
        "\nNo permanent champion found."
    )

    print(
        "Creating a temporary local "
        "production demonstration model."
    )

    if DEMO_ROOT.exists():
        shutil.rmtree(
            DEMO_ROOT
        )

    DEMO_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = ProductionInferenceEngine(
        ProductionModelLoader(
            model_root=str(
                DEMO_ROOT
            )
        )
    )

    features = engine.build_features(
        history
    )

    clean = features.dropna(
        subset=engine.FEATURE_COLUMNS
    )

    X = clean[
        engine.FEATURE_COLUMNS
    ].astype(float)

    y = clean[
        "deconstrained_demand"
    ].astype(float)

    model = XGBRegressor(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )

    start = time.perf_counter()

    model.fit(
        X,
        y,
    )

    train_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    joblib.dump(
        model,
        DEMO_ROOT / "model.joblib",
    )

    (
        DEMO_ROOT
        / "manifest.json"
    ).write_text(
        json.dumps(
            {
                "model_name":
                    "production_demo_champion",
                "run_id":
                    "part29-local-demo",
                "model_type":
                    "XGBRegressor",
                "training_rows":
                    len(clean),
                "feature_columns":
                    engine.FEATURE_COLUMNS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Temporary model training: "
        f"{train_ms:.2f} ms"
    )

    return DEMO_ROOT


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PART 29 - LOW-COST PRODUCTION ML"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # DATA
    # ========================================================

    df = load_data()

    print(
        f"Source rows: {len(df):,}"
    )

    df = select_series(
        df,
        max_series=4,
    )

    print(
        f"Selected rows: {len(df):,}"
    )

    print(
        "Selected series:",
        df[
            [
                "outlet_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    # ========================================================
    # CHAMPION DISCOVERY
    # ========================================================

    if (
        CHAMPION_ROOT.exists()
        and (
            CHAMPION_ROOT
            / "model.joblib"
        ).exists()
    ):

        model_root = (
            CHAMPION_ROOT
        )

        print(
            "\nUsing permanent champion:"
        )

        print(
            model_root
        )

    else:

        model_root = (
            train_demo_champion(
                df
            )
        )

        print(
            "Using temporary production "
            "demonstration model."
        )

    # ========================================================
    # 29A / 29B
    # ========================================================

    print(
        "\n29A - PRODUCTION INFERENCE"
    )

    loader = ProductionModelLoader(
        model_root=str(
            model_root
        )
    )

    engine = ProductionInferenceEngine(
        loader
    )

    model = loader.load()

    if model is None:
        fail(
            "Production model did not load."
        )

    print(
        f"Model name: "
        f"{loader.model_name()}"
    )

    print(
        f"Model version: "
        f"{loader.model_version()}"
    )

    print(
        f"Model load: "
        f"{loader.load_time_ms():.3f} ms"
    )

    print(
        "MODEL LOADING: PASS"
    )

    # ========================================================
    # FEATURE FIREWALL
    # ========================================================

    forbidden = {
        "future_demand",
        "true_demand",
        "lost_demand",
        "prediction",
        "actual",
        "target",
    }

    leakage = (
        forbidden
        & set(
            engine.FEATURE_COLUMNS
        )
    )

    print(
        "\nForecast-time feature firewall:"
    )

    print(
        f"Features: "
        f"{len(engine.FEATURE_COLUMNS)}"
    )

    print(
        f"Forbidden leakage: "
        f"{sorted(leakage)}"
    )

    if leakage:
        fail(
            "Production feature firewall failed."
        )

    print(
        "FEATURE FIREWALL: PASS"
    )

    # ========================================================
    # SINGLE INFERENCE
    # ========================================================

    print(
        "\n29C - SINGLE CPU INFERENCE"
    )

    first_series = (
        df[
            [
                "outlet_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .iloc[0]
    )

    outlet_id = (
        first_series["outlet_id"]
    )

    product_id = (
        first_series["product_id"]
    )

    history = df[
        (
            df["outlet_id"]
            == outlet_id
        )
        & (
            df["product_id"]
            == product_id
        )
    ].copy()

    prediction, inference_ms = (
        engine.forecast_latest(
            history
        )
    )

    print(
        f"Outlet: {outlet_id}"
    )

    print(
        f"Product: {product_id}"
    )

    print(
        f"Prediction: "
        f"{prediction:.6f}"
    )

    print(
        f"Inference: "
        f"{inference_ms:.3f} ms"
    )

    if (
        not np.isfinite(
            prediction
        )
        or prediction < 0
    ):
        fail(
            "Invalid production prediction."
        )

    print(
        "SINGLE INFERENCE: PASS"
    )

    # ========================================================
    # BATCH
    # ========================================================

    print(
        "\n29D - BATCH INFERENCE"
    )

    batch = BatchInferenceEngine(
        engine
    )

    batch_start = (
        time.perf_counter()
    )

    batch_result = (
        batch.predict(df)
    )

    batch_ms = (
        time.perf_counter()
        - batch_start
    ) * 1000.0

    print(
        f"Batch series: "
        f"{len(batch_result)}"
    )

    print(
        f"Batch time: "
        f"{batch_ms:.3f} ms"
    )

    print(
        f"Mean inference: "
        f"{batch_result['inference_ms'].mean():.3f} ms"
    )

    if len(batch_result) != 4:
        fail(
            "Unexpected batch series count."
        )

    if not np.isfinite(
        batch_result[
            "prediction"
        ]
    ).all():
        fail(
            "Batch produced non-finite predictions."
        )

    print(
        "BATCH INFERENCE: PASS"
    )

    # ========================================================
    # MODEL CACHE
    # ========================================================

    print(
        "\n29C - MODEL CACHE"
    )

    model_a = loader.load()
    model_b = loader.load()

    if model_a is not model_b:
        fail(
            "Model cache is not reusing "
            "the loaded model object."
        )

    print(
        "Repeated load reused in-memory model."
    )

    print(
        "MODEL CACHE: PASS"
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    print(
        "\n29E - PERFORMANCE VALIDATION"
    )

    service = ProductionMLService(
        model_root=str(
            model_root
        )
    )

    report = service.benchmark(
        df,
        iterations=20,
    )

    print(
        f"Requests: "
        f"{report.requests}"
    )

    print(
        f"Successful: "
        f"{report.successful_requests}"
    )

    print(
        f"Failed: "
        f"{report.failed_requests}"
    )

    print(
        f"Mean: "
        f"{report.mean_ms:.3f} ms"
    )

    print(
        f"P50: "
        f"{report.p50_ms:.3f} ms"
    )

    print(
        f"P95: "
        f"{report.p95_ms:.3f} ms"
    )

    print(
        f"P99: "
        f"{report.p99_ms:.3f} ms"
    )

    print(
        f"Requests/sec: "
        f"{report.requests_per_second:.3f}"
    )

    print(
        f"Rows/sec: "
        f"{report.rows_per_second:.3f}"
    )

    print(
        f"Model load: "
        f"{report.model_load_ms:.3f} ms"
    )

    print(
        f"CPU-only path: "
        f"{report.cpu_only}"
    )

    if report.failed_requests != 0:
        fail(
            "Production benchmark contained failures."
        )

    if not report.cpu_only:
        fail(
            "Production path is not marked CPU-only."
        )

    if not np.isfinite(
        report.mean_ms
    ):
        fail(
            "Non-finite performance metric."
        )

    print(
        "PERFORMANCE VALIDATION: PASS"
    )

    # ========================================================
    # SERVICE
    # ========================================================

    print(
        "\n29E - SERVICE VALIDATION"
    )

    validation = service.validate()

    print(
        f"Validation: "
        f"{validation['passed']}"
    )

    if validation["errors"]:
        print(
            "Errors:",
            validation["errors"],
        )

    if not validation["passed"]:
        fail(
            "Production service validation failed."
        )

    print(
        "SERVICE VALIDATION: PASS"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    result = {
        "part": 29,
        "status": "PASS",
        "model_name":
            loader.model_name(),
        "model_version":
            loader.model_version(),
        "model_load_ms":
            loader.load_time_ms(),
        "single_prediction":
            prediction,
        "single_inference_ms":
            inference_ms,
        "batch_series":
            len(batch_result),
        "batch_ms":
            batch_ms,
        "performance":
            report.to_dict(),
        "cpu_only":
            True,
        "feature_count":
            len(
                engine.FEATURE_COLUMNS
            ),
        "feature_firewall":
            "PASS",
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PART 29 ACCEPTANCE: PASS"
    )

    print(
        "=" * 72
    )

    print(
        f"Result: {OUTPUT_PATH}"
    )

    print(
        f"Model root: {model_root}"
    )


if __name__ == "__main__":
    main()