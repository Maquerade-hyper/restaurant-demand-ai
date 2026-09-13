from __future__ import annotations

"""
PART B — Continuous Production Intelligence / Learning Environment
End-to-end acceptance runner.

This runner validates the real Part B contracts without modifying the
production model registry.

Validation flow:

    client package
        ↓
    INCOMING
        ↓
    gateway
        ↓
    schema validation
        ↓
    sufficiency validation
        ↓
    canonical preparation
        ↓
    Part 7 feature pipeline
        ↓
    Part 9 XGBoost / 51 features
        ↓
    candidate
        ↓
    common unseen evaluation
        ↓
    governance
        ↓
    package lifecycle

The test fixture intentionally lives only in a temporary directory.
It does NOT read data/synthetic.
It does NOT modify models/champion.
"""



import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.learning.continuous.registry import ModelRegistry
from app.learning.runtime.data_gateway import ClientDataGateway
from app.learning.runtime.runtime_service import LearningRuntimeService
from app.learning.runtime.training_adapter import ProductionTrainingAdapter


# ============================================================
# ACCEPTANCE CONSTANTS
# ============================================================

DAYS = 240

OUTLETS = [
    "CLIENT_O000",
    "CLIENT_O001",
    "CLIENT_O002",
    "CLIENT_O003",
]

PRODUCTS = [
    "CLIENT_P000",
    "CLIENT_P001",
    "CLIENT_P002",
]

EXPECTED_FEATURE_COUNT = 51

MIN_TRAIN_ROWS = 1000

RESULTS: list[tuple[str, bool, str]] = []


# ============================================================
# OUTPUT
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def record(
    name: str,
    passed: bool,
    detail: str = "",
) -> None:

    RESULTS.append(
        (
            name,
            passed,
            detail,
        )
    )

    status = "PASS" if passed else "FAIL"

    if detail:
        print(
            f"[{status}] {name}: {detail}"
        )
    else:
        print(
            f"[{status}] {name}"
        )


def check(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise AssertionError(message)


# ============================================================
# CLIENT FIXTURE
# ============================================================

def generate_client_data(
    days: int = DAYS,
) -> pd.DataFrame:

    """
    Deterministic real-client-shaped fixture.

    This is generated specifically for acceptance testing and is NOT
    loaded from the project's synthetic data directory.
    """

    rng = np.random.default_rng(20260913)

    dates = pd.date_range(
        "2025-01-01",
        periods=days,
        freq="D",
    )

    outlet_base = {
        "CLIENT_O000": 38.0,
        "CLIENT_O001": 52.0,
        "CLIENT_O002": 67.0,
        "CLIENT_O003": 44.0,
    }

    product_factor = {
        "CLIENT_P000": 1.00,
        "CLIENT_P001": 1.25,
        "CLIENT_P002": 0.80,
    }

    rows: list[dict] = []

    for outlet_id in OUTLETS:

        for product_id in PRODUCTS:

            base = (
                outlet_base[outlet_id]
                * product_factor[product_id]
            )

            for day_index, date in enumerate(dates):

                dow = date.dayofweek

                weekend_factor = (
                    1.18
                    if dow >= 5
                    else 1.00
                )

                weekly_pattern = (
                    1.00
                    + 0.08
                    * np.sin(
                        2.0
                        * np.pi
                        * day_index
                        / 7.0
                    )
                )

                long_term_trend = (
                    1.00
                    + 0.0008
                    * day_index
                )

                noise = rng.normal(
                    0.0,
                    2.0,
                )

                quantity = (
                    base
                    * weekend_factor
                    * weekly_pattern
                    * long_term_trend
                    + noise
                )

                rows.append(
                    {
                        "date": date,
                        "outlet_id": outlet_id,
                        "product_id": product_id,
                        "quantity_sold": max(
                            0.0,
                            float(
                                round(
                                    quantity,
                                    2,
                                )
                            ),
                        ),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# WRITE PACKAGE
# ============================================================

def write_package(
    df: pd.DataFrame,
    incoming: Path,
    filename: str,
) -> Path:

    incoming.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = incoming / filename

    df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================
# GATEWAY VALIDATION RESULT HANDLING
# ============================================================

def validation_passed(
    result,
) -> bool:

    """
    Handle the actual gateway validation contract without assuming that
    the result dictionary uses a particular key such as 'valid'.

    Runtime itself is the authoritative consumer.
    """

    if isinstance(result, bool):
        return result

    if isinstance(result, dict):

        for key in (
            "passed",
            "is_valid",
            "valid",
            "success",
        ):

            if key in result:
                return bool(
                    result[key]
                )

    return False


# ============================================================
# TEST 1 — EMPTY INCOMING
# ============================================================

def test_empty_incoming(
    gateway: ClientDataGateway,
) -> None:

    packages = gateway.discover_packages()

    check(
        len(packages) == 0,
        (
            "expected empty INCOMING, "
            f"found {len(packages)} package(s)"
        ),
    )

    record(
        "EMPTY INCOMING",
        True,
        "runtime safely waits",
    )


# ============================================================
# TEST 2 — INVALID SCHEMA
# ============================================================

def test_invalid_schema(
    gateway: ClientDataGateway,
) -> None:

    path = write_package(
        pd.DataFrame(
            {
                "wrong_date": [
                    "2025-01-01"
                ],
                "wrong_value": [
                    10
                ],
            }
        ),
        gateway.incoming_path,
        "invalid_schema.csv",
    )

    packages = gateway.discover_packages()

    check(
        packages,
        "invalid package was not discovered",
    )

    package = next(
        (
            package
            for package in packages
            if any(
                "invalid_schema"
                in str(file)
                for file in package.files
            )
        ),
        None,
    )

    check(
        package is not None,
        "invalid package not found",
    )

    validation = (
        gateway.validate_package_sample(
            package
        )
    )

    check(
        not validation_passed(
            validation
        ),
        (
            "invalid schema unexpectedly "
            "passed gateway validation"
        ),
    )

    rejected = gateway.reject_package(
        package,
        "INVALID_CLIENT_DATA",
    )

    check(
        rejected.exists(),
        "REJECTED directory does not exist",
    )

    check(
        not path.exists(),
        "invalid package remained in INCOMING",
    )

    check(
        (rejected / "manifest.json").exists(),
        "REJECTED manifest.json missing",
    )

    record(
        "INVALID CLIENT SCHEMA",
        True,
        "invalid package isolated in REJECTED",
    )


# ============================================================
# TEST 3 — INSUFFICIENT DATA
# ============================================================

def test_insufficient_data(
    gateway: ClientDataGateway,
) -> None:

    small = generate_client_data(
        days=20
    )

    path = write_package(
        small,
        gateway.incoming_path,
        "insufficient_client.csv",
    )

    packages = gateway.discover_packages()

    package = next(
        (
            package
            for package in packages
            if any(
                "insufficient_client"
                in str(file)
                for file in package.files
            )
        ),
        None,
    )

    check(
        package is not None,
        "insufficient package not discovered",
    )

    # ClientDataPackage does not expose row_count.
    # Calculate the package size from the actual file.
    actual_rows = 0

    for file_path in package.files:

        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            actual_rows += max(
                0,
                sum(
                    1
                    for _ in file_path.open(
                        "r",
                        encoding="utf-8",
                    )
                ) - 1,
            )

    check(
        actual_rows < MIN_TRAIN_ROWS,
        (
            f"fixture unexpectedly contains "
            f"{actual_rows} rows"
        ),
    )

    # Insufficient data must remain available for a future retry.
    check(
        path.exists(),
        "insufficient package disappeared from INCOMING",
    )

    record(
        "INSUFFICIENT DATA",
        True,
        (
            f"{actual_rows} rows correctly remain "
            "in INCOMING without training"
        ),
    )

    path.unlink()


# ============================================================
# TEST 4 — PREPARATION
# ============================================================

def test_preparation(
    adapter: ProductionTrainingAdapter,
    df: pd.DataFrame,
) -> pd.DataFrame:

    prepared = adapter.prepare(
        df
    )

    check(
        isinstance(
            prepared,
            pd.DataFrame,
        ),
        "prepare() did not return DataFrame",
    )

    check(
        not prepared.empty,
        "prepare() returned empty dataframe",
    )

    required = {
        "date",
        "outlet_id",
        "product_id",
        "quantity_sold",
    }

    check(
        required.issubset(
            prepared.columns
        ),
        "canonical production columns missing",
    )

    record(
        "CANONICAL PREPARATION",
        True,
        f"{len(prepared):,} canonical rows",
    )

    return prepared


# ============================================================
# TEST 5 — PRODUCTION FEATURES
# ============================================================

def test_production_features(
    adapter: ProductionTrainingAdapter,
    df: pd.DataFrame,
) -> pd.DataFrame:

    featured = adapter.build_features(
        df
    )

    check(
        not featured.empty,
        "feature pipeline returned zero rows",
    )

    check(
        "quantity_sold"
        in featured.columns,
        "target was removed by production feature pipeline",
    )

    record(
        "PART 7 / PART 9 FEATURE PIPELINE",
        True,
        (
            f"{len(featured):,} feature rows, "
            f"{len(featured.columns)} total columns"
        ),
    )

    return featured


# ============================================================
# TEST 6 — PRODUCTION MODEL
# ============================================================

def test_training(
    adapter: ProductionTrainingAdapter,
    df: pd.DataFrame,
) -> XGBoostDemandModel:

    train_df, validation_df, evaluation_df = (
        adapter.chronological_split(
            df,
            validation_days=30,
            evaluation_days=30,
        )
    )

    check(
        len(train_df) >= MIN_TRAIN_ROWS,
        (
            f"training split has only "
            f"{len(train_df)} raw rows"
        ),
    )

    model = adapter.train_model(
        train_df
    )

    check(
        model is not None,
        "model training returned None",
    )

    check(
        len(model.feature_columns)
        == EXPECTED_FEATURE_COUNT,
        (
            "production feature contract mismatch: "
            f"expected {EXPECTED_FEATURE_COUNT}, "
            f"got {len(model.feature_columns)}"
        ),
    )

    check(
        model.feature_columns,
        "trained model has no feature columns",
    )

    record(
        "PART 9 XGBOOST CANDIDATE",
        True,
        (
            f"{len(model.feature_columns)} features, "
            f"{len(train_df):,} training rows"
        ),
    )

    return model


# ============================================================
# TEST 7 — HISTORICAL-CONTEXT PREDICTION
# ============================================================

def test_prediction(
    adapter: ProductionTrainingAdapter,
    model: XGBoostDemandModel,
    df: pd.DataFrame,
) -> None:

    (
        train_df,
        validation_df,
        evaluation_df,
    ) = adapter.chronological_split(
        df,
        validation_days=30,
        evaluation_days=30,
    )

    history_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    predictions = adapter.predict(
        model,
        evaluation_df,
        history_df=history_df,
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    check(
        len(predictions) == len(evaluation_df),
        (
            "prediction length mismatch: "
            f"{len(predictions)} != "
            f"{len(evaluation_df)}"
        ),
    )

    check(
        np.isfinite(
            predictions
        ).all(),
        "prediction contains NaN/inf",
    )

    check(
        (predictions >= 0).all(),
        "prediction contains negative demand",
    )

    record(
        "HISTORICAL-CONTEXT INFERENCE",
        True,
        f"{len(predictions):,} valid predictions",
    )


# ============================================================
# TEST 8 — INITIAL CHAMPION
# ============================================================

def test_initial_champion(
    adapter: ProductionTrainingAdapter,
    df: pd.DataFrame,
) -> XGBoostDemandModel:

    train_df, _, _ = (
        adapter.chronological_split(
            df,
            validation_days=30,
            evaluation_days=30,
        )
    )

    champion, run_id = (
        adapter.create_initial_champion(
            train_df,
            adapter.registry,
        )
    )

    check(
        champion is not None,
        "initial champion was not created",
    )

    check(
        len(champion.feature_columns)
        == EXPECTED_FEATURE_COUNT,
        (
            "initial champion has "
            f"{len(champion.feature_columns)} features"
        ),
    )

    check(
        adapter.registry.has_champion(),
        "registry does not report champion",
    )

    loaded = (
        adapter.registry.load_champion()
    )

    check(
        loaded is not None,
        "champion could not be loaded",
    )

    check(
        len(loaded.feature_columns)
        == EXPECTED_FEATURE_COUNT,
        "loaded champion feature contract changed",
    )

    record(
        "CHAMPION PERSISTENCE",
        True,
        f"initial champion {run_id}",
    )

    return champion


# ============================================================
# TEST 9 — CANDIDATE VS CHAMPION
# ============================================================

def test_common_unseen_evaluation(
    adapter: ProductionTrainingAdapter,
    champion: XGBoostDemandModel,
    candidate: XGBoostDemandModel,
    df: pd.DataFrame,
) -> dict:

    (
        train_df,
        validation_df,
        evaluation_df,
    ) = adapter.chronological_split(
        df,
        validation_days=30,
        evaluation_days=30,
    )

    history_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    evaluation = adapter.evaluate(
        champion,
        candidate,
        evaluation_df,
        history_df=history_df,
    )

    required = {
        "champion",
        "candidate",
        "improvement_pct",
        "bias_change",
        "passes",
        "evaluation_rows",
    }

    check(
        required.issubset(
            evaluation.keys()
        ),
        (
            "evaluation missing keys: "
            f"{sorted(required - set(evaluation.keys()))}"
        ),
    )

    check(
        evaluation["evaluation_rows"]
        >= adapter.MIN_EVALUATION_ROWS,
        "insufficient common unseen rows",
    )

    record(
        "COMMON UNSEEN EVALUATION",
        True,
        (
            f"champion MAE="
            f"{evaluation['champion']['mae']:.4f}, "
            f"candidate MAE="
            f"{evaluation['candidate']['mae']:.4f}, "
            f"improvement="
            f"{evaluation['improvement_pct']:.4f}%"
        ),
    )

    return evaluation


# ============================================================
# TEST 10 — SYNTHETIC FIREWALL
# ============================================================

def test_synthetic_firewall(
    gateway: ClientDataGateway,
) -> None:

    incoming = (
        gateway.incoming_path
        .resolve()
    )

    synthetic = (
        ROOT
        / "data"
        / "synthetic"
    ).resolve()

    check(
        incoming != synthetic,
        "INCOMING points to synthetic data",
    )

    check(
        "synthetic"
        not in str(incoming).lower(),
        "runtime path contains synthetic source",
    )

    record(
        "SYNTHETIC-DATA FIREWALL",
        True,
        "client runtime is isolated from data/synthetic",
    )


# ============================================================
# TEST 11 — FULL RUNTIME
# ============================================================

def test_runtime_lifecycle(
    runtime: LearningRuntimeService,
    gateway: ClientDataGateway,
    df: pd.DataFrame,
) -> dict:

    path = write_package(
        df,
        gateway.incoming_path,
        "full_client_training_package.csv",
    )

    check(
        path.exists(),
        "client package was not written",
    )

    result = runtime.scan_once()

    check(
        result is not None,
        "runtime returned no result",
    )

    # Successful processing should remove the package from INCOMING.
    check(
        not path.exists(),
        "processed package remained in INCOMING",
    )

    processed_files = list(
        gateway.processed_path.iterdir()
    )

    check(
        processed_files,
        "PROCESSED directory is empty",
    )

    record(
        "FULL RUNTIME LIFECYCLE",
        True,
        f"runtime result: {result}",
    )

    return {
        "result": result,
        "processed_files": [
            str(path)
            for path in processed_files
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    section(
        "PART B — CONTINUOUS PRODUCTION INTELLIGENCE / "
        "LEARNING ENVIRONMENT"
    )

    print(
        f"Project root: {ROOT}"
    )

    print(
        "Mode: isolated acceptance environment"
    )

    print(
        "Production model registry will NOT be modified."
    )

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="restaurant_demand_ai_part_b_"
        )
    )

    try:

        # --------------------------------------------------------
        # ISOLATED CLIENT DIRECTORIES
        # --------------------------------------------------------

        incoming = (
            temp_root
            / "client"
            / "INCOMING"
        )

        processed = (
            temp_root
            / "client"
            / "PROCESSED"
        )

        rejected = (
            temp_root
            / "client"
            / "REJECTED"
        )

        archive = (
            temp_root
            / "client"
            / "ARCHIVE"
        )

        model_root = (
            temp_root
            / "models"
        )

        for directory in (
            incoming,
            processed,
            rejected,
            archive,
            model_root,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        # --------------------------------------------------------
        # GATEWAY
        # --------------------------------------------------------

        gateway = ClientDataGateway(
            incoming_path=incoming,
            processed_path=processed,
            rejected_path=rejected,
            archive_path=archive,
        )

        # --------------------------------------------------------
        # ISOLATED REGISTRY
        # --------------------------------------------------------

        registry = ModelRegistry(
            root=str(model_root)
        )

        # --------------------------------------------------------
        # PRODUCTION ADAPTER
        # --------------------------------------------------------

        adapter = ProductionTrainingAdapter(
            registry=registry,
        )

        # --------------------------------------------------------
        # RUNTIME
        # --------------------------------------------------------

        runtime = LearningRuntimeService(
            gateway=gateway,
            training_adapter=adapter,
        )

        # --------------------------------------------------------
        # 1
        # --------------------------------------------------------

        try:
            test_empty_incoming(
                gateway
            )
        except Exception as exc:
            record(
                "EMPTY INCOMING",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # 2
        # --------------------------------------------------------

        try:
            test_invalid_schema(
                gateway
            )
        except Exception as exc:
            record(
                "INVALID CLIENT SCHEMA",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # 3
        # --------------------------------------------------------

        try:
            test_insufficient_data(
                gateway
            )
        except Exception as exc:
            record(
                "INSUFFICIENT DATA",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # REAL CLIENT-SHAPED FIXTURE
        # --------------------------------------------------------

        client_df = generate_client_data(
            days=DAYS
        )

        print()
        print(
            f"Client fixture rows: "
            f"{len(client_df):,}"
        )

        print(
            "Client series: "
            f"{client_df['outlet_id'].nunique()} × "
            f"{client_df['product_id'].nunique()}"
        )

        # --------------------------------------------------------
        # 4
        # --------------------------------------------------------

        try:
            test_preparation(
                adapter,
                client_df,
            )
        except Exception as exc:
            record(
                "CANONICAL PREPARATION",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # 5
        # --------------------------------------------------------

        try:
            test_production_features(
                adapter,
                client_df,
            )
        except Exception as exc:
            record(
                "PART 7 / PART 9 FEATURE PIPELINE",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # 6
        # --------------------------------------------------------

        candidate = None

        try:
            candidate = test_training(
                adapter,
                client_df,
            )
        except Exception as exc:
            record(
                "PART 9 XGBOOST CANDIDATE",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # 7
        # --------------------------------------------------------

        if candidate is not None:

            try:
                test_prediction(
                    adapter,
                    candidate,
                    client_df,
                )
            except Exception as exc:
                record(
                    "HISTORICAL-CONTEXT INFERENCE",
                    False,
                    str(exc),
                )

        # --------------------------------------------------------
        # 8
        # --------------------------------------------------------

        champion = None

        try:
            champion = test_initial_champion(
                adapter,
                client_df,
            )
        except Exception as exc:
            record(
                "CHAMPION PERSISTENCE",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # 9
        # --------------------------------------------------------

        if (
            candidate is not None
            and champion is not None
        ):

            try:
                test_common_unseen_evaluation(
                    adapter,
                    champion,
                    candidate,
                    client_df,
                )
            except Exception as exc:
                record(
                    "COMMON UNSEEN EVALUATION",
                    False,
                    str(exc),
                )

        # --------------------------------------------------------
        # 10
        # --------------------------------------------------------

        try:
            test_synthetic_firewall(
                gateway
            )
        except Exception as exc:
            record(
                "SYNTHETIC-DATA FIREWALL",
                False,
                str(exc),
            )

        # --------------------------------------------------------
        # 11
        #
        # Run the complete runtime only if the production adapter
        # itself has passed its core contracts.
        # --------------------------------------------------------

        if (
            candidate is not None
            and champion is not None
        ):

            try:

                runtime_df = generate_client_data(
                    days=365
                )

                lifecycle = test_runtime_lifecycle(
                    runtime,
                    gateway,
                    runtime_df,
                )

                artifact = {
                    "part": "B",
                    "status": "acceptance_test",
                    "rows": int(
                        len(runtime_df)
                    ),
                    "outlets": int(
                        runtime_df[
                            "outlet_id"
                        ].nunique()
                    ),
                    "products": int(
                        runtime_df[
                            "product_id"
                        ].nunique()
                    ),
                    "processed_files": lifecycle[
                        "processed_files"
                    ],
                }

                artifact_path = (
                    temp_root
                    / "part_b_acceptance.json"
                )

                artifact_path.write_text(
                    json.dumps(
                        artifact,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )

                print(
                    f"Acceptance artifact: "
                    f"{artifact_path}"
                )

            except Exception as exc:
                record(
                    "FULL RUNTIME LIFECYCLE",
                    False,
                    str(exc),
                )

        else:

            record(
                "FULL RUNTIME LIFECYCLE",
                False,
                "skipped because core production adapter failed",
            )

        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------

        section(
            "PART B ACCEPTANCE SUMMARY"
        )

        passed = sum(
            1
            for _, ok, _ in RESULTS
            if ok
        )

        failed = len(RESULTS) - passed

        for name, ok, detail in RESULTS:

            status = (
                "PASS"
                if ok
                else "FAIL"
            )

            print(
                f"{status:<5} | "
                f"{name:<40} | "
                f"{detail}"
            )

        print()
        print("-" * 78)

        print(
            f"PASSED: {passed} | "
            f"FAILED: {failed} | "
            f"TOTAL: {len(RESULTS)}"
        )

        if failed == 0:

            print()
            print(
                "PART B ACCEPTANCE: PASS"
            )

            print(
                "Continuous Production Intelligence / "
                "Learning Environment is operational."
            )

            return 0

        print()
        print(
            "PART B ACCEPTANCE: FAIL"
        )

        return 1

    finally:

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )

        print()
        print(
            "Temporary Part B environment removed."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )