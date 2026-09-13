from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.learning.runtime.data_gateway import (
    ClientDataGateway,
)
from app.learning.runtime.training_adapter import (
    ProductionTrainingAdapter,
)


logger = logging.getLogger(__name__)


class LearningRuntimeService:
    """
    Part B continuous production-learning runtime.

    Flow:

        INCOMING
            ↓
        discover real client package
            ↓
        sample validation
            ↓
        canonicalization
            ↓
        data sufficiency gate
            ↓
        chronological split
            ↓
        production feature pipeline
            ↓
        Part 9 XGBoost candidate
            ↓
        validation
            ↓
        unseen evaluation
            ↓
        promotion gate
            ↓
        champion / rejection
            ↓
        processed package

    Synthetic demonstration data is never discovered or trained on.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".parquet",
        ".json",
    }

    def __init__(
        self,
        gateway: ClientDataGateway,
        training_adapter: Optional[
            ProductionTrainingAdapter
        ] = None,
        minimum_rows: int = 1000,
        validation_days: int = 30,
        evaluation_days: int = 30,
    ):
        self.gateway = gateway

        self.training_adapter = (
            training_adapter
            or ProductionTrainingAdapter()
        )

        self.minimum_rows = int(
            minimum_rows
        )

        self.validation_days = int(
            validation_days
        )

        self.evaluation_days = int(
            evaluation_days
        )

        self.last_package_id: Optional[
            str
        ] = None

        self.last_status: str = "IDLE"

        self.last_message: str = (
            "Runtime initialized."
        )

        self.last_run_at: Optional[
            str
        ] = None

        self.last_training_result: Optional[
            dict
        ] = None

    # ============================================================
    # STATUS
    # ============================================================

    def status(self) -> dict:
        """
        Return the current runtime state.
        """

        return {
            "status": self.last_status,
            "message": self.last_message,
            "last_package_id": (
                self.last_package_id
            ),
            "last_run_at": (
                self.last_run_at
            ),
            "last_training_result": (
                self.last_training_result
            ),
        }

    def _set_status(
        self,
        status: str,
        message: str,
    ) -> None:
        """
        Update runtime state and emit a log entry.
        """

        self.last_status = status

        self.last_message = message

        self.last_run_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        logger.info(
            "[LEARNING] %s: %s",
            status,
            message,
        )

    # ============================================================
    # FILE LOADING
    # ============================================================

    @staticmethod
    def _read_file(
        path: Path,
    ) -> pd.DataFrame:
        """
        Read one supported client data file.
        """

        suffix = path.suffix.lower()

        if suffix == ".csv":

            return pd.read_csv(
                path
            )

        if suffix == ".parquet":

            return pd.read_parquet(
                path
            )

        if suffix == ".json":

            return pd.read_json(
                path
            )

        raise ValueError(
            "Unsupported client data format: "
            f"{path.suffix}"
        )

    def _load_package(
        self,
        package,
    ) -> pd.DataFrame:
        """
        Load all supported files belonging to a
        discovered client package.
        """

        frames = []

        for file_path in package.files:

            path = Path(
                file_path
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            if (
                path.suffix.lower()
                not in self.SUPPORTED_EXTENSIONS
            ):
                continue

            logger.info(
                "[LEARNING] Reading client file: %s",
                path.name,
            )

            frame = self._read_file(
                path
            )

            if not frame.empty:
                frames.append(
                    frame
                )

        if not frames:
            raise ValueError(
                "Client package contains no readable data."
            )

        return pd.concat(
            frames,
            ignore_index=True,
        )

    # ============================================================
    # COLUMN NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_columns(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize common client column names into
        the canonical training contract.
        """

        frame = df.copy()

        frame.columns = [
            str(column)
            .strip()
            .lower()
            for column in frame.columns
        ]

        aliases = {
            "timestamp": "date",
            "datetime": "date",
            "sale_date": "date",
            "sales_date": "date",
            "transaction_date": "date",
            "quantity": "quantity_sold",
            "qty": "quantity_sold",
            "units_sold": "quantity_sold",
            "sold_quantity": "quantity_sold",
            "outlet": "outlet_id",
            "store_id": "outlet_id",
            "restaurant_id": "outlet_id",
            "branch_id": "outlet_id",
            "product": "product_id",
            "item_id": "product_id",
            "sku": "product_id",
            "item": "product_id",
        }

        rename = {}

        for source, target in aliases.items():

            if (
                source in frame.columns
                and target not in frame.columns
            ):
                rename[source] = target

        if rename:
            frame = frame.rename(
                columns=rename
            )

        return frame

    # ============================================================
    # CANONICAL CLIENT DATA
    # ============================================================

    @classmethod
    def _prepare_client_data(
        cls,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert incoming client data into the
        canonical sales training contract.
        """

        frame = cls._normalize_columns(
            df
        )

        required = {
            "date",
            "outlet_id",
            "product_id",
            "quantity_sold",
        }

        missing = (
            required
            - set(frame.columns)
        )

        if missing:
            raise ValueError(
                "Client data is missing required "
                "canonical columns: "
                f"{sorted(missing)}"
            )

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        )

        frame["quantity_sold"] = pd.to_numeric(
            frame["quantity_sold"],
            errors="coerce",
        )

        frame = frame.dropna(
            subset=[
                "date",
                "outlet_id",
                "product_id",
                "quantity_sold",
            ]
        ).copy()

        if frame.empty:
            raise ValueError(
                "No valid canonical client rows remain."
            )

        if (
            frame["quantity_sold"] < 0
        ).any():

            raise ValueError(
                "Client quantity_sold contains negative values."
            )

        frame = frame.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        ).reset_index(
            drop=True
        )

        return frame

    # ============================================================
    # DATA SUFFICIENCY
    # ============================================================

    def _check_sufficiency(
        self,
        df: pd.DataFrame,
    ) -> tuple[bool, str]:
        """
        Determine whether the client package contains
        enough rows and temporal history to train safely.
        """

        rows = len(df)

        if rows < self.minimum_rows:

            return (
                False,
                "INSUFFICIENT_DATA:"
                f"{rows}<"
                f"{self.minimum_rows}",
            )

        unique_dates = (
            df["date"]
            .dt.normalize()
            .nunique()
        )

        required_days = (
            self.validation_days
            + self.evaluation_days
            + 28
        )

        if unique_dates < required_days:

            return (
                False,
                "INSUFFICIENT_HISTORY:"
                f"{unique_dates}<"
                f"{required_days}",
            )

        series_count = (
            df[
                [
                    "outlet_id",
                    "product_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        if series_count == 0:

            return (
                False,
                "NO_OUTLET_PRODUCT_SERIES",
            )

        return (
            True,
            "SUFFICIENT_DATA",
        )

    # ============================================================
    # CHAMPION
    # ============================================================

    def _load_or_create_champion(
        self,
        train_df: pd.DataFrame,
    ):
        """
        Load the existing production champion.

        If none exists, create the first production
        champion from historical training data.
        """

        registry = (
            self.training_adapter.registry
        )

        champion = (
            registry.load_champion()
        )

        if champion is not None:

            logger.info(
                "[LEARNING] Existing production champion loaded."
            )

            return (
                champion,
                False,
            )

        logger.info(
            "[LEARNING] No production champion exists."
        )

        logger.info(
            "[LEARNING] Creating initial production champion."
        )

        (
            champion,
            run_id,
        ) = (
            self.training_adapter.create_initial_champion(
                train_df,
                registry,
            )
        )

        logger.info(
            "[LEARNING] Initial champion created: %s",
            run_id,
        )

        return (
            champion,
            True,
        )

    # ============================================================
    # CANDIDATE TRAINING
    # ============================================================

    def _train_candidate(
        self,
        train_df: pd.DataFrame,
    ):
        """
        Train the genuine Part 9 production candidate.
        """

        logger.info(
            "[LEARNING] Building production features."
        )

        logger.info(
            "[LEARNING] Training Part 9 XGBoost candidate."
        )

        candidate = (
            self.training_adapter.train_model(
                train_df
            )
        )

        logger.info(
            "[LEARNING] Candidate training completed."
        )

        logger.info(
            "[LEARNING] Candidate feature count: %d",
            len(
                candidate.feature_columns
            ),
        )

        logger.info(
            "[LEARNING] Candidate features: %s",
            candidate.feature_columns,
        )

        return candidate

    # ============================================================
    # TRAIN / VALIDATION / EVALUATION
    # ============================================================

    def _run_training(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Execute the complete candidate-learning cycle.

        Train:
            Historical training period.

        Validation:
            Recent historical period used as a sanity check.

        Evaluation:
            Completely later unseen period used for
            candidate/champion comparison.
        """

        (
            train_df,
            validation_df,
            evaluation_df,
        ) = (
            self.training_adapter.chronological_split(
                df,
                validation_days=(
                    self.validation_days
                ),
                evaluation_days=(
                    self.evaluation_days
                ),
            )
        )

        logger.info(
            "[LEARNING] TRAIN rows: %d",
            len(train_df),
        )

        logger.info(
            "[LEARNING] VALIDATION rows: %d",
            len(validation_df),
        )

        logger.info(
            "[LEARNING] UNSEEN EVALUATION rows: %d",
            len(evaluation_df),
        )

        # --------------------------------------------------------
        # Train candidate
        # --------------------------------------------------------

        candidate = (
            self._train_candidate(
                train_df
            )
        )

        # --------------------------------------------------------
        # Candidate validation
        # --------------------------------------------------------

        validation_prediction = (
            self.training_adapter.predict(
                candidate,
                validation_df,
            )
        )

        validation_actual = (
            validation_df[
                "quantity_sold"
            ].to_numpy(
                dtype=float
            )
        )

        validation_metrics = (
            self.training_adapter.metrics(
                validation_actual,
                validation_prediction,
            )
        )

        logger.info(
            "[LEARNING] Candidate validation MAE: %.6f",
            validation_metrics[
                "mae"
            ],
        )

        logger.info(
            "[LEARNING] Candidate validation RMSE: %.6f",
            validation_metrics[
                "rmse"
            ],
        )

        logger.info(
            "[LEARNING] Candidate validation bias: %.6f",
            validation_metrics[
                "bias"
            ],
        )

        # --------------------------------------------------------
        # Champion
        # --------------------------------------------------------

        (
            champion,
            created_initial,
        ) = (
            self._load_or_create_champion(
                train_df
            )
        )

        # --------------------------------------------------------
        # Initial champion
        # --------------------------------------------------------

        if created_initial:

            result = {
                "action": (
                    "INITIAL_CHAMPION_CREATED"
                ),
                "promoted": False,
                "candidate": {
                    "feature_count": len(
                        candidate.feature_columns
                    ),
                    "validation": (
                        validation_metrics
                    ),
                },
                "champion": {
                    "feature_count": len(
                        champion.feature_columns
                    ),
                },
                "evaluation": None,
            }

            return result

        # --------------------------------------------------------
        # Common unseen evaluation
        # --------------------------------------------------------

        logger.info(
            "[LEARNING] Evaluating candidate "
            "and champion on common unseen data."
        )

        # ------------------------------------------------------------
        # COMMON UNSEEN EVALUATION
        # ------------------------------------------------------------

        history_df = pd.concat(
            [
                train_df,
                validation_df,
            ],
            ignore_index=True,
        )

        evaluation = self.training_adapter.evaluate(
            champion,
            candidate,
            evaluation_df,
            history_df=history_df,
        )

        logger.info(
            "[LEARNING] Champion MAE: %.6f",
            evaluation[
                "champion"
            ][
                "mae"
            ],
        )

        logger.info(
            "[LEARNING] Candidate MAE: %.6f",
            evaluation[
                "candidate"
            ][
                "mae"
            ],
        )

        logger.info(
            "[LEARNING] Candidate improvement: %.4f%%",
            evaluation[
                "improvement_pct"
            ],
        )

        logger.info(
            "[LEARNING] Bias change: %.6f",
            evaluation[
                "bias_change"
            ],
        )

        # --------------------------------------------------------
        # Save candidate
        # --------------------------------------------------------

        run_id = (
            "runtime_"
            + uuid.uuid4().hex[:12]
        )

        candidate_path = (
            self.training_adapter.save_candidate(
                model=candidate,
                registry=(
                    self.training_adapter.registry
                ),
                run_id=run_id,
                evaluation=evaluation,
                training_rows=len(
                    train_df
                ),
            )
        )

        # --------------------------------------------------------
        # Promotion
        # --------------------------------------------------------

        if evaluation["passes"]:

            manifest = {
                "run_id": run_id,
                "model_name": (
                    "production_xgboost"
                ),
                "model_type": (
                    "XGBoostDemandModel"
                ),
                "feature_count": len(
                    candidate.feature_columns
                ),
                "feature_columns": (
                    candidate.feature_columns
                ),
                "training_rows": len(
                    train_df
                ),
                "validation_rows": len(
                    validation_df
                ),
                "evaluation_rows": len(
                    evaluation_df
                ),
                "candidate_mae": evaluation[
                    "candidate"
                ][
                    "mae"
                ],
                "candidate_rmse": evaluation[
                    "candidate"
                ][
                    "rmse"
                ],
                "candidate_bias": evaluation[
                    "candidate"
                ][
                    "bias"
                ],
                "champion_mae": evaluation[
                    "champion"
                ][
                    "mae"
                ],
                "champion_rmse": evaluation[
                    "champion"
                ][
                    "rmse"
                ],
                "champion_bias": evaluation[
                    "champion"
                ][
                    "bias"
                ],
                "improvement_pct": evaluation[
                    "improvement_pct"
                ],
                "bias_change": evaluation[
                    "bias_change"
                ],
                "promotion_gate": {
                    "minimum_improvement_pct": (
                        self.training_adapter.MIN_IMPROVEMENT_PCT
                    ),
                    "maximum_bias_worsening": (
                        self.training_adapter.MAX_BIAS_WORSENING
                    ),
                },
                "created_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "promotion_reason": (
                    "Candidate passed empirical "
                    "production gates."
                ),
            }

            timestamp = (
                self.training_adapter.registry.promote(
                    candidate_path,
                    manifest,
                )
            )

            logger.info(
                "[LEARNING] CANDIDATE PROMOTED."
            )

            logger.info(
                "[LEARNING] Promotion timestamp: %s",
                timestamp,
            )

            action = (
                "PROMOTE_CANDIDATE"
            )

            promoted = True

        else:

            logger.info(
                "[LEARNING] CANDIDATE REJECTED."
            )

            logger.info(
                "[LEARNING] Rejection reasons: %s",
                "; ".join(
                    evaluation[
                        "reasons"
                    ]
                ),
            )

            action = (
                "REJECT_CANDIDATE"
            )

            promoted = False

        return {
            "action": action,
            "promoted": promoted,
            "run_id": run_id,
            "candidate_artifact": str(
                candidate_path
            ),
            "candidate": {
                "feature_count": len(
                    candidate.feature_columns
                ),
                "validation": (
                    validation_metrics
                ),
            },
            "evaluation": evaluation,
        }

    # ============================================================
    # SINGLE SCAN
    # ============================================================

    def scan_once(self) -> dict:
        """
        Perform one complete scan of INCOMING/.

        This method is intentionally one-shot.
        The worker calls it repeatedly.
        """

        self._set_status(
            "SCANNING",
            "Checking INCOMING for real client data.",
        )

        packages = (
            self.gateway.discover_packages()
        )

        if not packages:

            self._set_status(
                "WAITING",
                "No real client data package available.",
            )

            return {
                "status": "WAITING",
                "packages": 0,
                "results": [],
            }

        logger.info(
            "[LEARNING] Client data packages detected: %d",
            len(packages),
        )

        results = []

        for package in packages:

            result = (
                self._process_package(
                    package
                )
            )

            results.append(
                result
            )

        return {
            "status": "PROCESSED",
            "packages": len(packages),
            "results": results,
        }

    # ============================================================
    # PACKAGE PROCESSING
    # ============================================================

    def _process_package(
        self,
        package,
    ) -> dict:
        """
        Process one real client package.
        """

        package_id = getattr(
            package,
            "package_id",
            None,
        )

        self.last_package_id = (
            package_id
        )

        logger.info(
            "[LEARNING] Package detected: %s",
            package_id,
        )

        logger.info(
            "[LEARNING] Source: REAL CLIENT DATA"
        )

        # --------------------------------------------------------
        # HARD REAL-DATA GATE
        # --------------------------------------------------------

        source_type = str(
            getattr(
                package,
                "source_type",
                "real_client",
            )
        ).lower()

        if source_type not in {
            "real_client",
            "client",
            "real",
        }:

            self._set_status(
                "REJECTED",
                "Package is not marked as real client data.",
            )

            try:
                rejected = (
                    self.gateway.reject_package(
                        package,
                        reason=(
                            "NON_REAL_SOURCE"
                        ),
                    )
                )

                logger.info(
                    "[LEARNING] Package rejected: %s",
                    rejected,
                )

            except Exception:
                logger.exception(
                    "[LEARNING] Failed to move "
                    "non-real package to REJECTED."
                )

            return {
                "package_id": package_id,
                "action": "REJECTED",
                "reason": "NON_REAL_SOURCE",
            }

        try:

            # ----------------------------------------------------
            # SAMPLE VALIDATION
            # ----------------------------------------------------

            logger.info(
                "[LEARNING] Validating client package schema."
            )

            validation = (
                self.gateway.validate_package_sample(
                    package
                )
            )

            if not validation[
                "passed"
            ]:

                logger.error(
                    "[LEARNING] Client package validation failed."
                )

                rejected = (
                    self.gateway.reject_package(
                        package,
                        reason=(
                            "INVALID_CLIENT_DATA"
                        ),
                    )
                )

                self._set_status(
                    "REJECTED",
                    "Client package failed schema validation.",
                )

                return {
                    "package_id": package_id,
                    "action": "REJECTED",
                    "reason": (
                        "INVALID_CLIENT_DATA"
                    ),
                    "validation": validation,
                    "rejected_to": str(
                        rejected
                    ),
                }

            logger.info(
                "[LEARNING] Package validation: PASS"
            )

            # ----------------------------------------------------
            # FULL LOAD
            # ----------------------------------------------------

            raw = self._load_package(
                package
            )

            logger.info(
                "[LEARNING] Raw rows detected: %d",
                len(raw),
            )

            # ----------------------------------------------------
            # CANONICALIZATION
            # ----------------------------------------------------

            df = (
                self._prepare_client_data(
                    raw
                )
            )

            logger.info(
                "[LEARNING] Canonical rows: %d",
                len(df),
            )

            logger.info(
                "[LEARNING] Outlet/product series: %d",
                df[
                    [
                        "outlet_id",
                        "product_id",
                    ]
                ]
                .drop_duplicates()
                .shape[0],
            )

            logger.info(
                "[LEARNING] Date range: %s → %s",
                df["date"].min(),
                df["date"].max(),
            )

            # ----------------------------------------------------
            # SUFFICIENCY
            # ----------------------------------------------------

            (
                sufficient,
                reason,
            ) = (
                self._check_sufficiency(
                    df
                )
            )

            if not sufficient:

                self._set_status(
                    "WAITING",
                    reason,
                )

                logger.info(
                    "[LEARNING] Training not started: %s",
                    reason,
                )

                # Important:
                # insufficient data stays in INCOMING.
                return {
                    "package_id": package_id,
                    "action": "WAITING",
                    "reason": reason,
                    "rows": len(df),
                }

            # ----------------------------------------------------
            # TRAINING
            # ----------------------------------------------------

            self._set_status(
                "TRAINING",
                "Real client data is sufficient; "
                "starting production candidate training.",
            )

            training_result = (
                self._run_training(
                    df
                )
            )

            self.last_training_result = (
                training_result
            )

            # ----------------------------------------------------
            # SUCCESSFUL PACKAGE PROCESSING
            # ----------------------------------------------------

            processed_destination = (
                self.gateway.process_package(
                    package
                )
            )

            logger.info(
                "[LEARNING] Client package processed: %s",
                processed_destination,
            )

            self._set_status(
                training_result[
                    "action"
                ],
                "Client package processed successfully.",
            )

            return {
                "package_id": package_id,
                "rows": len(df),
                "processed_to": str(
                    processed_destination
                ),
                **training_result,
            }

        except Exception as exc:

            logger.exception(
                "[LEARNING] Package processing failed."
            )

            self._set_status(
                "ERROR",
                str(exc),
            )

            return {
                "package_id": package_id,
                "action": "ERROR",
                "error": str(exc),
            }