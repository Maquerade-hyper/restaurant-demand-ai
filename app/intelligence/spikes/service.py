from __future__ import annotations

import numpy as np
import pandas as pd

from app.intelligence.spikes.detection import (
    detect_demand_spikes,
)
from app.intelligence.spikes.severity import (
    classify_spike_severity,
)
from app.intelligence.spikes.drivers import (
    analyze_spike_drivers,
)
from app.intelligence.spikes.persistence import (
    analyze_spike_persistence,
)


FORBIDDEN_COLUMNS = {
    "revenue",
    "actual_demand",
    "true_demand",
    "lost_demand",
    "future_demand",
    "closing_stock",
    "demand_truth",
    "lost_demand_truth",
    "prediction",
    "target",
}


class DemandSpikeIntelligenceService:
    """
    Part 23 - Unified Demand Spike Intelligence.

    Pipeline:

        Detection
            ↓
        Severity
            ↓
        Context / Driver Signals
            ↓
        Persistence / Recovery
            ↓
        Validation
    """

    def __init__(
        self,
        demand_column: str = "quantity_sold",
        baseline_window: int = 28,
        minimum_history: int = 7,
    ) -> None:

        if baseline_window < 2:
            raise ValueError(
                "baseline_window must be >= 2"
            )

        if minimum_history < 1:
            raise ValueError(
                "minimum_history must be >= 1"
            )

        self.demand_column = demand_column
        self.baseline_window = baseline_window
        self.minimum_history = minimum_history

    # ============================================================
    # TRANSFORM
    # ============================================================

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the complete Part 23 pipeline.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame"
            )

        if df.empty:
            raise ValueError(
                "Input dataframe cannot be empty"
            )

        required = {
            "outlet_id",
            "product_id",
            "date",
            self.demand_column,
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Missing required input columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        result = df.copy()

        # --------------------------------------------------------
        # 23A Detection
        # --------------------------------------------------------

        result = detect_demand_spikes(
            result,
            demand_column=self.demand_column,
            baseline_window=self.baseline_window,
            minimum_history=self.minimum_history,
        )

        # --------------------------------------------------------
        # 23B Severity
        # --------------------------------------------------------

        result = classify_spike_severity(
            result
        )

        # --------------------------------------------------------
        # 23C Drivers
        # --------------------------------------------------------

        result = analyze_spike_drivers(
            result
        )

        # --------------------------------------------------------
        # 23D Persistence
        # --------------------------------------------------------

        result = analyze_spike_persistence(
            result
        )

        return result

    # ============================================================
    # FEATURE COLUMNS
    # ============================================================

    def feature_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:

        excluded = (
            FORBIDDEN_COLUMNS
            |
            {
                self.demand_column,
                "date",
                "outlet_id",
                "product_id",
                "unit",

                # Labels / intelligence outputs
                "is_demand_spike",
                "spike_severity",
                "spike_priority",
                "spike_pattern",
                "spike_primary_context",
                "spike_driver_class",
            }
        )

        return [
            column
            for column in df.columns
            if column not in excluded
        ]

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Validate the complete Part 23 output.
        """

        errors: list[str] = []

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            return {
                "passed": False,
                "errors": [
                    "Output must be pandas DataFrame"
                ],
                "rows": 0,
                "spike_rows": 0,
                "feature_count": 0,
            }

        if df.empty:
            errors.append(
                "Output dataframe is empty"
            )

        required = {
            "outlet_id",
            "product_id",
            "date",
            self.demand_column,

            "spike_baseline",
            "spike_baseline_std",
            "spike_baseline_median",
            "spike_history_count",

            "spike_absolute_lift",
            "spike_ratio",
            "spike_percentage_lift",
            "spike_z_score",
            "spike_robust_score",
            "spike_score",

            "spike_history_ready",
            "is_demand_spike",

            "spike_severity",
            "spike_severity_score",
            "spike_priority",

            "spike_holiday_signal",
            "spike_promotion_signal",
            "spike_event_signal",
            "spike_weather_signal",
            "spike_context_signal_count",
            "spike_context_intensity",
            "spike_primary_context",
            "spike_multi_context",
            "spike_driver_class",

            "days_since_previous_spike",
            "spike_episode_id",
            "spike_episode_length",
            "spike_pattern",
            "spike_recovery_day",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            errors.append(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        # --------------------------------------------------------
        # Forbidden columns
        # --------------------------------------------------------

        # --------------------------------------------------------
        # --------------------------------------------------------
        # Forbidden feature contract
        # --------------------------------------------------------
        #
        # Source datasets may legitimately contain operational or
        # target/truth columns. They must simply never be exposed
        # through feature_columns().
        #
        # Therefore we validate the actual feature set rather than
        # rejecting the complete source dataframe.

        features = set(
            self.feature_columns(df)
        )

        forbidden_features = (
            FORBIDDEN_COLUMNS
            &
            features
        )

        if forbidden_features:
            errors.append(
                "Forbidden columns exposed as features: "
                + ", ".join(
                    sorted(forbidden_features)
                )
            )

        # --------------------------------------------------------
        # Boolean contract
        # --------------------------------------------------------

        for column in [
            "is_demand_spike",
            "spike_history_ready",
            "spike_multi_context",
            "spike_recovery_day",
        ]:

            if column in df.columns:

                if not pd.api.types.is_bool_dtype(
                    df[column]
                ):
                    errors.append(
                        f"{column} must be boolean"
                    )

        # --------------------------------------------------------
        # Numeric contracts
        # --------------------------------------------------------

        numeric_columns = [
            "spike_baseline",
            "spike_baseline_std",
            "spike_baseline_median",
            "spike_history_count",
            "spike_absolute_lift",
            "spike_ratio",
            "spike_percentage_lift",
            "spike_z_score",
            "spike_robust_score",
            "spike_score",
            "spike_severity_score",
            "spike_context_signal_count",
            "spike_context_intensity",
            "spike_episode_id",
            "spike_episode_length",
        ]

        for column in numeric_columns:

            if column not in df.columns:
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if np.isinf(
                values
            ).any():

                errors.append(
                    f"Infinite values detected in {column}"
                )

        # --------------------------------------------------------
        # Valid severity values
        # --------------------------------------------------------

        valid_severity = {
            "none",
            "moderate",
            "major",
            "extreme",
        }

        if "spike_severity" in df.columns:

            invalid = (
                set(
                    df["spike_severity"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                -
                valid_severity
            )

            if invalid:

                errors.append(
                    "Invalid spike severity values: "
                    +
                    ", ".join(
                        sorted(invalid)
                    )
                )

        # --------------------------------------------------------
        # Severity consistency
        # --------------------------------------------------------

        if {
            "is_demand_spike",
            "spike_severity",
        }.issubset(df.columns):

            invalid_none = (
                ~df["is_demand_spike"].astype(bool)
            ) & (
                df["spike_severity"]
                != "none"
            )

            if invalid_none.any():

                errors.append(
                    "Non-spike rows must have severity='none'"
                )

        # --------------------------------------------------------
        # Spike score consistency
        # --------------------------------------------------------

        if {
            "is_demand_spike",
            "spike_score",
        }.issubset(df.columns):

            negative_score = (
                pd.to_numeric(
                    df["spike_score"],
                    errors="coerce",
                )
                < 0
            )

            if negative_score.any():

                errors.append(
                    "spike_score cannot be negative"
                )

        # --------------------------------------------------------
        # History consistency
        # --------------------------------------------------------

        if {
            "spike_history_count",
            "spike_history_ready",
        }.issubset(df.columns):

            expected_ready = (
                pd.to_numeric(
                    df["spike_history_count"],
                    errors="coerce",
                )
                >= self.minimum_history
            )

            actual_ready = (
                df["spike_history_ready"]
                .astype(bool)
            )

            if not (
                expected_ready
                == actual_ready
            ).all():

                errors.append(
                    "spike_history_ready is inconsistent "
                    "with spike_history_count"
                )

        # --------------------------------------------------------
        # Date contract
        # --------------------------------------------------------

        if "date" in df.columns:

            if not pd.api.types.is_datetime64_any_dtype(
                df["date"]
            ):
                errors.append(
                    "date must be datetime"
                )

        # --------------------------------------------------------
        # Duplicate series/date contract
        # --------------------------------------------------------

        identity = [
            "outlet_id",
            "product_id",
            "date",
        ]

        if all(
            column in df.columns
            for column in identity
        ):

            duplicates = (
                df.duplicated(
                    subset=identity
                )
            )

            if duplicates.any():

                errors.append(
                    "Duplicate outlet-product-date rows detected"
                )

        # --------------------------------------------------------
        # Result
        # --------------------------------------------------------

        spike_rows = 0

        if "is_demand_spike" in df.columns:

            spike_rows = int(
                df[
                    "is_demand_spike"
                ]
                .astype(bool)
                .sum()
            )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "rows": int(len(df)),
            "spike_rows": spike_rows,
            "feature_count": int(
                len(
                    self.feature_columns(df)
                )
            ),
        }

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(
        self,
        df: pd.DataFrame,
    ) -> dict:

        if df.empty:
            return {
                "rows": 0,
                "spikes": 0,
                "spike_rate": 0.0,
                "moderate_spikes": 0,
                "major_spikes": 0,
                "extreme_spikes": 0,
                "persistent_spikes": 0,
                "repeated_spikes": 0,
                "isolated_spikes": 0,
                "recovery_days": 0,
            }

        spike = (
            df["is_demand_spike"]
            .astype(bool)
        )

        severity = df[
            "spike_severity"
        ]

        pattern = df[
            "spike_pattern"
        ]

        recovery = df[
            "spike_recovery_day"
        ].astype(bool)

        return {
            "rows": int(
                len(df)
            ),

            "spikes": int(
                spike.sum()
            ),

            "spike_rate": float(
                spike.mean()
            ),

            "moderate_spikes": int(
                (
                    severity
                    == "moderate"
                ).sum()
            ),

            "major_spikes": int(
                (
                    severity
                    == "major"
                ).sum()
            ),

            "extreme_spikes": int(
                (
                    severity
                    == "extreme"
                ).sum()
            ),

            "persistent_spikes": int(
                (
                    pattern
                    == "persistent"
                ).sum()
            ),

            "repeated_spikes": int(
                (
                    pattern
                    == "repeated"
                ).sum()
            ),

            "isolated_spikes": int(
                (
                    pattern
                    == "isolated"
                ).sum()
            ),

            "recovery_days": int(
                recovery.sum()
            ),
        }