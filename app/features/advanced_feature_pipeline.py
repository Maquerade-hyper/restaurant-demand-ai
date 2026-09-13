from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.features.advanced_temporal import (
    add_advanced_temporal_features,
)
from app.features.demand_dynamics import (
    add_demand_dynamics,
)
from app.features.outlet_product_features import (
    add_outlet_product_features,
)
from app.features.advanced_context import (
    add_advanced_context_features,
)


FORBIDDEN_FEATURE_COLUMNS = {
    "quantity_sold",
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


IDENTITY_COLUMNS = {
    "date",
    "outlet_id",
    "product_id",
    "unit",
}


class AdvancedFeaturePipeline:
    """
    Part 22E - Unified Advanced Feature Pipeline.

    Builds the complete Part 22 feature set while ensuring that
    forbidden future/target-derived columns do not remain in the
    final transformed dataset.

    quantity_sold is retained as the training target.
    Other forbidden columns are removed.
    """

    def __init__(
        self,
        target_column: str = "quantity_sold",
    ) -> None:
        self.target_column = target_column

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df.empty:
            raise ValueError(
                "Input dataframe cannot be empty"
            )

        required = {
            "outlet_id",
            "product_id",
            "date",
            self.target_column,
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing))
            )

        out = df.copy()

        out["date"] = pd.to_datetime(
            out["date"]
        )

        # --------------------------------------------------------
        # Remove forbidden raw columns BEFORE feature generation.
        #
        # The target itself is retained because it is required
        # for supervised training.
        # --------------------------------------------------------

        forbidden_input_columns = (
            FORBIDDEN_FEATURE_COLUMNS
            - {self.target_column}
        )

        columns_to_drop = [
            column
            for column in forbidden_input_columns
            if column in out.columns
        ]

        if columns_to_drop:
            out = out.drop(
                columns=columns_to_drop
            )

        # --------------------------------------------------------
        # Feature layers
        # --------------------------------------------------------

        out = add_advanced_temporal_features(
            out
        )

        out = add_demand_dynamics(
            out,
            target_column=self.target_column,
        )

        out = add_outlet_product_features(
            out,
            target_column=self.target_column,
        )

        out = add_advanced_context_features(
            out
        )

        # --------------------------------------------------------
        # Final safety filter.
        #
        # This protects the contract even if a feature layer
        # accidentally propagates a forbidden column.
        # --------------------------------------------------------

        final_forbidden = (
            FORBIDDEN_FEATURE_COLUMNS
            - {self.target_column}
        )

        final_drop = [
            column
            for column in final_forbidden
            if column in out.columns
        ]

        if final_drop:
            out = out.drop(
                columns=final_drop
            )

        return out

    def feature_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:

        excluded = (
            FORBIDDEN_FEATURE_COLUMNS
            |
            IDENTITY_COLUMNS
        )

        columns = [
            column
            for column in df.columns
            if column not in excluded
        ]

        return columns

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        transformed = self.transform(
            df
        )

        columns = self.feature_columns(
            transformed
        )

        if not columns:
            raise ValueError(
                "No feature columns generated"
            )

        features = transformed[
            columns
        ].copy()

        # --------------------------------------------------------
        # Numeric contract
        # --------------------------------------------------------

        for column in features.columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce",
            )

        return features

    def validate(
        self,
        transformed: pd.DataFrame,
    ) -> dict:

        errors: list[str] = []

        if transformed.empty:
            errors.append(
                "Feature dataframe is empty"
            )

        # --------------------------------------------------------
        # Forbidden columns
        #
        # quantity_sold is explicitly allowed in the transformed
        # dataset because it is the training label.
        # --------------------------------------------------------

        forbidden_for_matrix = (
            FORBIDDEN_FEATURE_COLUMNS
            - {self.target_column}
        )

        forbidden_present = (
            forbidden_for_matrix
            &
            set(transformed.columns)
        )

        if forbidden_present:
            errors.append(
                "Forbidden columns detected: "
                +
                ", ".join(
                    sorted(forbidden_present)
                )
            )

        # --------------------------------------------------------
        # Feature columns
        # --------------------------------------------------------

        feature_columns = self.feature_columns(
            transformed
        )

        if not feature_columns:
            errors.append(
                "No feature columns available"
            )

        # --------------------------------------------------------
        # Numeric feature contract
        # --------------------------------------------------------

        numeric_failures: list[str] = []

        for column in feature_columns:

            if not pd.api.types.is_numeric_dtype(
                transformed[column]
            ):
                numeric_failures.append(
                    column
                )

        if numeric_failures:
            errors.append(
                "Non-numeric feature columns: "
                +
                ", ".join(
                    numeric_failures
                )
            )

        # --------------------------------------------------------
        # Required identity columns
        # --------------------------------------------------------

        required_identity = {
            "outlet_id",
            "product_id",
            "date",
        }

        missing_identity = (
            required_identity
            - set(transformed.columns)
        )

        if missing_identity:
            errors.append(
                "Missing identity columns: "
                +
                ", ".join(
                    sorted(missing_identity)
                )
            )

        # --------------------------------------------------------
        # Target contract
        # --------------------------------------------------------

        if self.target_column not in transformed.columns:
            errors.append(
                f"Target column missing: "
                f"{self.target_column}"
            )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "rows": int(
                len(transformed)
            ),
            "feature_count": int(
                len(feature_columns)
            ),
            "feature_columns": feature_columns,
        }


def build_advanced_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    pipeline = AdvancedFeaturePipeline()

    return pipeline.transform(
        df
    )


def save_advanced_features(
    df: pd.DataFrame,
    path: str | Path,
) -> Path:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    return path