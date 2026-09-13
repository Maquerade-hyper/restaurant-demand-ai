from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
# IMPORT
# ============================================================

from app.intelligence.multimodal import (
    MultimodalIntelligenceService,
)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "multimodal_intelligence.csv"
)


# ============================================================
# CONFIG
# ============================================================

MAX_ROWS = 5000


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "PART 27 - MULTIMODAL INTELLIGENCE"
    )
    print("=" * 70)

    print(
        f"Input: {INPUT_PATH}"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    if "date" not in df.columns:
        raise ValueError(
            "date column required"
        )

    if "outlet_id" not in df.columns:
        raise ValueError(
            "outlet_id column required"
        )

    if "product_id" not in df.columns:
        raise ValueError(
            "product_id column required"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # --------------------------------------------------------
    # SELECT SAFE CONTEXT COLUMNS
    # --------------------------------------------------------

    candidate_numeric = [
        "temperature",
        "rainfall",
        "humidity",
        "holiday_active",
        "promotion_active",
        "event_active",
        "tourism_index",
        "student_index",
        "business_density",
        "location_tourism_score",
        "location_student_score",
        "cultural_context_score",
        "cultural_demand_pressure",
        "stockout",
        "fulfillment",
    ]

    available_numeric = [
        column
        for column in candidate_numeric
        if column in df.columns
    ]

    print()
    print(
        f"Rows available: {len(df):,}"
    )

    print(
        f"Numeric context columns: "
        f"{len(available_numeric)}"
    )

    # --------------------------------------------------------
    # SAMPLE DETERMINISTICALLY
    # --------------------------------------------------------

    if len(df) > MAX_ROWS:

        working = (
            df.sort_values(
                [
                    "date",
                    "outlet_id",
                    "product_id",
                ]
            )
            .head(MAX_ROWS)
            .copy()
        )

    else:
        working = df.copy()

    print(
        f"Rows processed: "
        f"{len(working):,}"
    )

    # --------------------------------------------------------
    # SERVICE
    # --------------------------------------------------------

    service = (
        MultimodalIntelligenceService()
    )

    output_rows = []

    # --------------------------------------------------------
    # TEXT REPRESENTATION
    #
    # This creates contextual text from already observed
    # metadata/context columns. It does not use the target.
    # --------------------------------------------------------

    for _, row in working.iterrows():

        numeric_features = {}

        for column in available_numeric:

            value = row[column]

            try:
                value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if np.isfinite(value):
                numeric_features[
                    column
                ] = value

        text_parts = []

        if (
            "holiday_active" in row
            and float(
                row[
                    "holiday_active"
                ]
            ) > 0
        ):
            text_parts.append(
                "holiday"
            )

        if (
            "promotion_active" in row
            and float(
                row[
                    "promotion_active"
                ]
            ) > 0
        ):
            text_parts.append(
                "promotion"
            )

        if (
            "event_active" in row
            and float(
                row[
                    "event_active"
                ]
            ) > 0
        ):
            text_parts.append(
                "event"
            )

        if (
            "temperature" in row
        ):

            temperature = float(
                row[
                    "temperature"
                ]
            )

            if temperature >= 30:
                text_parts.append(
                    "hot weather"
                )

            elif temperature <= 10:
                text_parts.append(
                    "cold weather"
                )

        if (
            "tourism_index" in row
            and float(
                row[
                    "tourism_index"
                ]
            ) >= 0.6
        ):
            text_parts.append(
                "tourism"
            )

        if (
            "student_index" in row
            and float(
                row[
                    "student_index"
                ]
            ) >= 0.6
        ):
            text_parts.append(
                "student"
            )

        text = " ".join(
            text_parts
        )

        result = service.process(
            numeric_features=(
                numeric_features
            ),
            text=text,
            text_source=(
                "structured_context"
            ),
        )

        flat = {
            "outlet_id": row[
                "outlet_id"
            ],
            "product_id": row[
                "product_id"
            ],
            "date": row[
                "date"
            ],
            "text_available": float(
                result[
                    "text_available"
                ]
            ),
            "visual_available": float(
                result[
                    "visual_available"
                ]
            ),
            "modality_count": float(
                result[
                    "modality_count"
                ]
            ),
            "text_quality": float(
                result[
                    "text_quality"
                ]
            ),
            "visual_quality": float(
                result[
                    "visual_quality"
                ]
            ),
        }

        for key, value in (
            result[
                "features"
            ].items()
        ):
            flat[key] = value

        output_rows.append(
            flat
        )

    output = pd.DataFrame(
        output_rows
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    errors = []

    if output.empty:
        errors.append(
            "Output is empty."
        )

    required_columns = {
        "outlet_id",
        "product_id",
        "date",
        "modality_count",
        "text_available",
        "visual_available",
        "mm_fusion_score",
        "mm_quality_score",
    }

    missing = (
        required_columns
        - set(output.columns)
    )

    if missing:
        errors.append(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    if not output.empty:

        for column in output.columns:

            if column in {
                "outlet_id",
                "product_id",
                "date",
            }:
                continue

            values = pd.to_numeric(
                output[column],
                errors="coerce",
            )

            if not np.all(
                np.isfinite(
                    values.fillna(0)
                )
            ):
                errors.append(
                    f"Non-finite values: "
                    f"{column}"
                )

        if (
            output[
                "mm_fusion_score"
            ]
            .min()
            < 0
            or
            output[
                "mm_fusion_score"
            ]
            .max()
            > 1
        ):
            errors.append(
                "Fusion score outside [0,1]."
            )

        if (
            output[
                "mm_quality_score"
            ]
            .min()
            < 0
            or
            output[
                "mm_quality_score"
            ].max()
            > 1
        ):
            errors.append(
                "Quality score outside [0,1]."
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "PART 27 RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Input rows: "
        f"{len(df):,}"
    )

    print(
        f"Output rows: "
        f"{len(output):,}"
    )

    print(
        f"Outlets: "
        f"{output['outlet_id'].nunique()}"
    )

    print(
        f"Products: "
        f"{output['product_id'].nunique()}"
    )

    print(
        f"Text-active rows: "
        f"{int(output['text_available'].sum()):,}"
    )

    print(
        f"Visual-active rows: "
        f"{int(output['visual_available'].sum()):,}"
    )

    print(
        f"Average modality count: "
        f"{output['modality_count'].mean():.4f}"
    )

    print(
        f"Average text quality: "
        f"{output['text_quality'].mean():.4f}"
    )

    print(
        f"Average visual quality: "
        f"{output['visual_quality'].mean():.4f}"
    )

    print(
        f"Average fusion score: "
        f"{output['mm_fusion_score'].mean():.4f}"
    )

    print(
        f"Average multimodal quality: "
        f"{output['mm_quality_score'].mean():.4f}"
    )

    print()

    if errors:

        print(
            "VALIDATION ERRORS"
        )

        for error in errors:
            print(
                f" - {error}"
            )

        print()
        print(
            "PART 27 FAILED"
        )

        raise RuntimeError(
            "Part 27 validation failed."
        )

    print(
        "Validation: PASS"
    )

    print(
        f"Output saved: "
        f"{OUTPUT_PATH}"
    )

    print()
    print(
        "PART 27 MULTIMODAL PIPELINE PASSED"
    )


if __name__ == "__main__":
    main()