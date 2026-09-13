from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd

from app.features.advanced_feature_pipeline import (
    AdvancedFeaturePipeline,
)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "sales.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "advanced_features.csv"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PART 22 - ADVANCED FEATURE ENGINEERING")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Outlets: {df['outlet_id'].nunique():,}"
    )

    print(
        f"Products: {df['product_id'].nunique():,}"
    )

    pipeline = AdvancedFeaturePipeline()

    result = pipeline.transform(df)

    # --------------------------------------------------------
    # Validate BEFORE saving
    # --------------------------------------------------------

    validation = pipeline.validate(
        result
    )

    print()
    print("VALIDATION")
    print("-" * 70)

    print(
        f"PASSED: {validation['passed']}"
    )

    print(
        f"Rows: {validation['rows']:,}"
    )

    print(
        f"Feature columns: "
        f"{validation['feature_count']:,}"
    )

    errors = validation.get(
        "errors",
        [],
    )

    print(
        f"ERRORS: {len(errors)}"
    )

    for error in errors[:20]:
        print(
            f" - {error}"
        )

    # --------------------------------------------------------
    # Never save an invalid Part 22 dataset
    # --------------------------------------------------------

    if not validation["passed"]:
        raise RuntimeError(
            "PART 22 VALIDATION FAILED"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
    print(
        "PART 22 PASSED"
    )


if __name__ == "__main__":
    main()