
from pathlib import Path

import pandas as pd

from app.intelligence.outlet.validation import (
    validate_outlet_intelligence,
)


ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_profiles.csv"
)

BEHAVIOR_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_demand_behavior.csv"
)

PERFORMANCE_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_performance.csv"
)

SEGMENT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_segments.csv"
)


def main():
    print("=" * 70)
    print("PART 17E - OUTLET INTELLIGENCE VALIDATION")
    print("=" * 70)

    profiles = pd.read_csv(
        PROFILE_PATH
    )

    behavior = pd.read_csv(
        BEHAVIOR_PATH
    )

    performance = pd.read_csv(
        PERFORMANCE_PATH
    )

    segments = pd.read_csv(
        SEGMENT_PATH
    )

    print(
        f"Profiles    : {len(profiles)}"
    )

    print(
        f"Behavior    : {len(behavior)}"
    )

    print(
        f"Performance : {len(performance)}"
    )

    print(
        f"Segments    : {len(segments)}"
    )

    result = validate_outlet_intelligence(
        profiles=profiles,
        behavior=behavior,
        performance=performance,
        segments=segments,
    )

    print()
    print("VALIDATION RESULTS")

    for name, validation in (
        result["results"].items()
    ):
        status = (
            "PASS"
            if validation["passed"]
            else "FAIL"
        )

        print(
            f"{name.upper():<15}: "
            f"{status}"
        )

        for error in validation["errors"]:
            print(
                f"  ERROR: {error}"
            )

    print()
    print(
        f"Outlet count : "
        f"{result['outlet_count']}"
    )

    print(
        f"Total errors : "
        f"{len(result['errors'])}"
    )

    print()

    if not result["passed"]:
        print(
            "PART 17E VALIDATION : FAILED"
        )
        raise SystemExit(1)

    print(
        "PART 17E VALIDATION : PASSED"
    )


if __name__ == "__main__":
    main()

