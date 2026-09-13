
from pathlib import Path

import pandas as pd

from app.intelligence.outlet.clustering import (
    build_outlet_segments,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_performance.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_segments.csv"
)


def main():
    print("=" * 70)
    print("PART 17D - OUTLET CLUSTERING & SEGMENTATION")
    print("=" * 70)

    performance = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Performance outlets : "
        f"{len(performance)}"
    )

    (
        clustered,
        profiles,
        evaluation,
        metadata,
    ) = build_outlet_segments(
        performance,
        min_clusters=2,
        max_clusters=min(
            8,
            len(performance) - 1,
        ),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clustered.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("CLUSTER EVALUATION")

    print(
        evaluation[
            [
                "n_clusters",
                "silhouette_score",
                "min_cluster_size",
                "max_cluster_size",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"SELECTED K          : "
        f"{metadata['best_k']}"
    )

    print(
        f"BEST SILHOUETTE     : "
        f"{metadata['silhouette_score']:.4f}"
    )

    print()
    print("SEGMENT PROFILES")

    print(
        profiles[
            [
                "cluster_id",
                "outlet_count",
                "segment_name",
                "average_daily_demand",
                "coefficient_variation",
                "fulfillment_rate",
                "lost_demand_rate",
                "stockout_product_day_rate",
                "opportunity_index",
                "risk_index",
            ]
        ]
        .sort_values("cluster_id")
        .to_string(index=False)
    )

    print()
    print("SEGMENT DISTRIBUTION")

    print(
        clustered[
            "segment_name"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("OUTPUT")
    print(
        f"Rows saved : {len(clustered)}"
    )
    print(
        f"Output     : {OUTPUT_PATH}"
    )

    print()
    print(
        "PART 17D RUN : PASSED"
    )


if __name__ == "__main__":
    main()

