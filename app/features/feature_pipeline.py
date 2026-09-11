import pandas as pd

from app.features.time_features import (
    create_time_features,
)
from app.features.lag_features import (
    create_lag_features,
)
from app.features.rolling_features import (
    create_rolling_features,
)
from app.features.context_features import (
    create_context_features,
)


def build_features(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
) -> pd.DataFrame:

    result = df.copy()

    result = create_time_features(
        result
    )

    if target_column in result.columns:

        required = {
            "outlet_id",
            "product_id",
            "date",
        }

        if required.issubset(result.columns):

            result = create_lag_features(
                result,
                target_column=target_column,
            )

            result = create_rolling_features(
                result,
                target_column=target_column,
            )

    result = create_context_features(
        result
    )

    return result