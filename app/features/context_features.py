import pandas as pd


def create_context_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    if "temperature" in result.columns:
        result["temperature_squared"] = (
            result["temperature"] ** 2
        )

    if "precipitation" in result.columns:
        result["is_rainy"] = (
            result["precipitation"] > 0
        ).astype(int)

    if "holiday_importance" in result.columns:
        result["holiday_active"] = (
            result["holiday_importance"] > 0
        ).astype(int)

    if "discount" in result.columns:
        result["promotion_active"] = (
            result["discount"] > 0
        ).astype(int)

    if {
        "tourism_index",
        "student_index",
    }.issubset(result.columns):

        result["tourism_student_interaction"] = (
            result["tourism_index"]
            * result["student_index"]
        )

    if {
        "business_index",
        "population_density",
    }.issubset(result.columns):

        result["business_density_interaction"] = (
            result["business_index"]
            * result["population_density"]
        )

    return result