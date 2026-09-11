import pandas as pd


def check_foreign_keys(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_column: str,
    parent_column: str,
) -> list[str]:

    if child_column not in child.columns:
        return [f"Missing child column: {child_column}"]

    if parent_column not in parent.columns:
        return [f"Missing parent column: {parent_column}"]

    parent_values = set(
        parent[parent_column].dropna()
    )

    invalid = (
        ~child[child_column].isin(parent_values)
    )

    count = int(invalid.sum())

    if count:
        return [
            f"{child_column}: {count} invalid references"
        ]

    return []


def check_sales_references(
    sales: pd.DataFrame,
    outlets: pd.DataFrame,
    products: pd.DataFrame,
) -> list[str]:

    errors = []

    errors.extend(
        check_foreign_keys(
            sales,
            outlets,
            "outlet_id",
            "outlet_id",
        )
    )

    errors.extend(
        check_foreign_keys(
            sales,
            products,
            "product_id",
            "product_id",
        )
    )

    return errors