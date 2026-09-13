import pandas as pd


path = "data/synthetic/inventory.csv"

df = pd.read_csv(path)

accounting_error = (
    df["opening_stock"]
    + df["received_stock"]
    - df["closing_stock"]
    - df["wastage"]
    - pd.read_csv("data/synthetic/sales.csv")[
        "quantity_sold"
    ]
)

print("=" * 60)
print("SYNTHETIC INVENTORY VALIDATION")
print("=" * 60)

print(f"Rows                  : {len(df):,}")
print(
    f"Negative stock values : "
    f"{(df[['opening_stock', 'received_stock', 'closing_stock']] < 0).any().any()}"
)
print(
    f"Stockout records      : "
    f"{int(df['stockout'].sum()):,}"
)
print(
    f"Max accounting error  : "
    f"{accounting_error.abs().max():.10f}"
)