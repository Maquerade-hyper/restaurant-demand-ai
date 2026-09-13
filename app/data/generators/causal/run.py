from pathlib import Path

from app.data.generators.outlets import generate_outlets
from app.data.generators.products import generate_products
from app.data.generators.causal.demand_simulator import (
    CausalDemandSimulator,
)


OUTPUT_DIR = Path("data/synthetic")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outlets = generate_outlets()
    products = generate_products()

    simulator = CausalDemandSimulator(
        seed=42,
        target_stockout_rate=0.05,
    )

    sales, inventory, truth = simulator.generate(
        outlets=outlets,
        products=products,
    )

    outlets.to_csv(
        OUTPUT_DIR / "outlets.csv",
        index=False,
    )

    products.to_csv(
        OUTPUT_DIR / "products.csv",
        index=False,
    )

    sales.to_csv(
        OUTPUT_DIR / "sales.csv",
        index=False,
    )

    inventory.to_csv(
        OUTPUT_DIR / "inventory.csv",
        index=False,
    )

    truth.to_csv(
        OUTPUT_DIR / "demand_truth.csv",
        index=False,
    )

    print("=" * 70)
    print("CAUSAL SYNTHETIC DEMAND GENERATION")
    print("=" * 70)

    print(f"Outlets          : {len(outlets):,}")
    print(f"Products         : {len(products):,}")
    print(f"Sales rows       : {len(sales):,}")
    print(f"Inventory rows   : {len(inventory):,}")
    print(f"Truth rows       : {len(truth):,}")

    print(
        f"Stockout rate    : "
        f"{inventory['stockout'].mean():.4f}"
    )

    print(
        f"True demand      : "
        f"{truth['true_demand'].sum():,.2f}"
    )

    print(
        f"Observed sales   : "
        f"{truth['observed_sales'].sum():,.2f}"
    )

    print(
        f"Lost demand      : "
        f"{truth['lost_demand'].sum():,.2f}"
    )

    print(
        f"Fulfillment rate : "
        f"{truth['observed_sales'].sum() / truth['true_demand'].sum():.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()