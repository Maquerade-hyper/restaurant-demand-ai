from app.data.generators.outlets import generate_outlets
from app.data.generators.products import generate_products
from app.data.generators.sales import generate_sales
from app.data.generators.inventory import generate_inventory


def main():

    outlets = generate_outlets()
    products = generate_products()

    sales = generate_sales(
        outlets,
        products,
    )

    inventory = generate_inventory(
        sales,
        products,
    )

    outlets.to_csv(
        "data/synthetic/outlets.csv",
        index=False,
    )

    products.to_csv(
        "data/synthetic/products.csv",
        index=False,
    )

    sales.to_csv(
        "data/synthetic/sales.csv",
        index=False,
    )

    inventory.to_csv(
        "data/synthetic/inventory.csv",
        index=False,
    )

    print(
        f"OUTLETS: {len(outlets)}"
    )

    print(
        f"PRODUCTS: {len(products)}"
    )

    print(
        f"SALES ROWS: {len(sales)}"
    )

    print(
        f"INVENTORY ROWS: {len(inventory)}"
    )


if __name__ == "__main__":
    main()