from app.data.generators.outlets import generate_outlets
from app.data.generators.products import generate_products
from app.data.generators.sales import generate_sales


def main():

    outlets = generate_outlets()
    products = generate_products()

    sales = generate_sales(
        outlets,
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

    print(
        f"OUTLETS: {len(outlets)}"
    )

    print(
        f"PRODUCTS: {len(products)}"
    )

    print(
        f"SALES ROWS: {len(sales)}"
    )


if __name__ == "__main__":
    main()