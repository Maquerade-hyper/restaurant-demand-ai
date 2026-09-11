from pathlib import Path

from app.data.generators.outlets import generate_outlets
from app.data.generators.products import generate_products


OUTPUT_DIR = Path("data/synthetic")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outlets = generate_outlets()
    products = generate_products()

    outlets.to_csv(OUTPUT_DIR / "outlets.csv", index=False)
    products.to_csv(OUTPUT_DIR / "products.csv", index=False)

    print(f"Generated {len(outlets)} outlets")
    print(f"Generated {len(products)} products")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()