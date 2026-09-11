import pandas as pd


PRODUCTS = [
    ("P001", "Milk", "dairy", "liters"),
    ("P002", "Chicken", "meat", "kg"),
    ("P003", "Beef", "meat", "kg"),
    ("P004", "Rice", "grain", "kg"),
    ("P005", "Flour", "grain", "kg"),
    ("P006", "Cheese", "dairy", "kg"),
    ("P007", "Cooking Oil", "oil", "liters"),
    ("P008", "Eggs", "protein", "pieces"),
    ("P009", "Bread", "bakery", "pieces"),
    ("P010", "Pizza Dough", "bakery", "pieces"),
    ("P011", "Pizza Base 12-inch", "bakery", "pieces"),
    ("P012", "Pizza Base 14-inch", "bakery", "pieces"),
    ("P013", "Pizza Base 16-inch", "bakery", "pieces"),
    ("P014", "Soft Drink", "beverage", "bottles"),
    ("P015", "Bottled Water", "beverage", "bottles"),
    ("P016", "Coffee", "beverage", "kg"),
    ("P017", "Tea", "beverage", "kg"),
    ("P018", "Tomatoes", "vegetable", "kg"),
    ("P019", "Potatoes", "vegetable", "kg"),
    ("P020", "Packaging Box", "packaging", "pieces"),
]


def generate_products() -> pd.DataFrame:
    rows = []

    for product_id, name, category, unit in PRODUCTS:
        rows.append(
            {
                "product_id": product_id,
                "product_name": name,
                "category": category,
                "unit": unit,
                "unit_size": None,
                "dimension": (
                    "12-inch"
                    if product_id == "P011"
                    else "14-inch"
                    if product_id == "P012"
                    else "16-inch"
                    if product_id == "P013"
                    else None
                ),
                "pack_size": 1,
                "conversion_factor": 1,
                "shelf_life_days": (
                    3
                    if category in {"dairy", "meat", "vegetable"}
                    else 30
                ),
            }
        )

    return pd.DataFrame(rows)