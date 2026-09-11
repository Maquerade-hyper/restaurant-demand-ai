from app.data.pipeline import (
    load_synthetic_dataset,
    save_processed,
)
from app.data.cleaning import clean_dataframe
from app.data.transform import add_date_features


DATASETS = [
    "outlets.csv",
    "products.csv",
]


def process_dataset(filename: str) -> None:
    df = load_synthetic_dataset(filename)

    df = clean_dataframe(df)

    if "date" in df.columns:
        df = add_date_features(df)

    save_processed(
        df,
        filename,
    )

    print(
        f"PROCESSED: {filename} "
        f"({len(df)} rows)"
    )


def main() -> None:
    for filename in DATASETS:
        process_dataset(filename)

    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()