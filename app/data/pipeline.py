from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
SYNTHETIC_DIR = Path("data/synthetic")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def load_synthetic_dataset(
    filename: str,
) -> pd.DataFrame:
    return load_csv(
        SYNTHETIC_DIR / filename
    )


def save_interim(
    df: pd.DataFrame,
    filename: str,
) -> Path:

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = INTERIM_DIR / filename
    df.to_csv(path, index=False)

    return path


def save_processed(
    df: pd.DataFrame,
    filename: str,
) -> Path:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)

    return path