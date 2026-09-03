from pathlib import Path
from sklift.datasets import fetch_x5


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
X5_DATA_DIR = RAW_DATA_DIR / "x5"


def load_x5():
    """
    Download and load the X5 RetailHero dataset.

    The raw dataset is stored in:
        data/raw/x5/
    """

    # Make sure the raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_x5(
        data_home=str(RAW_DATA_DIR),
        dest_subdir="x5",
        download_if_missing=True,
    )

    return dataset