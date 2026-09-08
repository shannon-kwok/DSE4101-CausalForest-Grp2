from pathlib import Path
from sklift.datasets import fetch_x5


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
X5_DATA_DIR = RAW_DATA_DIR / "x5"


def download_x5():
    """Download the raw X5 RetailHero dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading X5 RetailHero dataset...")
    print(f"Destination: {X5_DATA_DIR.resolve()}")

    dataset = fetch_x5(
        data_home=str(RAW_DATA_DIR),
        dest_subdir="x5",
        download_if_missing=True,
    )

    print("Download complete.")

    return dataset


if __name__ == "__main__":
    download_x5()