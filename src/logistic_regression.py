import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Get root directory and data directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "retailhero_causal_forest_features.csv"

def main():

    # 1. Load data
    print("=" * 60)
    print("Logistic Regression - Simple Interaction Models")
    print("=" * 60)

    print(f"\nLoading data from:")
    print(PROCESSED_DATA_PATH.resolve())

    df = pd.read_csv(PROCESSED_DATA_PATH)

    print(f"\nFull dataset shape: {df.shape}")

    # 2. Train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["treatment"]
    )

    print(f"\nTrain set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # 3. Check treatment proportions
    print("\nTreatment proportions:")

    print("\nFull dataset:")
    print(df["treatment"].value_counts(normalize=True))

    print("\nTrain set:")
    print(train_df["treatment"].value_counts(normalize=True))

    print("\nTest set:")
    print(test_df["treatment"].value_counts(normalize=True))


if __name__ == "__main__":
    main()