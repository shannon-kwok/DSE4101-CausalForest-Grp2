from src.data import load_x5, X5_DATA_DIR


def main():
    print("=" * 60)
    print("X5 RetailHero Data Download")
    print("=" * 60)

    print(f"\nData directory:")
    print(X5_DATA_DIR.resolve())

    print("\nDownloading/loading dataset...")

    dataset = load_x5()

    print("\nDataset successfully loaded!")

    print("\nAvailable dataset components:")
    print(dataset.keys())

    print("\nFiles saved in:")

    if X5_DATA_DIR.exists():
        for file in X5_DATA_DIR.rglob("*"):
            if file.is_file():
                print(f"  - {file.relative_to(X5_DATA_DIR)}")
    else:
        print("Could not locate the expected X5 data directory.")

    print("\nDone!")


if __name__ == "__main__":
    main()