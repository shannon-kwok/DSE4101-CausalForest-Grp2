import pandas as pd
from pathlib import Path

# Get root directory and data directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "retailhero_causal_forest_features.csv"

def compute_ate_benchmark(features:pd.DataFrame,
                          treatment_col: str = "treatment",
                          target_col: str = "target") -> dict:


    # Filter out treated and untreated columns
    treated = features.loc[features["treatment"] == 1, target_col]
    control = features.loc[features["treatment"] == 0, target_col]

    # Find the mean of (Y | T=1) and (Y | T=0)
    mean_treated = treated.mean()
    mean_control = control.mean()

    # Find ATE for entire population
    ate_hat = mean_treated - mean_control

    return{
        "n_treated": len(treated),
        "n_control": len(control),
        "mean_treated": mean_treated,
        "mean_control": mean_control,
        "ATE_benchmark": ate_hat
    }


def main():
    print("=" * 60)
    print("ATE Benchmark (simple difference in means)")
    print("=" * 60)

    print(f"\nLoading features from:")
    print(PROCESSED_DATA_PATH.resolve())

    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    results = compute_ate_benchmark(processed_df)

    print(f"\nn_treated = {results['n_treated']}")
    print(f"n_control = {results['n_control']}")
    print(f"P(purchase | treated) = {results['mean_treated']:.4f}")
    print(f"P(purchase | control) = {results['mean_control']:.4f}")
    print(f"\nBenchmark ATE estimate = {results['ATE_benchmark']:.4f}")


if __name__ == "__main__":
    main()
 
