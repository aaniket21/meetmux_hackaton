"""
Generate a sample dataset for demo purposes.
Creates a synthetic tabular dataset, splits into baseline (clean) and test portions,
and uses inject_faults to create a faulty test batch.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.inject_faults import inject_faults

def generate_sample_data(n_rows: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.normal(55000, 15000, n_rows).round(2),
        "hours_per_week": np.random.randint(10, 60, n_rows),
        "education_years": np.random.randint(6, 20, n_rows),
        "occupation": np.random.choice(["Tech", "Sales", "Admin", "Service", "Craft"], n_rows),
        "marital_status": np.random.choice(["Single", "Married", "Divorced"], n_rows, p=[0.4, 0.45, 0.15]),
    })
    return df

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    df = generate_sample_data(n_rows=1000, random_seed=42)

    # Split: 70% baseline, 30% test
    split_idx = int(len(df) * 0.7)
    baseline_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    baseline_df.to_csv(os.path.join(data_dir, "baseline.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test_clean.csv"), index=False)

    # Generate faulty batch
    faulty_df, ground_truth = inject_faults(test_df, random_seed=123)
    faulty_df.to_csv(os.path.join(data_dir, "test_faulty.csv"), index=False)
    pd.DataFrame(ground_truth).to_csv(os.path.join(data_dir, "ground_truth.csv"), index=False)

    print(f"Baseline: {len(baseline_df)} rows -> data/baseline.csv")
    print(f"Test clean: {len(test_df)} rows -> data/test_clean.csv")
    print(f"Test faulty: {len(faulty_df)} rows -> data/test_faulty.csv")
    print(f"Ground truth: {len(ground_truth)} faults -> data/ground_truth.csv")

if __name__ == "__main__":
    main()
