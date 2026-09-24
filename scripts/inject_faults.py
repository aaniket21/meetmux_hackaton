import pandas as pd
import numpy as np
import random

def inject_faults(df: pd.DataFrame, random_seed: int = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Copies a clean dataset and injects known faults per PRD FR9:
    1. Shift a column's mean (drift)
    2. Insert impossible values (e.g. age = -5 or 999)
    3. Rename a column or change a column's type (schema violation)
    4. Randomly set cells to null

    Returns:
        (corrupted_df, ground_truth)
        ground_truth: list of dicts with keys: row_idx, col_name, fault_type, old_value, new_value
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    corrupted_df = df.copy()
    ground_truth = []

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    # Cast numeric columns to float to allow inserting floats/None safely
    for c in numeric_cols:
        corrupted_df[c] = corrupted_df[c].astype(float)

    # 1. Drift — shift a numeric column's mean
    if numeric_cols:
        drift_col = random.choice(numeric_cols)
        col_std = df[drift_col].std()
        shift = col_std * 3 if col_std > 0 else 10
        for idx in corrupted_df.index:
            old_val = corrupted_df.at[idx, drift_col]
            new_val = old_val + shift
            corrupted_df.at[idx, drift_col] = new_val
            ground_truth.append({
                "row_idx": idx,
                "col_name": drift_col,
                "fault_type": "drift",
                "old_value": old_val,
                "new_value": new_val
            })

    # 2. Impossible values — inject into a few rows of a numeric column
    if numeric_cols:
        impossible_col = random.choice(numeric_cols)
        num_impossible = max(1, len(df) // 5)
        impossible_rows = random.sample(list(df.index), min(num_impossible, len(df)))
        for idx in impossible_rows:
            old_val = corrupted_df.at[idx, impossible_col]
            new_val = random.choice([-999, -5, 9999])
            corrupted_df.at[idx, impossible_col] = new_val
            ground_truth.append({
                "row_idx": idx,
                "col_name": impossible_col,
                "fault_type": "impossible_value",
                "old_value": old_val,
                "new_value": new_val
            })

    # 3. Schema violation — rename one column
    all_cols = list(corrupted_df.columns)
    rename_col = random.choice(all_cols)
    new_col_name = rename_col + "_RENAMED"
    corrupted_df = corrupted_df.rename(columns={rename_col: new_col_name})
    ground_truth.append({
        "row_idx": -1,
        "col_name": rename_col,
        "fault_type": "schema_violation",
        "old_value": rename_col,
        "new_value": new_col_name
    })

    # 4. Null injection — randomly set cells to null
    num_nulls = max(1, len(df) // 5)
    available_cols = [c for c in corrupted_df.columns]
    for _ in range(num_nulls):
        null_row = random.choice(list(corrupted_df.index))
        null_col = random.choice(available_cols)
        old_val = corrupted_df.at[null_row, null_col]
        corrupted_df.at[null_row, null_col] = None
        ground_truth.append({
            "row_idx": null_row,
            "col_name": null_col,
            "fault_type": "null_injection",
            "old_value": old_val,
            "new_value": None
        })

    return corrupted_df, ground_truth
