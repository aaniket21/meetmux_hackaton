import pandas as pd
import numpy as np

def infer_contract(df: pd.DataFrame) -> dict:
    contract = {"columns": {}}
    for col in df.columns:
        col_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        null_rate = df[col].isnull().mean()
        
        col_info = {
            "type": col_type,
            "null_rate": float(null_rate)
        }
        
        if col_type == "numeric":
            # 0.5th to 99.5th percentile
            clean_series = df[col].dropna()
            if not clean_series.empty:
                col_info["min_allowed"] = float(np.percentile(clean_series, 0.5))
                col_info["max_allowed"] = float(np.percentile(clean_series, 99.5))
            else:
                col_info["min_allowed"] = None
                col_info["max_allowed"] = None
        else:
            # allowed categories
            col_info["allowed_values"] = df[col].dropna().unique().tolist()
            
        contract["columns"][col] = col_info
        
    return contract

def validate_batch(df: pd.DataFrame, contract: dict) -> list[dict]:
    violations = []
    expected_cols = set(contract["columns"].keys())
    actual_cols = set(df.columns)
    
    # Missing columns
    for col in expected_cols - actual_cols:
        violations.append({"type": "missing_column", "column": col, "details": f"Missing expected column {col}"})
        
    # Extra columns
    for col in actual_cols - expected_cols:
        violations.append({"type": "extra_column", "column": col, "details": f"Found extra column {col}"})
        
    # Check each column in both
    for col in expected_cols.intersection(actual_cols):
        col_info = contract["columns"][col]
        series = df[col]
        
        # Type mismatch check (basic)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if (col_info["type"] == "numeric" and not is_numeric) or (col_info["type"] == "categorical" and is_numeric):
             violations.append({"type": "type_mismatch", "column": col, "details": f"Expected {col_info['type']}"})
             
        # Null rate spikes (simple check: if null rate > baseline + 0.2)
        null_rate = series.isnull().mean()
        if null_rate > col_info["null_rate"] + 0.2:
            violations.append({"type": "null_rate_spike", "column": col, "details": f"Null rate {null_rate:.2f} > expected {col_info['null_rate']:.2f}"})
            
        if col_info["type"] == "numeric" and is_numeric:
            clean_series = series.dropna()
            if not clean_series.empty and col_info["min_allowed"] is not None:
                out_of_range = clean_series[(clean_series < col_info["min_allowed"]) | (clean_series > col_info["max_allowed"])]
                if len(out_of_range) > 0:
                    violations.append({"type": "out_of_range", "column": col, "details": f"Found {len(out_of_range)} out of range values"})
        
        elif col_info["type"] == "categorical" and not is_numeric:
            clean_series = series.dropna()
            unseen = set(clean_series.unique()) - set(col_info.get("allowed_values", []))
            if unseen:
                violations.append({"type": "unseen_category", "column": col, "details": f"Found unseen categories: {unseen}"})
                
    return violations
