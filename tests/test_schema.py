import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.schema import infer_contract, validate_batch

def test_schema_contract():
    # 1. Create a clean baseline dataset
    df_clean = pd.DataFrame({
        "age": [20, 25, 30, 35, 40], # Numeric
        "category": ["A", "B", "A", "B", "A"], # Categorical
        "score": [1.0, 2.0, 3.0, 4.0, 5.0] # Numeric
    })
    
    # 2. Infer contract
    contract = infer_contract(df_clean)
    
    assert "columns" in contract
    assert "age" in contract["columns"]
    assert contract["columns"]["category"]["type"] == "categorical"
    assert contract["columns"]["age"]["type"] == "numeric"
    assert "allowed_values" in contract["columns"]["category"]
    assert "A" in contract["columns"]["category"]["allowed_values"]
    
    # 3. Create a bad batch with multiple violations
    df_bad = pd.DataFrame({
        "age": [20, 1000], # Out of range
        "category": ["A", "C"], # Unseen category 'C'
        "new_col": [1, 2] # Extra column, and 'score' is missing
    })
    
    # 4. Validate batch
    violations = validate_batch(df_bad, contract)
    
    assert len(violations) > 0
    
    v_types = [v["type"] for v in violations]
    assert "missing_column" in v_types
    assert "extra_column" in v_types
    assert "unseen_category" in v_types
    assert "out_of_range" in v_types

if __name__ == "__main__":
    test_schema_contract()
    print("test_schema_contract passed!")
