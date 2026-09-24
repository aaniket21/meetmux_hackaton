import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.heal import attribute_bad_cells, impute_cells

def test_healing():
    # Mocking a scoring function
    # It returns a score based on how far 'age' is from 30
    def mock_score_fn(df):
        return np.abs(df['age'].values - 30)

    # Flagged row with a bad age
    df_flagged = pd.DataFrame({"age": [1000], "income": [50000]})
    
    # Baseline stats for replacement
    baseline_stats = {"age": 30, "income": 50000}
    
    # 1. Attribution
    bad_cells = attribute_bad_cells(df_flagged, mock_score_fn, baseline_stats)
    
    assert len(bad_cells) == 1
    assert bad_cells[0]["row_idx"] == 0
    assert bad_cells[0]["col"] == "age"
    
    # 2. Imputation (Mock kNN imputation)
    # Suppose the nearest neighbor in baseline has age=32
    def mock_knn_impute(row, col):
        return 32 if col == "age" else row[col]
        
    df_repaired, audit_log = impute_cells(df_flagged, bad_cells, mock_knn_impute, mock_score_fn, threshold=10)
    
    assert df_repaired.at[0, "age"] == 32
    assert len(audit_log) == 1
    assert audit_log[0]["status"] == "accepted"
    assert audit_log[0]["old_value"] == 1000
    assert audit_log[0]["new_value"] == 32

if __name__ == "__main__":
    test_healing()
    print("test_healing passed!")
