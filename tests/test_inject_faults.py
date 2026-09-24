import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.inject_faults import inject_faults

def test_inject_faults_all_types():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    })
    
    corrupted_df, ground_truth = inject_faults(df, random_seed=42)
    
    # Ground truth should not be empty
    assert len(ground_truth) > 0, "Ground truth should have entries"
    
    # Collect all fault types injected
    fault_types = set(gt["fault_type"] for gt in ground_truth)
    
    # All 4 types required by PRD FR9
    assert "drift" in fault_types, "Should inject drift faults"
    assert "impossible_value" in fault_types, "Should inject impossible values"
    assert "null_injection" in fault_types, "Should inject null values"
    # schema_violation (rename/type change) may alter columns
    assert "schema_violation" in fault_types or corrupted_df.columns.tolist() != df.columns.tolist(), \
        "Should inject schema violations (column rename or type change)"
    
    # Every ground truth entry has required fields
    for gt in ground_truth:
        assert "row_idx" in gt
        assert "col_name" in gt
        assert "fault_type" in gt
        assert "old_value" in gt
        assert "new_value" in gt

if __name__ == "__main__":
    test_inject_faults_all_types()
    print("test_inject_faults_all_types passed!")
