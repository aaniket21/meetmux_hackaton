import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from scripts.inject_faults import inject_faults

def test_inject_faults():
    # Create dummy clean data
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 30, 35, 40, 45],
        "income": [50000, 60000, 70000, 80000, 90000],
        "category": ["A", "B", "A", "B", "A"]
    })
    
    corrupted_df, ground_truth = inject_faults(df, random_seed=42)
    
    # Check that faults were injected
    assert not corrupted_df.equals(df), "Corrupted dataframe should be different from the original"
    assert len(ground_truth) > 0, "Ground truth record should not be empty"
    
    # Check that ground truth tracks row, col, and old/new values
    first_fault = ground_truth[0]
    assert "row_idx" in first_fault
    assert "col_name" in first_fault
    assert "fault_type" in first_fault
    assert "old_value" in first_fault
    assert "new_value" in first_fault

if __name__ == "__main__":
    test_inject_faults()
    print("test_inject_faults passed!")
