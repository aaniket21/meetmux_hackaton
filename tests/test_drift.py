import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.drift import detect_drift, compute_mmd

def test_drift_detection():
    # 1. Test statistical drift detection
    baseline_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 100), # numeric
        "category": np.random.choice(["A", "B", "C"], 100) # categorical
    })
    
    # New batch with drifted age and category
    new_batch_df = pd.DataFrame({
        "age": np.random.normal(50, 5, 100), # Mean shifted heavily
        "category": np.random.choice(["A", "B"], 100, p=[0.9, 0.1]) # Distribution changed
    })
    
    contract = {
        "columns": {
            "age": {"type": "numeric"},
            "category": {"type": "categorical"}
        }
    }
    
    drift_results = detect_drift(baseline_df, new_batch_df, contract)
    
    assert "age" in drift_results
    assert "category" in drift_results
    
    # Both should flag as drifted due to significant changes
    assert drift_results["age"]["drifted"] == True
    assert drift_results["category"]["drifted"] == True
    assert "p_value" in drift_results["age"]
    assert "p_value" in drift_results["category"]
    
    # 2. Test MMD for embeddings
    base_emb = np.random.normal(0, 1, (100, 16))
    drifted_emb = np.random.normal(5, 1, (100, 16))
    
    mmd_score = compute_mmd(base_emb, drifted_emb)
    
    assert isinstance(mmd_score, float)
    assert mmd_score > 0.01 # Should be > 0 for drifted embeddings
    
if __name__ == "__main__":
    test_drift_detection()
    print("test_drift_detection passed!")
