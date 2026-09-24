import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.drift import detect_drift, compute_mmd, compute_psi, compute_severity

def test_drift_detection():
    baseline_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 100),
        "category": np.random.choice(["A", "B", "C"], 100)
    })
    new_batch_df = pd.DataFrame({
        "age": np.random.normal(50, 5, 100),
        "category": np.random.choice(["A", "B"], 100, p=[0.9, 0.1])
    })
    contract = {"columns": {"age": {"type": "numeric"}, "category": {"type": "categorical"}}}

    drift_results = detect_drift(baseline_df, new_batch_df, contract)

    assert drift_results["age"]["drifted"] == True
    assert drift_results["category"]["drifted"] == True
    assert "psi" in drift_results["age"]

def test_psi():
    baseline = pd.Series(np.random.normal(0, 1, 500))
    drifted = pd.Series(np.random.normal(5, 1, 500))
    same = pd.Series(np.random.normal(0, 1, 500))

    psi_drifted = compute_psi(baseline, drifted)
    psi_same = compute_psi(baseline, same)

    assert psi_drifted > psi_same

def test_severity():
    assert compute_severity({}) == "green"
    assert compute_severity({"a": {"drifted": False}}) == "green"
    assert compute_severity({"a": {"drifted": True}, "b": {"drifted": False}, "c": {"drifted": False}}) == "amber"
    assert compute_severity({"a": {"drifted": True}, "b": {"drifted": True}}) == "red"

def test_mmd():
    base_emb = np.random.normal(0, 1, (100, 16))
    drifted_emb = np.random.normal(5, 1, (100, 16))
    mmd_score = compute_mmd(base_emb, drifted_emb)
    assert isinstance(mmd_score, float)
    assert mmd_score > 0.01

if __name__ == "__main__":
    test_drift_detection()
    print("test_drift_detection passed!")
    test_psi()
    print("test_psi passed!")
    test_severity()
    print("test_severity passed!")
    test_mmd()
    print("test_mmd passed!")
