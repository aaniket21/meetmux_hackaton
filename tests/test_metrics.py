import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.metrics import evaluate_metrics

def test_metrics_basic():
    ground_truth = [
        {"row_idx": 0, "col_name": "age", "fault_type": "drift"},
        {"row_idx": 2, "col_name": "income", "fault_type": "null"}
    ]
    predicted_anomalies = [0, 1, 2]
    predicted_cells = [
        {"row_idx": 0, "col": "age"},
        {"row_idx": 1, "col": "age"},
        {"row_idx": 2, "col": "income"}
    ]

    metrics = evaluate_metrics(ground_truth, predicted_anomalies, predicted_cells)

    assert abs(metrics["anomaly_precision"] - 0.666) < 0.01
    assert metrics["anomaly_recall"] == 1.0
    assert metrics["cell_attribution_accuracy"] == 1.0

def test_metrics_imputation():
    ground_truth = [{"row_idx": 0, "col_name": "age", "fault_type": "drift"}]

    original_values = {(0, "age"): 30, (1, "cat"): "A"}
    repaired_values = {(0, "age"): 32, (1, "cat"): "A"}

    metrics = evaluate_metrics(
        ground_truth, [0], [{"row_idx": 0, "col": "age"}],
        original_values=original_values, repaired_values=repaired_values
    )
    assert metrics["imputation_rmse"] is not None
    assert metrics["imputation_rmse"] == 2.0
    assert metrics["imputation_categorical_accuracy"] == 1.0

def test_metrics_verification():
    ground_truth = [{"row_idx": 0, "col_name": "age", "fault_type": "drift"}]
    audit_log = [
        {"status": "accepted", "score_after": 1.0},
        {"status": "human_review", "score_after": 5.0}
    ]
    metrics = evaluate_metrics(
        ground_truth, [0], [{"row_idx": 0, "col": "age"}],
        audit_log=audit_log, threshold=3.0
    )
    assert metrics["verification_rate"] == 0.5

def test_metrics_drift():
    ground_truth = [{"row_idx": 0, "col_name": "age", "fault_type": "drift"}]
    drift_results = {"age": {"drifted": True}, "income": {"drifted": False}}
    drift_gt = ["age"]

    metrics = evaluate_metrics(
        ground_truth, [0], [{"row_idx": 0, "col": "age"}],
        drift_results=drift_results, drift_ground_truth=drift_gt
    )
    assert metrics["drift_precision"] == 1.0
    assert metrics["drift_recall"] == 1.0

if __name__ == "__main__":
    test_metrics_basic()
    print("test_metrics_basic passed!")
    test_metrics_imputation()
    print("test_metrics_imputation passed!")
    test_metrics_verification()
    print("test_metrics_verification passed!")
    test_metrics_drift()
    print("test_metrics_drift passed!")
