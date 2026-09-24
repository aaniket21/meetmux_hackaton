import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.metrics import evaluate_metrics

def test_metrics_evaluation():
    ground_truth = [
        {"row_idx": 0, "col_name": "age", "fault_type": "drift"},
        {"row_idx": 2, "col_name": "income", "fault_type": "null"}
    ]
    
    # Simulate predicted anomalies (row indices)
    predicted_anomalies = [0, 1, 2] # Row 1 is a false positive
    
    # Simulate predicted bad cells
    predicted_cells = [
        {"row_idx": 0, "col": "age"},
        {"row_idx": 1, "col": "age"},
        {"row_idx": 2, "col": "income"}
    ]
    
    metrics = evaluate_metrics(ground_truth, predicted_anomalies, predicted_cells)
    
    assert "anomaly_precision" in metrics
    assert "anomaly_recall" in metrics
    assert "anomaly_f1" in metrics
    assert "cell_attribution_accuracy" in metrics
    
    # 2 true positives (0, 2), 1 false positive (1)
    # Precision = 2 / 3 = 0.66
    # Recall = 2 / 2 = 1.0
    assert abs(metrics["anomaly_precision"] - 0.666) < 0.01
    assert metrics["anomaly_recall"] == 1.0
    
    # Cell attribution accuracy = 2 correctly identified out of 2 ground truth = 1.0
    assert metrics["cell_attribution_accuracy"] == 1.0

if __name__ == "__main__":
    test_metrics_evaluation()
    print("test_metrics_evaluation passed!")
