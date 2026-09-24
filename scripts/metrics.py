def evaluate_metrics(ground_truth: list[dict], predicted_anomalies: list[int], predicted_cells: list[dict]) -> dict:
    # Anomaly Detection Metrics (Row level)
    gt_rows = set([gt["row_idx"] for gt in ground_truth])
    pred_rows = set(predicted_anomalies)
    
    true_positives = len(gt_rows.intersection(pred_rows))
    false_positives = len(pred_rows - gt_rows)
    false_negatives = len(gt_rows - pred_rows)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Cell Attribution Metrics
    gt_cells = set([(gt["row_idx"], gt["col_name"]) for gt in ground_truth])
    pred_cell_set = set([(c["row_idx"], c["col"]) for c in predicted_cells])
    
    correct_cells = len(gt_cells.intersection(pred_cell_set))
    cell_accuracy = correct_cells / len(gt_cells) if len(gt_cells) > 0 else 1.0
    
    return {
        "anomaly_precision": float(precision),
        "anomaly_recall": float(recall),
        "anomaly_f1": float(f1),
        "cell_attribution_accuracy": float(cell_accuracy)
    }
