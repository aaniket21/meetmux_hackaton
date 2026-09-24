import numpy as np

def evaluate_metrics(ground_truth: list[dict], predicted_anomalies: list[int], predicted_cells: list[dict],
                     original_values: dict = None, repaired_values: dict = None,
                     audit_log: list[dict] = None, threshold: float = None,
                     drift_results: dict = None, drift_ground_truth: list[str] = None) -> dict:
    """
    Comprehensive metrics per PRD Section 8:
    - Anomaly detection: precision, recall, F1
    - Cell attribution: % of injected bad cells correctly identified
    - Imputation: RMSE (numeric), accuracy (categorical)
    - Verification: % of repaired rows that drop below threshold
    - Drift: correctly flags drifted columns
    """
    # ── Anomaly Detection ──
    gt_rows = set(gt["row_idx"] for gt in ground_truth)
    pred_rows = set(predicted_anomalies)

    tp = len(gt_rows.intersection(pred_rows))
    fp = len(pred_rows - gt_rows)
    fn = len(gt_rows - pred_rows)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "anomaly_precision": float(precision),
        "anomaly_recall": float(recall),
        "anomaly_f1": float(f1),
    }

    # ── Cell Attribution ──
    gt_cells = set((gt["row_idx"], gt["col_name"]) for gt in ground_truth)
    pred_cell_set = set((c["row_idx"], c["col"]) for c in predicted_cells)

    correct_cells = len(gt_cells.intersection(pred_cell_set))
    cell_accuracy = correct_cells / len(gt_cells) if len(gt_cells) > 0 else 1.0
    result["cell_attribution_accuracy"] = float(cell_accuracy)

    # ── Imputation (RMSE numeric, accuracy categorical) ──
    if original_values and repaired_values:
        numeric_errors = []
        cat_correct = 0
        cat_total = 0
        for key, orig in original_values.items():
            repaired = repaired_values.get(key)
            if repaired is None:
                continue
            if isinstance(orig, (int, float)) and isinstance(repaired, (int, float)):
                numeric_errors.append((orig - repaired) ** 2)
            else:
                cat_total += 1
                if str(orig) == str(repaired):
                    cat_correct += 1

        result["imputation_rmse"] = float(np.sqrt(np.mean(numeric_errors))) if numeric_errors else None
        result["imputation_categorical_accuracy"] = float(cat_correct / cat_total) if cat_total > 0 else None

    # ── Verification ──
    if audit_log and threshold is not None:
        accepted = sum(1 for a in audit_log if a.get("status") == "accepted")
        result["verification_rate"] = float(accepted / len(audit_log)) if len(audit_log) > 0 else 0.0

    # ── Drift Detection ──
    if drift_results and drift_ground_truth:
        flagged_cols = set(col for col, r in drift_results.items() if r.get("drifted", False))
        gt_drift_cols = set(drift_ground_truth)
        drift_tp = len(flagged_cols.intersection(gt_drift_cols))
        drift_fp = len(flagged_cols - gt_drift_cols)
        drift_fn = len(gt_drift_cols - flagged_cols)
        result["drift_precision"] = float(drift_tp / (drift_tp + drift_fp)) if (drift_tp + drift_fp) > 0 else 0.0
        result["drift_recall"] = float(drift_tp / (drift_tp + drift_fn)) if (drift_tp + drift_fn) > 0 else 0.0

    return result
