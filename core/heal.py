import pandas as pd
import numpy as np

def attribute_bad_cells(df_flagged: pd.DataFrame, score_fn, baseline_stats: dict) -> list[dict]:
    bad_cells = []
    
    # Original scores
    original_scores = score_fn(df_flagged)
    
    for idx, row in df_flagged.iterrows():
        orig_score = original_scores[idx] if isinstance(original_scores, dict) or isinstance(original_scores, pd.Series) else original_scores[df_flagged.index.get_loc(idx)]
        
        max_drop = 0
        worst_col = None
        
        for col in df_flagged.columns:
            if col not in baseline_stats:
                continue
                
            # Create a copy and replace with baseline stat
            test_df = df_flagged.loc[[idx]].copy()
            test_df.at[idx, col] = baseline_stats[col]
            
            # Score again
            new_score_arr = score_fn(test_df)
            new_score = new_score_arr[0] if isinstance(new_score_arr, (list, np.ndarray)) else new_score_arr[test_df.index.get_loc(idx)]
            
            drop = orig_score - new_score
            if drop > max_drop:
                max_drop = drop
                worst_col = col
                
        if worst_col is not None and max_drop > 0:
            bad_cells.append({
                "row_idx": idx,
                "col": worst_col,
                "score_drop": float(max_drop)
            })
            
    return bad_cells

def impute_cells(df_flagged: pd.DataFrame, bad_cells: list[dict], knn_impute_fn, score_fn, threshold: float) -> tuple[pd.DataFrame, list[dict]]:
    df_repaired = df_flagged.copy()
    audit_log = []
    
    for cell in bad_cells:
        idx = cell["row_idx"]
        col = cell["col"]
        old_val = df_flagged.at[idx, col]
        
        # Get new value from KNN imputation
        new_val = knn_impute_fn(df_flagged.loc[idx], col)
        
        # Apply repair
        df_repaired.at[idx, col] = new_val
        
        # Verify
        new_score_arr = score_fn(df_repaired.loc[[idx]])
        new_score = new_score_arr[0] if isinstance(new_score_arr, (list, np.ndarray)) else new_score_arr[df_repaired.loc[[idx]].index.get_loc(idx)]
        
        status = "accepted" if new_score < threshold else "human_review"
        
        audit_log.append({
            "row_idx": idx,
            "col": col,
            "old_value": old_val,
            "new_value": new_val,
            "confidence": float(max(0, 1 - (new_score / (threshold + 1e-9)))),
            "status": status,
            "score_after": float(new_score)
        })
        
    return df_repaired, audit_log
