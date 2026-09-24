import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare

def detect_drift(baseline_df: pd.DataFrame, new_batch_df: pd.DataFrame, contract: dict, p_val_threshold: float = 0.05) -> dict:
    results = {}
    
    for col, info in contract["columns"].items():
        if col not in baseline_df.columns or col not in new_batch_df.columns:
            continue
            
        base_series = baseline_df[col].dropna()
        new_series = new_batch_df[col].dropna()
        
        if len(base_series) == 0 or len(new_series) == 0:
            continue
            
        if info["type"] == "numeric":
            # Kolmogorov-Smirnov test
            stat, p_value = ks_2samp(base_series, new_series)
            drifted = bool(p_value < p_val_threshold)
            results[col] = {
                "stat": stat,
                "p_value": float(p_value),
                "drifted": drifted,
                "test": "KS"
            }
        else:
            # Chi-square test
            # Align categories
            categories = pd.concat([base_series, new_series]).unique()
            
            base_counts = base_series.value_counts().reindex(categories, fill_value=0)
            new_counts = new_series.value_counts().reindex(categories, fill_value=0)
            
            # Normalize to probabilities and scale to new size
            base_probs = base_counts / base_counts.sum()
            expected = base_probs * new_counts.sum()
            
            # Chi-square requires expected frequencies >= 5 usually, but we'll use a simple fallback
            # Add a small epsilon to avoid division by zero
            expected = expected + 1e-8
            
            stat, p_value = chisquare(f_obs=new_counts, f_exp=expected)
            drifted = bool(p_value < p_val_threshold)
            results[col] = {
                "stat": stat,
                "p_value": float(p_value),
                "drifted": drifted,
                "test": "Chi-Square"
            }
            
    return results

def compute_mmd(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    # MMD using RBF kernel
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)
    
    rx = np.diag(xx)
    ry = np.diag(yy)
    
    K_xx = np.exp(-gamma * (rx[:, None] + rx[None, :] - 2 * xx))
    K_yy = np.exp(-gamma * (ry[:, None] + ry[None, :] - 2 * yy))
    K_xy = np.exp(-gamma * (rx[:, None] + ry[None, :] - 2 * xy))
    
    return float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
