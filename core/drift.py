import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare

def compute_psi(baseline_series: pd.Series, new_series: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index for a numeric column."""
    baseline_clean = baseline_series.dropna()
    new_clean = new_series.dropna()

    if len(baseline_clean) == 0 or len(new_clean) == 0:
        return 0.0

    # Create buckets from baseline
    breakpoints = np.percentile(baseline_clean, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    base_counts = np.histogram(baseline_clean, bins=breakpoints)[0]
    new_counts = np.histogram(new_clean, bins=breakpoints)[0]

    # Convert to proportions with epsilon to avoid log(0)
    eps = 1e-8
    base_pct = base_counts / base_counts.sum() + eps
    new_pct = new_counts / new_counts.sum() + eps

    psi = np.sum((new_pct - base_pct) * np.log(new_pct / base_pct))
    return float(psi)

def compute_severity(drift_results: dict) -> str:
    """Compute overall drift severity: green, amber, or red."""
    drifted_count = sum(1 for r in drift_results.values() if r.get("drifted", False))
    total = len(drift_results)

    if total == 0:
        return "green"

    ratio = drifted_count / total

    if ratio == 0:
        return "green"
    elif ratio < 0.5:
        return "amber"
    else:
        return "red"

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
            # KS test
            ks_stat, ks_p = ks_2samp(base_series, new_series)
            # PSI
            psi_val = compute_psi(baseline_df[col], new_batch_df[col])
            drifted = bool(ks_p < p_val_threshold) or psi_val > 0.2
            results[col] = {
                "stat": ks_stat,
                "p_value": float(ks_p),
                "psi": psi_val,
                "drifted": drifted,
                "test": "KS+PSI"
            }
        else:
            # Chi-square test
            categories = pd.concat([base_series, new_series]).unique()

            base_counts = base_series.value_counts().reindex(categories, fill_value=0)
            new_counts = new_series.value_counts().reindex(categories, fill_value=0)

            base_probs = base_counts / base_counts.sum()
            expected = base_probs * new_counts.sum()
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
    """MMD using RBF kernel."""
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)

    rx = np.diag(xx)
    ry = np.diag(yy)

    K_xx = np.exp(-gamma * (rx[:, None] + rx[None, :] - 2 * xx))
    K_yy = np.exp(-gamma * (ry[:, None] + ry[None, :] - 2 * yy))
    K_xy = np.exp(-gamma * (rx[:, None] + ry[None, :] - 2 * xy))

    return float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
