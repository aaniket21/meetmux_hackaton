import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.preprocess import Preprocessor
from core.schema import infer_contract

def test_preprocessing():
    # 1. Baseline data
    df_clean = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
        "category": ["A", "B", "A", "B", "A"]
    })
    
    contract = infer_contract(df_clean)
    
    # 2. Fit preprocessor
    preprocessor = Preprocessor(contract)
    preprocessor.fit(df_clean)
    
    # 3. Transform baseline
    X_clean = preprocessor.transform(df_clean)
    
    # age should be standardized (mean 0, std 1)
    # category should be one-hot encoded (2 columns: A, B)
    # Total features: 1 (age) + 2 (category) = 3
    assert X_clean.shape[1] == 3
    
    # 4. Transform new data with unseen category and missing value
    df_new = pd.DataFrame({
        "age": [50, np.nan], 
        "category": ["C", "A"] # C is unseen
    })
    
    X_new = preprocessor.transform(df_new)
    assert X_new.shape[1] == 3
    
    # Missing numeric is typically imputed with median/mean by default or kept as NaN 
    # For now let's assume it gets imputed to 0 after standard scaling or we handle it gracefully.
    # The unseen category C should result in all zeros for the one-hot columns.

if __name__ == "__main__":
    test_preprocessing()
    print("test_preprocessing passed!")
