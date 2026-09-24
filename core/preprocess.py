import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor:
    def __init__(self, contract: dict):
        self.contract = contract
        self.numeric_features = [col for col, info in contract["columns"].items() if info["type"] == "numeric"]
        self.categorical_features = [col for col, info in contract["columns"].items() if info["type"] == "categorical"]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='<UNK>')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
            
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        self.preprocessor.fit(df)
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform")
            
        # Ensure only columns in contract are passed (ignore extra, fill missing with NaN)
        df_prepared = pd.DataFrame()
        for col in self.contract["columns"].keys():
            if col in df.columns:
                df_prepared[col] = df[col]
            else:
                df_prepared[col] = np.nan
                
        return self.preprocessor.transform(df_prepared)
