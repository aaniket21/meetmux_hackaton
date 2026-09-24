import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

class KNNAnomalyScorer:
    def __init__(self, k: int = 5, threshold_percentile: float = 99.0):
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.nn = NearestNeighbors(n_neighbors=k)
        self.threshold = None
        self.baseline_embeddings = None
        
    def fit(self, baseline_embeddings: np.ndarray):
        self.baseline_embeddings = baseline_embeddings
        self.nn.fit(baseline_embeddings)
        
        # Self-score to find threshold
        # We query for k+1 neighbors because the closest is the point itself (distance 0)
        distances, _ = self.nn.kneighbors(baseline_embeddings, n_neighbors=self.k + 1)
        
        # Mean distance to k nearest neighbors (excluding the point itself)
        mean_distances = distances[:, 1:].mean(axis=1)
        
        self.threshold = np.percentile(mean_distances, self.threshold_percentile)
        
    def score(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Scorer must be fitted before calling score")
            
        distances, _ = self.nn.kneighbors(embeddings, n_neighbors=self.k)
        scores = distances.mean(axis=1)
        
        is_anomaly = scores > self.threshold
        return scores, is_anomaly

class IFBaselineScorer:
    def __init__(self, contamination: float = 0.01):
        self.contamination = contamination
        self.iso = IsolationForest(contamination=self.contamination, random_state=42)
        
    def fit(self, baseline_features: np.ndarray):
        self.iso.fit(baseline_features)
        
    def score(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # IF returns 1 for inliers, -1 for outliers
        # IF score_samples returns negative anomaly score (lower is more anomalous)
        preds = self.iso.predict(features)
        is_anomaly = preds == -1
        
        # Invert scores so higher means more anomalous
        scores = -self.iso.score_samples(features)
        
        return scores, is_anomaly
