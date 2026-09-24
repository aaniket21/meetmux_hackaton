import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.scoring import KNNAnomalyScorer, IFBaselineScorer

def test_anomaly_scoring():
    # Dummy baseline embeddings
    baseline_emb = np.random.randn(100, 16)
    
    # 1. kNN Scorer
    knn_scorer = KNNAnomalyScorer(k=5)
    knn_scorer.fit(baseline_emb)
    
    # Check threshold is set (99th percentile)
    assert hasattr(knn_scorer, 'threshold')
    assert knn_scorer.threshold > 0
    
    # Score new embeddings (some normal, some anomalous)
    normal_emb = baseline_emb[:5] # from baseline, should be normal
    anomalous_emb = np.random.randn(5, 16) * 10 + 10 # shifted/scaled, should be anomalous
    
    test_emb = np.vstack([normal_emb, anomalous_emb])
    scores, is_anomaly = knn_scorer.score(test_emb)
    
    assert len(scores) == 10
    assert len(is_anomaly) == 10
    
    # The normal ones should likely not be anomalies, the others should
    assert sum(is_anomaly[:5]) < sum(is_anomaly[5:])
    
    # 2. IF Baseline Scorer
    # IF works on raw features usually, we can simulate with embeddings or raw
    baseline_feats = np.random.randn(100, 10)
    if_scorer = IFBaselineScorer()
    if_scorer.fit(baseline_feats)
    
    test_feats = np.vstack([baseline_feats[:5], np.random.randn(5, 10) * 10 + 10])
    if_scores, if_is_anomaly = if_scorer.score(test_feats)
    
    assert len(if_scores) == 10
    assert len(if_is_anomaly) == 10

if __name__ == "__main__":
    test_anomaly_scoring()
    print("test_anomaly_scoring passed!")
