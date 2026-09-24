import os
import sys
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.scarf import SCARF
from core.scoring import KNNAnomalyScorer
from core.persistence import save_pipeline, load_pipeline

def test_save_load_pipeline():
    torch.manual_seed(42)
    np.random.seed(42)
    save_dir = os.path.join(os.path.dirname(__file__), '..', '_test_save')

    # Create model
    model = SCARF(input_dim=10, emb_dim=16, head_dim=8)
    x = torch.randn(1, 10)
    emb_before, _ = model(x)

    # Create scorer
    scorer = KNNAnomalyScorer(k=3)
    scorer.fit(np.random.randn(50, 16))

    contract = {"columns": {"age": {"type": "numeric"}}}

    # Save
    save_pipeline(save_dir, model=model, scorer=scorer, contract=contract)

    assert os.path.exists(os.path.join(save_dir, "scarf_model.pt"))
    assert os.path.exists(os.path.join(save_dir, "scorer.pkl"))
    assert os.path.exists(os.path.join(save_dir, "contract.pkl"))

    # Load
    loaded = load_pipeline(save_dir, model_class=SCARF)

    assert "model" in loaded
    assert "scorer" in loaded
    assert "contract" in loaded

    emb_after, _ = loaded["model"](x)
    assert torch.allclose(emb_before, emb_after, atol=1e-5)

    assert loaded["scorer"].threshold == scorer.threshold
    assert loaded["contract"] == contract

    # Cleanup
    shutil.rmtree(save_dir)

if __name__ == "__main__":
    test_save_load_pipeline()
    print("test_save_load_pipeline passed!")
