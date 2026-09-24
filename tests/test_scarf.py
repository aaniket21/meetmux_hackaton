import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.scarf import SCARF, SCARFLoss, SCARFDataset, train_scarf

def test_scarf_model():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Create dummy dataset
    num_samples = 100
    input_dim = 10
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # 2. Init dataset
    dataset = SCARFDataset(X, corruption_rate=0.4)
    
    # Check corruption
    x, x_corrupted = dataset[0]
    assert x.shape == (input_dim,)
    assert x_corrupted.shape == (input_dim,)
    assert not torch.allclose(x, x_corrupted)
    
    # 3. Init model and loss
    model = SCARF(input_dim=input_dim, emb_dim=16, head_dim=8)
    criterion = SCARFLoss(temperature=1.0)
    
    # Forward pass
    emb, head = model(x.unsqueeze(0))
    emb_c, head_c = model(x_corrupted.unsqueeze(0))
    
    assert emb.shape == (1, 16)
    assert head.shape == (1, 8)
    
    # 4. Train one epoch
    loss_val = train_scarf(model, dataset, epochs=1, batch_size=16, lr=0.01)
    
    assert isinstance(loss_val, float)
    assert loss_val > 0.0

if __name__ == "__main__":
    test_scarf_model()
    print("test_scarf_model passed!")
