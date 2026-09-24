import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SCARFDataset(Dataset):
    def __init__(self, data: np.ndarray, corruption_rate: float = 0.6):
        self.data = data
        self.corruption_rate = corruption_rate
        self.num_samples, self.num_features = data.shape
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        x = self.data[idx]
        
        # Corrupt features
        x_corrupted = x.copy()
        num_corrupt = int(self.corruption_rate * self.num_features)
        
        if num_corrupt > 0:
            corrupt_indices = np.random.choice(self.num_features, num_corrupt, replace=False)
            
            # For each feature to corrupt, sample from marginal distribution
            for c_idx in corrupt_indices:
                # Randomly pick another sample to borrow the feature from
                random_sample_idx = np.random.randint(0, self.num_samples)
                x_corrupted[c_idx] = self.data[random_sample_idx, c_idx]
                
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x_corrupted, dtype=torch.float32)


class SCARF(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 256, head_dim: int = 128):
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Projection head
        self.head = nn.Sequential(
            nn.Linear(emb_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim)
        )
        
    def forward(self, x):
        emb = self.encoder(x)
        proj = self.head(emb)
        return emb, proj


class SCARFLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z, z_corrupted):
        # Normalize embeddings
        z = F.normalize(z, dim=1)
        z_corrupted = F.normalize(z_corrupted, dim=1)
        
        # Compute cosine similarity
        batch_size = z.shape[0]
        
        # Concatenate z and z_corrupted
        representations = torch.cat([z, z_corrupted], dim=0)
        
        # Similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Labels for contrastive learning
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        similarity_matrix[mask] = -float('inf')
        
        # Scale by temperature
        logits = similarity_matrix / self.temperature
        
        loss = self.criterion(logits, labels)
        return loss


def train_scarf(model: SCARF, dataset: SCARFDataset, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3) -> float:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = SCARFLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    total_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, x_corrupted in dataloader:
            optimizer.zero_grad()
            
            _, z = model(x)
            _, z_c = model(x_corrupted)
            
            loss = criterion(z, z_c)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        total_loss += epoch_loss / len(dataloader)
        
    return total_loss / epochs
