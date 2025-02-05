# gameapi/dataset.py
from torch.utils.data import Dataset
import torch
from .models import GamePerformance

class GamePerformanceDataset(Dataset):
    def __init__(self):
        self.records = GamePerformance.objects.all()
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        # Construct the feature vector.
        features = torch.tensor([
            record.bubbles_caught,
            record.bubbles_missed,
            record.jellyfish_collisions,
            record.mountain_collisions
        ], dtype=torch.float32)
        
        # Generate a target label using a simple heuristic.
        if record.bubbles_caught >= 20:
            label = 2  # Hard
        elif record.bubbles_missed >= 10 or record.jellyfish_collisions > 3:
            label = 0  # Easy
        else:
            label = 1  # Medium
        
        label = torch.tensor(label, dtype=torch.long)
        return features, label
