#!/usr/bin/env python
import os
import sys
import django
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# --- Django Environment Setup ---
# Adjust the path below to point to your Django project directory (where manage.py resides)
sys.path.append('/home/your_username/portfolio')  # CHANGE this to your actual path
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gauba.settings')  # CHANGE to your settings module
django.setup()

# --- Import your Django model ---
from gameapi.models import GamePerformance

# --- Import your quantum-inspired model ---
from gameapi.model import QuantumDifficultyAdjuster

# Define where to save the trained model
MODEL_SAVE_PATH = os.path.join('trained_models', 'q_difficulty_adjuster_complex_2qubits.pth')

# ------------------------------------------
# Database-Based Dataset Definition
# ------------------------------------------
class GamePerformanceDataset(Dataset):
    """
    PyTorch Dataset that loads game performance data from the database.
    It uses the GamePerformance model and computes two engineered features:
      - feature1 = bubbles_caught - bubbles_missed
      - feature2 = jellyfish_collisions - mountain_collisions
    The target label is determined by a simple heuristic:
      - If feature1 >= 10, label = 2 (Hard)
      - If feature1 < 0, label = 0 (Easy)
      - Otherwise, label = 1 (Medium)
    """
    def __init__(self):
        # Load all records from the database (consider filtering if dataset is very large)
        self.records = list(GamePerformance.objects.all())
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        feature1 = record.bubbles_caught - record.bubbles_missed
        feature2 = record.jellyfish_collisions - record.mountain_collisions
        features = torch.tensor([feature1, feature2], dtype=torch.float32)
        
        # Simple heuristic for target labeling:
        if feature1 >= 10:
            label = 2  # Hard
        elif feature1 < 0:
            label = 0  # Easy
        else:
            label = 1  # Medium
        
        return features, torch.tensor(label, dtype=torch.long)

# ------------------------------------------
# Training Routine (Using Database Data)
# ------------------------------------------
def train_model():
    # Load dataset from the database
    dataset = GamePerformanceDataset()
    
    if len(dataset) == 0:
        print("No game performance data available in the database.")
        return None

    # Split into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Instantiate the model.
    # For a 2-dimensional input (engineered features) use: n_heads=2, head_dim=1 (2 qubits).
    model = QuantumDifficultyAdjuster(n_heads=2, head_dim=1, num_classes=3)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 20  # Adjust as needed
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += features.size(0)
        
        avg_loss = epoch_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}")
    
    # Evaluate on the validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += features.size(0)
    if val_total > 0:
        print(f"Validation Loss: {val_loss/val_total:.4f}, Val Acc: {val_correct/val_total:.2f}")
    
    # Save the trained model parameters to disk
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to {MODEL_SAVE_PATH}")
    
    return model

if __name__ == "__main__":
    print("Starting weekly training of the quantum-inspired difficulty adjuster using DB data...")
    trained_model = train_model()
    if trained_model is not None:
        print("Weekly training complete.")
