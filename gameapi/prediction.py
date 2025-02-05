# gameapi/prediction.py
import torch
from .utils import load_trained_model

def predict_difficulty_from_model(performance_data):
    """
    Predicts difficulty based on performance data using the trained model.
    
    Expects performance_data as a dict with keys:
      - bubbles_caught
      - bubbles_missed
      - jellyfish_collisions
      - mountain_collisions
      
    Computes:
      feature1 = bubbles_caught - bubbles_missed
      feature2 = jellyfish_collisions - mountain_collisions
      
    Returns one of "Easy", "Medium", or "Hard".
    """
    feature1 = performance_data.get("bubbles_caught", 0) - performance_data.get("bubbles_missed", 0)
    feature2 = performance_data.get("jellyfish_collisions", 0) - performance_data.get("mountain_collisions", 0)
    sample = torch.tensor([[feature1, feature2]], dtype=torch.float32)
    model = load_trained_model()
    with torch.no_grad():
        logits = model(sample)
    predicted_class = torch.argmax(logits, dim=1).item()
    mapping = {0: "Easy", 1: "Medium", 2: "Hard"}
    return mapping.get(predicted_class, "Unknown")


