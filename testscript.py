# production.py
import torch
import os
import time
from gameapi.utils import load_trained_model

def sample_prediction(model, sample_features):
    """
    Given a trained model and sample features (a list of 2 numbers),
    perform a prediction and measure inference time.
    Returns:
      - Predicted difficulty (as a string)
      - Prediction probabilities
      - Inference time in milliseconds
    """
    model.eval()
    sample_tensor = torch.tensor([sample_features], dtype=torch.float32)
    start_time = time.time()
    with torch.no_grad():
        logits = model(sample_tensor)
    end_time = time.time()
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    mapping = {0: "Easy", 1: "Medium", 2: "Hard"}
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    return mapping.get(predicted_class, "Unknown"), probabilities, elapsed_time_ms

if __name__ == "__main__":
    # Load the trained model using the utility function.
    model = load_trained_model()
    
    # Define a sample 2-dimensional input.
    # For example, if we combine our game metrics such that:
    # Feature 1 = bubbles_caught - bubbles_missed = 15 - 5 = 10
    # Feature 2 = jellyfish_collisions - mountain_collisions = 2 - 1 = 1
    sample_features = [20, 0]
    prediction, probs, inference_time = sample_prediction(model, sample_features)
    
    print("Sample Input Features:", sample_features)
    print("Predicted Difficulty:", prediction)
    print("Prediction Probabilities:", probs)
    print(f"Inference Time: {inference_time:.2f} ms")
