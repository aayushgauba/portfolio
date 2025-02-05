# gameapi/utils.py
import torch
import os
from .model import QuantumDifficultyAdjuster
from django.conf import settings
MODEL_FILENAME = "q_difficulty_adjuster_complex_2qubits.pth"
MODEL_SAVE_PATH = os.path.join(settings.BASE_DIR, "trained_models", MODEL_FILENAME)
def load_trained_model():
    model = QuantumDifficultyAdjuster(n_heads=2, head_dim=1, num_classes=3)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")