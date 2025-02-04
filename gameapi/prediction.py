# gameapi/prediction.py

def predict_difficulty(performance_data):
    """
    A dummy function to predict difficulty based on performance metrics.
    performance_data is expected to be a dict with keys:
      - bubbles_caught
      - bubbles_missed
      - jellyfish_collisions
      - mountain_collisions
    This function returns one of "Easy", "Medium", or "Hard".
    Replace this with your quantum-inspired prediction logic as needed.
    """
    # Example heuristic:
    if performance_data.get("bubbles_caught", 0) >= 20:
        return "Hard"
    elif performance_data.get("bubbles_missed", 0) >= 10 or performance_data.get("jellyfish_collisions", 0) > 3:
        return "Easy"
    else:
        return "Medium"
