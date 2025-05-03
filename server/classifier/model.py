import joblib
import numpy as np
from pathlib import Path

class DiabetesModel:
    def __init__(self, model_path: str):
        full_path = Path(__file__).resolve().parent.parent / model_path
        self.model = joblib.load(full_path)

    def predict(self, features: list):
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        return int(prediction[0])
