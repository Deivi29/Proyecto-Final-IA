# gesture_model.py
## Nombre Deivi Rodriguez Paulino 
## Matrícula 21-SISN-2-052 

import torch
import torch.nn as nn
import numpy as np

# Modelo simple
class SimpleGestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # salto, izquierda, derecha, quieto
        )

    def forward(self, x):
        return self.fc(x)

# Clasificador
class GestureClassifier:
    def __init__(self):
        self.model = SimpleGestureModel()
        self.model.load_state_dict(torch.load("models/gesture_model.pt", map_location="cpu"))
        self.model.eval()
        self.labels = ["salto", "izquierda", "derecha", "quieto"]

    def predict(self, keypoints: np.ndarray) -> str:
        if len(keypoints) < 34:
            return "quieto"
        
        x = torch.tensor(keypoints[:34], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(x)
            pred = torch.argmax(output, dim=1).item()
            return self.labels[pred]
