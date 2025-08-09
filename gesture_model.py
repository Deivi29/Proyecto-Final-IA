# gesture_model.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import os
import torch
import torch.nn as nn
import numpy as np

class SimpleGestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 4)   # [salto, izquierda, derecha, quieto]
        )

    def forward(self, x):
        return self.fc(x)

class GestureClassifier:
    def __init__(self):
        self.labels = ["salto", "izquierda", "derecha", "quieto"]
        self.model = None
        try:
            self.model = SimpleGestureModel()
            self.model.load_state_dict(torch.load("models/gesture_model.pt", map_location="cpu"))
            self.model.eval()
        except Exception:
            # Si falla la carga, usamos solo reglas
            self.model = None

    def _rule_based(self, keypoints: np.ndarray) -> str:
        """
        Regla simple con pose (BlazePose):
        0: nose, 15: left_wrist, 16: right_wrist
        Gesto si muñecas están por encima (y menor) que la nariz.
        """
        if len(keypoints) < 34:
            return "quieto"

        def y(idx):  # helper para y de landmark idx
            return keypoints[idx*2 + 1]

        nose_y = y(0)
        lw_y   = y(15)
        rw_y   = y(16)

        margin = 0.03  # hace la regla menos sensible al ruido

        left_up  = (lw_y + margin) < nose_y
        right_up = (rw_y + margin) < nose_y

        if left_up and right_up:
            return "salto"
        elif left_up:
            return "izquierda"
        elif right_up:
            return "derecha"
        else:
            return "quieto"

    def predict(self, keypoints: np.ndarray) -> str:
        # 1) Regla confiable
        rule_pred = self._rule_based(keypoints)

        # 2) Mantener integración PyTorch (Examen.md)
        if self.model is not None and len(keypoints) >= 34:
            x = torch.tensor(keypoints[:34], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
                model_pred = self.labels[int(torch.argmax(logits, dim=1).item())]
            # Prioriza la regla si no es "quieto"; si es "quieto", usa el modelo
            return rule_pred if rule_pred != "quieto" else model_pred
        else:
            return rule_pred
