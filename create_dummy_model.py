# create_dummy_model.py
import torch
import torch.nn as nn
import os

class SimpleGestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 gestos: salto, izquierda, derecha, quieto
        )

    def forward(self, x):
        return self.fc(x)

# Crear modelo
model = SimpleGestureModel()

# Crear carpeta 'models' si no existe
os.makedirs("models", exist_ok=True)

# Guardar modelo
torch.save(model.state_dict(), "models/gesture_model.pt")
print("✅ Modelo dummy guardado en: models/gesture_model.pt")
