# create_dummy_model.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import os
import torch
import torch.nn as nn

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

if __name__ == "__main__":
        os.makedirs("models", exist_ok=True)
        model = SimpleGestureModel()

        # Sesgar a "quieto" para que no parezca "pegado" a una dirección
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.fc[-1].bias[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        torch.save(model.state_dict(), "models/gesture_model.pt")
        print("✅ Modelo dummy guardado en models/gesture_model.pt")
