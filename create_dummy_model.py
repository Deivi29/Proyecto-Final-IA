# gesture_model.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import os, json
import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = "models"
CKPT_PATH = os.path.join(MODELS_DIR, "gesture_model.pt")
STATS_PATH = os.path.join(MODELS_DIR, "feature_stats.npz")
META_PATH  = os.path.join(MODELS_DIR, "meta.json")

class SimpleGestureModel(nn.Module):
    def __init__(self, in_dim=34, hidden=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class GestureClassifier:
    def __init__(self):
        # Defaults por si faltan archivos (modo seguro)
        self.pose_idxs = None
        self.mean = None
        self.std = None
        self.classes = ["salto","izquierda","derecha","quieto"]
        in_dim = 34

        # Cargar meta y stats
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.classes = meta.get("classes", self.classes)
            in_dim = meta.get("n_features", in_dim)

        if os.path.exists(STATS_PATH):
            d = np.load(STATS_PATH)
            self.mean = d["mean"]      # (1, in_dim)
            self.std  = d["std"]       # (1, in_dim)
            self.pose_idxs = d["pose_idxs"].astype(np.int32)  # (17,)
        else:
            # Si no hay stats, asumimos 17 puntos fijos [por compat]
            self.pose_idxs = np.array([0,5,6,11,12,13,14,15,16,23,24,25,26,27,28,29,30], dtype=np.int32)
            self.mean = np.zeros((1, in_dim), dtype=np.float32)
            self.std  = np.ones((1, in_dim), dtype=np.float32)

        # Modelo
        self.model = SimpleGestureModel(in_dim=in_dim, hidden=64, num_classes=len(self.classes))
        if os.path.exists(CKPT_PATH):
            self.model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"), strict=False)
        self.model.eval()

    def _select_and_scale(self, keypoints_xy: np.ndarray) -> np.ndarray:
        """
        keypoints_xy: vector (2*33) o (66,) de MediaPipe Pose (x0,y0,x1,y1,...,x32,y32)
        Selecciona los 17 índices usados y normaliza con mean/std del entrenamiento.
        """
        kp = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1)  # (>=66,)
        if kp.size < 66:
            # Si el caller ya envía (34,), respetar:
            if kp.size == 34:
                x = kp[None, :]
            else:
                # Fallback: cero
                x = np.zeros((1, 34), dtype=np.float32)
        else:
            xy = kp.reshape(-1, 2)          # (n, 2)
            sel = xy[self.pose_idxs]        # (17, 2)
            x = sel.reshape(1, -1)          # (1, 34)

        x = (x - self.mean) / (self.std + 1e-6)
        return x.astype(np.float32)

    def predict(self, keypoints_xy: np.ndarray) -> str:
        x = self._select_and_scale(keypoints_xy)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x))
            idx = int(torch.argmax(logits, dim=1).item())
        return self.classes[idx]
