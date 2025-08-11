# train_gesture.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import os, json, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================
# Configuración de landmarks (MediaPipe Pose)
# Usaremos 17 puntos (x,y) => 34 features
# =========================================
# Índices elegidos (17): nariz, ojos, hombros, codos, muñecas, caderas, rodillas, tobillos
POSE_IDXS = [0, 5, 6, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
CLASSES = ["salto", "izquierda", "derecha", "quieto"]
N_FEATURES = len(POSE_IDXS) * 2  # 34

# =========================
# Modelo simple (MLP)
# =========================
class SimpleGestureModel(nn.Module):
    def __init__(self, in_dim=N_FEATURES, hidden=64, num_classes=4):
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

# =========================
# Dataset sintético
# =========================
def base_skeleton():
    """
    Devuelve un esqueleto neutral en [0..1] (x,y) para los 33 landmarks de MediaPipe.
    Solo llenamos lo necesario; el resto se aproxima.
    """
    sk = np.zeros((33, 2), dtype=np.float32)

    # Centro del cuerpo aproximado
    cx, cy = 0.5, 0.6
    shoulder_span = 0.18
    hip_span = 0.16
    arm_len_y = 0.16
    forearm_len_y = 0.16
    leg_len_y = 0.25
    shin_len_y = 0.25

    # Nariz, ojos
    sk[0] = [cx, cy - 0.32]
    sk[5] = [cx - 0.03, cy - 0.30]  # ojo izq
    sk[6] = [cx + 0.03, cy - 0.30]  # ojo der

    # Hombros
    sk[11] = [cx - shoulder_span/2, cy - 0.12]
    sk[12] = [cx + shoulder_span/2, cy - 0.12]

    # Codos
    sk[13] = [sk[11][0], sk[11][1] + arm_len_y]
    sk[14] = [sk[12][0], sk[12][1] + arm_len_y]

    # Muñecas
    sk[15] = [sk[13][0], sk[13][1] + forearm_len_y]
    sk[16] = [sk[14][0], sk[14][1] + forearm_len_y]

    # Caderas
    sk[23] = [cx - hip_span/2, cy + 0.02]
    sk[24] = [cx + hip_span/2, cy + 0.02]

    # Rodillas
    sk[25] = [sk[23][0], sk[23][1] + leg_len_y]
    sk[26] = [sk[24][0], sk[24][1] + leg_len_y]

    # Tobillos
    sk[27] = [sk[25][0], sk[25][1] + shin_len_y]
    sk[28] = [sk[26][0], sk[26][1] + shin_len_y]

    # Talones (aprox)
    sk[29] = [sk[27][0], sk[27][1] + 0.03]
    sk[30] = [sk[28][0], sk[28][1] + 0.03]
    return sk

def add_noise(skel, scale_xy=(0.015, 0.02), jitter=0.01):
    """Pequeñas variaciones para simular personas/poses distintas."""
    out = skel.copy()
    sx = 1.0 + np.random.uniform(-scale_xy[0], scale_xy[0])
    sy = 1.0 + np.random.uniform(-scale_xy[1], scale_xy[1])
    out[:, 0] = (out[:, 0] - 0.5) * sx + 0.5
    out[:, 1] = (out[:, 1] - 0.5) * sy + 0.5
    out += np.random.normal(0, jitter, size=out.shape).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    return out

def apply_label_transform(skel, label):
    """
    Aplica transformaciones simples para simular los gestos:
    - salto: muñecas más arriba del nivel hombro.
    - izquierda: muñeca izquierda hacia fuera en X.
    - derecha: muñeca derecha hacia fuera en X.
    - quieto: mínimos cambios.
    """
    s = skel.copy()
    # Hombros, muñecas
    ls, rs = s[11], s[12]
    lw, rw = s[15], s[16]
    shoulder_y = (ls[1] + rs[1]) / 2.0
    shoulder_span = (rs[0] - ls[0])

    if label == "salto":
        # Subir muñecas
        delta_y = np.random.uniform(0.10, 0.18)
        lw[1] = max(0.0, shoulder_y - delta_y)
        rw[1] = max(0.0, shoulder_y - delta_y)
    elif label == "izquierda":
        # Empujar muñeca izquierda hacia afuera
        delta_x = np.random.uniform(0.15, 0.25) * max(0.15, shoulder_span)
        lw[0] = max(0.0, lw[0] - delta_x)
    elif label == "derecha":
        delta_x = np.random.uniform(0.15, 0.25) * max(0.15, shoulder_span)
        rw[0] = min(1.0, rw[0] + delta_x)
    elif label == "quieto":
        # Pequeñísimo movimiento
        s += np.random.normal(0, 0.003, size=s.shape).astype(np.float32)

    s = np.clip(s, 0.0, 1.0)
    return s

def build_sample(label):
    sk = base_skeleton()
    sk = add_noise(sk)
    sk = apply_label_transform(sk, label)
    # Extra: a veces espejear (simétrico horizontal) para robustez
    if random.random() < 0.3:
        sk[:, 0] = 1.0 - sk[:, 0]
    # Extra: pequeñas rotaciones alrededor del centro
    if random.random() < 0.2:
        ang = np.deg2rad(np.random.uniform(-8, 8))
        c, s = math.cos(ang), math.sin(ang)
        center = np.array([0.5, 0.5], dtype=np.float32)
        pts = sk - center
        rot = np.stack([c*pts[:,0] - s*pts[:,1], s*pts[:,0] + c*pts[:,1]], axis=1)
        sk = np.clip(rot + center, 0.0, 1.0)
    # Seleccionar solo los 17 puntos que usaremos
    sel = sk[POSE_IDXS]  # (17,2)
    feat = sel.reshape(-1).astype(np.float32)  # (34,)
    y = CLASSES.index(label)
    return feat, y

class SynthPoseDataset(Dataset):
    def __init__(self, n_per_class=1000, seed=42):
        rng = np.random.RandomState(seed)
        random.seed(seed)
        X, Y = [], []
        for cls in CLASSES:
            for _ in range(n_per_class):
                f, y = build_sample(cls)
                X.append(f); Y.append(y)
        self.X = np.stack(X, axis=0)
        self.Y = np.array(Y, dtype=np.int64)

        # Mezclar
        idx = rng.permutation(len(self.Y))
        self.X = self.X[idx]; self.Y = self.Y[idx]

        # Normalización por características (guardaremos mean/std)
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-6
        self.Xn = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.Xn[i], self.Y[i]

# =========================
# Entrenamiento
# =========================
def train_model(samples_per_class=1200, epochs=12, batch_size=128, lr=1e-3, outdir="models"):
    os.makedirs(outdir, exist_ok=True)
    ds = SynthPoseDataset(n_per_class=samples_per_class)
    n_train = int(len(ds)*0.9)
    Xtr, Ytr = ds.Xn[:n_train], ds.Y[:n_train]
    Xva, Yva = ds.Xn[n_train:], ds.Y[n_train:]

    tr_loader = DataLoader(list(zip(Xtr, Ytr)), batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(list(zip(Xva, Yva)), batch_size=256, shuffle=False, drop_last=False)

    model = SimpleGestureModel(in_dim=N_FEATURES, hidden=64, num_classes=len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb = xb.float()
            yb = yb.long()
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1)==yb).sum().item()
            tr_total += xb.size(0)

        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                logits = model(xb.float())
                va_correct += (logits.argmax(1)==yb.long()).sum().item()
                va_total += xb.size(0)

        print(f"Epoch {ep:02d} | loss {tr_loss/tr_total:.4f} | acc_tr {tr_correct/tr_total:.3f} | acc_va {va_correct/va_total:.3f}")

    # Guardar modelo y stats
    ckpt_path = os.path.join(outdir, "gesture_model.pt")
    torch.save(model.state_dict(), ckpt_path)

    # Guardar stats de normalización + orden de features + clases
    np.savez(os.path.join(outdir, "feature_stats.npz"),
             mean=ds.mean.astype(np.float32),
             std=ds.std.astype(np.float32),
             pose_idxs=np.array(POSE_IDXS, dtype=np.int32))
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "n_features": N_FEATURES}, f, ensure_ascii=False, indent=2)

    print(f"✅ Modelo guardado en {ckpt_path}")
    print(f"✅ Stats guardadas en models/feature_stats.npz y meta.json")
    return ckpt_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_per_class", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_model(args.samples_per_class, args.epochs, args.batch_size, args.lr)
