# train_gesture.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# ======== Config ========
DATA_PATH = "data/gestures.npz"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pt")
EPOCHS = 35
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42
# ========================

torch.manual_seed(SEED)
np.random.seed(SEED)

class SimpleGestureModel(nn.Module):
    def __init__(self, in_dim=34, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_data(path):
    pack = np.load(path, allow_pickle=True)
    X = pack["X"].astype(np.float32)   # (N,34)
    y = pack["y"].astype(np.int64)     # (N,)
    labels = list(pack["labels"])
    return X, y, labels

def main():
    assert os.path.exists(DATA_PATH), f"No existe dataset: {DATA_PATH}. Corre primero collect_gestures.py"
    X, y, labels = load_data(DATA_PATH)
    num_classes = len(labels)
    print(f"[INFO] Dataset: {X.shape[0]} muestras, {num_classes} clases -> {labels}")

    # Mezclar
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]; y = y[idx]

    # Tensores
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    # Split train/val
    N = len(X_t)
    n_val = int(N * VAL_SPLIT)
    n_train = N - n_val
    train_ds, val_ds = random_split(TensorDataset(X_t, y_t), [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    model = SimpleGestureModel(in_dim=34, num_classes=num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0
    os.makedirs(MODELS_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = loss_sum / total
        train_acc  = correct / total

        # Val
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = crit(logits, yb)
                loss_sum += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)

        val_loss = loss_sum / max(1, total)
        val_acc  = correct / max(1, total)

        print(f"[E{epoch:02d}] train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")

        # Guardar mejor
        if val_acc > best_val:
            torch.save(model.state_dict(), MODEL_PATH)
            best_val = val_acc
            print(f"  ↳ ✅ Nuevo mejor modelo guardado en {MODEL_PATH} (val_acc={val_acc:.3f})")

    print(f"[OK] Entrenamiento finalizado. Mejor val_acc={best_val:.3f}.")
    print("Ahora ejecuta tu app y prueba en tiempo real: python app.py")

if __name__ == "__main__":
    main()
