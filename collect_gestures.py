# collect_gestures.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict

# ========= Config =========
OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "gestures.npz")
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# Clases (MISMA ORDEN que tu GestureClassifier)
LABELS = ["salto", "izquierda", "derecha", "quieto"]

# Teclas -> etiqueta
KEY_TO_LABEL = {
    ord('w'): "salto",
    ord('a'): "izquierda",
    ord('d'): "derecha",
    ord('q'): "quieto",
}

SAMPLES_PER_BURST = 8   # cuántos frames se guardan por ráfaga
BURST_DELAY = 0.0       # pausa entre muestras de una ráfaga (0 = cada frame)
# =========================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract34(landmarks) -> np.ndarray:
    """Toma los primeros 17 landmarks (0..16) y retorna [x0,y0,...,x16,y16] (34 floats)."""
    arr = []
    for i, lm in enumerate(landmarks):
        if i >= 17:
            break
        arr.extend([lm.x, lm.y])
    if len(arr) < 34:
        arr += [0.0] * (34 - len(arr))
    return np.asarray(arr, dtype=np.float32)

def draw_hud(frame, current_label, counters):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    txt = "Teclas: [W]=salto  [A]=izquierda  [D]=derecha  [Q]=quieto  [ESPACIO]=GRABAR  [ESC]=salir"
    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Etiqueta actual: {current_label or '---'}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    y0 = 100
    for i, lbl in enumerate(LABELS):
        cv2.putText(frame, f"{lbl}: {counters[lbl]} muestras", (10, y0 + i*24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0,255,0) if lbl == current_label else (255,255,255), 2)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Cierra otras apps que la usen y vuelve a intentar.")

    X, y = [], []
    counters = defaultdict(int)

    current_label = None
    recording = False
    last_time = 0.0

    print("[INFO] W/A/D/Q eligen etiqueta. ESPACIO graba una ráfaga. ESC para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        draw_hud(frame, current_label, counters)
        cv2.imshow("Recolector de gestos - CamJumpAI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in KEY_TO_LABEL:
            current_label = KEY_TO_LABEL[key]
            print(f"[INFO] Etiqueta actual: {current_label}")
        elif key == 32:  # ESPACIO
            if current_label is None:
                print("[WARN] Selecciona primero una etiqueta con W/A/D/Q.")
            else:
                recording = True
                print(f"[REC] Grabando {SAMPLES_PER_BURST} muestras para '{current_label}'...")
        elif key == 27:  # ESC
            break

        if recording and results.pose_landmarks:
            now = time.time()
            if now - last_time >= BURST_DELAY:
                feats = extract34(results.pose_landmarks.landmark)
                X.append(feats)
                y.append(LABELS.index(current_label))
                counters[current_label] += 1
                last_time = now
                if counters[current_label] % SAMPLES_PER_BURST == 0:
                    recording = False
                    print(f"[REC] Ráfaga completada para '{current_label}'. Total {counters[current_label]}")

    cap.release()
    cv2.destroyAllWindows()

    if not X:
        print("[WARN] No se recolectaron datos. Saliendo sin guardar.")
        return

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    np.savez_compressed(OUT_PATH, X=X, y=y, labels=np.array(LABELS))
    print(f"[OK] Dataset guardado en: {OUT_PATH}")
    print({lbl: counters[lbl] for lbl in LABELS})

if __name__ == "__main__":
    main()
