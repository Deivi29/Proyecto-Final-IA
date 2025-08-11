# video_processor.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from gesture_model import GestureClassifier
from game_logic import GameLogic, GameSimulator

# ---------------- Parámetros ----------------
FRAME_W, FRAME_H = 640, 480
OUT_W, OUT_H = FRAME_W * 2, FRAME_H
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Procesa pesado (MediaPipe) solo ~12.5 FPS; la cámara se muestra SIEMPRE
TARGET_FPS_PROCESS = 12.5
PROCESS_DT = 1.0 / TARGET_FPS_PROCESS
CAM_INDEX = 0  # <--- cámbialo a 1 si tu webcam USB es 1

# ------------- MediaPipe Solutions ----------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(model_complexity=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
face = mp_face.FaceMesh(refine_landmarks=True)  # si va muy pesado, pon refine_landmarks=False
classifier = GestureClassifier()
logic = GameLogic()
simulator = GameSimulator(width=FRAME_W, height=FRAME_H)

holistic_lock = threading.Lock()

# ---------------- Utilidades ----------------
finger_names = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"]

def contar_dedos(hand_landmarks):
    dedos_arriba = []
    puntos = hand_landmarks.landmark
    # Pulgar (aprox): x4 < x3 en mano derecha (ajústalo si usas la izquierda)
    if puntos[4].x < puntos[3].x:
        dedos_arriba.append("Pulgar")
    # Índice, Medio, Anular, Meñique
    for i, idx in enumerate([8, 12, 16, 20]):
        if puntos[idx].y < puntos[idx - 2].y:
            dedos_arriba.append(finger_names[i+1])
    return dedos_arriba

# ------------- Captura de cámara ------------
# Usa DirectShow en Windows y fija propiedades
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
# Reduce “colas” de frames (no siempre lo respetan los drivers)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara. Cierra otras apps y verifica CAM_INDEX.")

# Calentamiento para evitar flash inicial
for _ in range(5):
    cap.read()

# Estado para ritmo de procesado
last_processed_time = 0.0
last_pred = "quieto"
last_vis_bgr = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)  # última cámara con landmarks

def dibuja_face_hands_pose(frame_bgr, results_pose, results_hands, results_face):
    # Pose
    if results_pose and results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
    # Manos
    if results_hands and results_hands.multi_hand_landmarks:
        for hlm in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                hlm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
    # Rostro
    if results_face and results_face.multi_face_landmarks:
        for flm in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, flm, mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                frame_bgr, flm, mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- Siempre cámara fresca ---
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    now = time.time()

    # ¿Toca procesar pesado?
    do_process = (now - last_processed_time) >= PROCESS_DT

    if do_process:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with holistic_lock:
            results_pose = pose.process(rgb)
            results_hands = hands.process(rgb)
            results_face = face.process(rgb)

        # Dibujo SOBRE COPIA para no congelar la cámara base
        vis_bgr = frame.copy()
        dibuja_face_hands_pose(vis_bgr, results_pose, results_hands, results_face)

        # Gesture (usa solo pose)
        if results_pose and results_pose.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y] for lm in results_pose.pose_landmarks.landmark], dtype=np.float32).flatten()
            last_pred = classifier.predict(keypoints) if keypoints.size >= 34 else "quieto"
            logic.actualizar_estado(last_pred)
        else:
            last_pred = "quieto"
            logic.actualizar_estado(last_pred)

        last_vis_bgr = vis_bgr  # guarda la última cámara con landmarks
        last_processed_time = now
        left_panel = last_vis_bgr
    else:
        # Sin procesar esta vuelta: muestra cámara cruda (viva)
        left_panel = frame

    # Juego SIEMPRE se actualiza
    sim_frame = simulator.update(logic.estado)
    sim_frame = cv2.resize(sim_frame, (FRAME_W, FRAME_H))

    # Textos
    cv2.putText(left_panel, f"Gesto IA: {last_pred}", (10, 30), FONT, 0.7, (0, 255, 0), 2)
    # Opcional: mostrar dedos levantados solo cuando hubo manos procesadas
    # (si quieres, puedes almacenar últimos dedos detectados igual que last_pred)

    cv2.putText(sim_frame, f"Estado: {logic.estado}", (10, 30), FONT, 0.8, (255, 0, 0), 2)

    # Unir y mostrar
    combined = np.hstack((left_panel, sim_frame))
    cv2.imshow("CamJumpAI - Vista Completa (fluido)", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
