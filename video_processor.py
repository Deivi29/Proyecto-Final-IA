# video_processor.py
## Nombre Deivi Rodriguez Paulino 
## Matrícula 21-SISN-2-052 

import cv2
import numpy as np
import mediapipe as mp
from gesture_model import GestureClassifier
from game_logic import GameLogic, GameSimulator

# MediaPipe Solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles  # Para colores avanzados

# Modelos
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
face = mp_face.FaceMesh()
classifier = GestureClassifier()
logic = GameLogic()
simulator = GameSimulator()

cap = cv2.VideoCapture(1)

finger_names = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"]

def contar_dedos(hand_landmarks):
    dedos_arriba = []
    puntos = hand_landmarks.landmark
    if puntos[4].x < puntos[3].x:
        dedos_arriba.append("Pulgar")
    for i, idx in enumerate([8, 12, 16, 20]):
        if puntos[idx].y < puntos[idx - 2].y:
            dedos_arriba.append(finger_names[i+1])
    return dedos_arriba

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resultados
    results_pose = pose.process(rgb)
    results_hands = hands.process(rgb)
    results_face = face.process(rgb)

    # Pose
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
        keypoints = np.array([[lm.x, lm.y] for lm in results_pose.pose_landmarks.landmark]).flatten()
        gesture = classifier.predict(keypoints) if len(keypoints) >= 34 else "quieto"
        logic.actualizar_estado(gesture)
    else:
        gesture = "quieto"

    # Manos
    dedos_levantados = []
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            dedos_levantados = contar_dedos(hand_landmarks)

    # Rostro (FaceMesh)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )

    # Juego
    sim_frame = simulator.update(logic.estado)

    # Textos
    texto_gesto = f"Gesto IA: {gesture}"
    texto_dedos = "Dedos: " + ", ".join(dedos_levantados) if dedos_levantados else "Sin dedos levantados"
    estado = f"Estado del juego: {logic.estado}"

    cv2.putText(frame, texto_gesto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, texto_dedos, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
    cv2.putText(frame, estado, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar combinados
    combined = np.hstack((frame, sim_frame))
    cv2.imshow("CamJumpAI - Vista Completa", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
