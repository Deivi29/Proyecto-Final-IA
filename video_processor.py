import cv2
import numpy as np
import mediapipe as mp
from gesture_model import GestureClassifier
from game_logic import GameLogic, GameSimulator

# Inicializar MediaPipe, modelo y lógica del juego
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
classifier = GestureClassifier()
logic = GameLogic()
simulator = GameSimulator()

# Iniciar captura de cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip para espejo
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    gesture = "No detectado"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
        keypoints = np.array(keypoints)
        gesture = classifier.predict(keypoints)
        logic.actualizar_estado(gesture)

    # Dibujar gesto en pantalla
    cv2.putText(frame, f"Gesto: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Estado del juego: {logic.estado}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Simular juego con gesto
    output = simulator.update(logic.estado)

    # Combinar cámara y simulador
    combined = np.hstack((frame, output))
    cv2.imshow("CamJumpAI - Juego Interactivo", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
