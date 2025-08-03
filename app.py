import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
from gesture_model import GestureClassifier

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
classifier = GestureClassifier()

def detectar_y_predecir(image):
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        prediction = "No se detecta persona"
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            keypoints = np.array(keypoints)
            prediction = classifier.predict(keypoints)

        cv2.putText(image, f"Gesto: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image
from game_logic import GameLogic

# Inicializar la lógica del juego
game = GameLogic()

def detectar_y_jugar(image):
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        prediccion = "No se detecta persona"
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            keypoints = np.array(keypoints)
            prediccion = classifier.predict(keypoints)
            game.actualizar_estado(prediccion)  # actualizar lógica del juego

        cv2.putText(image, f"Gesto: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Estado del Juego: {game.estado}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return image

iface = gr.Interface(
    fn=detectar_y_predecir,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=gr.Image(),
    title="CamJumpAI - Detección de Gestos con IA",
    live=True
)

if __name__ == "__main__":
    iface.launch()
