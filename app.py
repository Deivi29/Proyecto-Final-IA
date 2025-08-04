# app.py
import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

from gesture_model import GestureClassifier
from game_logic import GameLogic, GameSimulator

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

classifier = GestureClassifier()
game_logic = GameLogic()
simulador = GameSimulator()

# Función para dibujar todos los landmarks
def dibujar_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(250,44,250), thickness=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
    )

def procesar_video(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(static_image_mode=False,
                               model_complexity=1,
                               enable_segmentation=False,
                               refine_face_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        dibujar_landmarks(image, results)

        # Extraer keypoints del cuerpo
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            keypoints = np.array(keypoints)
            prediccion = classifier.predict(keypoints)
            game_logic.actualizar_estado(prediccion)
        else:
            prediccion = "quieto"

        # Generar simulador visual del juego
        game_frame = simulador.update(game_logic.estado)
        cv2.putText(game_frame, f"Gesto: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(game_frame, f"Estado del Juego: {game_logic.estado}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Redimensionar para unir pantallas
        image = cv2.resize(image, (640, 480))
        game_frame = cv2.resize(game_frame, (640, 480))
        output = np.hstack((image, game_frame))
        return output

iface = gr.Interface(
    fn=procesar_video,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=gr.Image(),
    title="CamJumpAI - Juego por Gestos en Tiempo Real",
    live=True
)

if __name__ == "__main__":
    iface.launch()
