# app.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import cv2
import numpy as np
import gradio as gr
import threading

import mediapipe as mp
from gesture_model import GestureClassifier
from game_logic import GameLogic, GameSimulator

# =========================
# Parámetros
# =========================
FRAME_W, FRAME_H = 640, 480
FONT = cv2.FONT_HERSHEY_SIMPLEX

# =========================
# MediaPipe (una sola vez)
# =========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lock para evitar concurrencia en holistic.process
holistic_lock = threading.Lock()

# =========================
# IA / Juego
# =========================
classifier = GestureClassifier()
game_logic = GameLogic()
simulador = GameSimulator(width=FRAME_W, height=FRAME_H)

def dibujar_landmarks(image_bgr, results):
    # Rostro
    mp_drawing.draw_landmarks(
        image_bgr, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
    )
    # Mano izquierda
    mp_drawing.draw_landmarks(
        image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(250,44,250), thickness=2)
    )
    # Mano derecha
    mp_drawing.draw_landmarks(
        image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
    )
    # Pose
    mp_drawing.draw_landmarks(
        image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
    )

def procesar_video(frame_rgb):
    """
    Gradio 3.50 entrega imagen RGB (uint8) cuando usamos Image(source="webcam", streaming=True).
    Devolvemos también RGB (uint8).
    """
    try:
        if frame_rgb is None:
            return np.zeros((FRAME_H, FRAME_W*2, 3), dtype=np.uint8)

        # Garantizar memoria contigua (evita glitches)
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # MediaPipe espera RGB; protegemos el acceso con lock
        with holistic_lock:
            results = holistic.process(frame_rgb)

        # Para dibujar con OpenCV, pasamos a BGR
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        dibujar_landmarks(bgr, results)

        # Keypoints de pose para el clasificador
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            keypoints = np.array(keypoints, dtype=np.float32)
            prediccion = classifier.predict(keypoints)
            game_logic.actualizar_estado(prediccion)
        else:
            prediccion = "quieto"
            game_logic.actualizar_estado(prediccion)

        # Frame del juego
        game_frame = simulador.update(game_logic.estado)
        cv2.putText(game_frame, f"Gesto: {prediccion}", (10, 30), FONT, 1, (0, 255, 0), 2)
        cv2.putText(game_frame, f"Estado del Juego: {game_logic.estado}", (10, 70), FONT, 0.8, (0, 0, 255), 2)

        # Unir vistas (asegurando tamaños)
        bgr = cv2.resize(bgr, (FRAME_W, FRAME_H))
        game_frame = cv2.resize(game_frame, (FRAME_W, FRAME_H))
        out_bgr = np.hstack((bgr, game_frame))

        # Devolver RGB a Gradio
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return out_rgb

    except Exception as e:
        # No dejamos el spinner; devolvemos un frame con el error
        canvas = np.zeros((FRAME_H, FRAME_W*2, 3), dtype=np.uint8)
        cv2.putText(canvas, f"Error: {str(e)}", (10, 40), FONT, 0.9, (0, 0, 255), 2)
        return canvas

# =========================
# Interfaz Gradio estable
# =========================
demo = gr.Interface(
    fn=procesar_video,
    inputs=gr.Image(source="webcam", streaming=True),   # webcam en vivo
    outputs=gr.Image(type="numpy"),                     # devolvemos numpy RGB
    title="CamJumpAI - Juego por Gestos en Tiempo Real",
    live=True,
    allow_flagging="never",
    analytics_enabled=False
).queue(max_size=1, concurrency_count=1)  # evita concurrencia

if __name__ == "__main__":
    # Si te molesta abrir navegador automáticamente:
    # demo.launch(inline=False, inbrowser=False)
    demo.launch()
