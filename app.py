# ============================================
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052
# Proyecto: CamJumpAI (Modo local y Gradio sin abrir navegador)
# ============================================

import argparse
import time
import threading
import cv2
import numpy as np
import gradio as gr
import mediapipe as mp

from gesture_model import GestureClassifier
from game_logic import GameLogic, GameSimulator

# =========================
# Parámetros
# =========================
FRAME_W, FRAME_H = 640, 480
OUT_W, OUT_H = FRAME_W * 2, FRAME_H           # vista cámara + juego
FONT = cv2.FONT_HERSHEY_SIMPLEX
TARGET_FPS = 15.0                              # limita procesado (anti-parpadeo)
FRAME_DT = 1.0 / TARGET_FPS
CAM_INDEX = 0                                  # cambia a 1 si usas cámara USB

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
    min_tracking_confidence=0.5,
)
holistic_lock = threading.Lock()

# =========================
# IA / Juego
# =========================
classifier = GestureClassifier()   # PyTorch adentro (cumple Examen.md)
logic = GameLogic()
sim = GameSimulator(width=FRAME_W, height=FRAME_H)

# =========================
# Estado anti-parpadeo
# =========================
_last_out_rgb = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
_last_process_t = 0.0
_last_pred = "quieto"


def _dibujar_landmarks(image_bgr, results):
    # Cara (opcional: comentar estas 2 líneas si quieres aún más FPS)
    if results.face_landmarks is not None:
        mp_drawing.draw_landmarks(
            image_bgr, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
        )
        mp_drawing.draw_landmarks(
            image_bgr, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS
        )
    # Manos
    if results.left_hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    # Pose
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )


def procesar_video(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Entrada: frame RGB (numpy uint8, HxWx3), desde webcam del navegador o desde OpenCV.
    Salida:  imagen RGB (uint8) tamaño fijo OUT_H x OUT_W x 3.
    """
    global _last_out_rgb, _last_process_t, _last_pred

    try:
        if frame_rgb is None or frame_rgb.size == 0:
            return _last_out_rgb

        # Throttling de FPS (reutiliza último frame para evitar parpadeo)
        now = time.time()
        if now - _last_process_t < FRAME_DT:
            return _last_out_rgb

        # Asegurar contiguo en memoria
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # 1) MediaPipe (usa RGB)
        with holistic_lock:
            results = holistic.process(frame_rgb)

        # 2) Dibujar landmarks sobre copia BGR
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (FRAME_W, FRAME_H))
        _dibujar_landmarks(bgr, results)

        # 3) Keypoints de pose -> clasificador
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            keypoints = np.asarray(keypoints, dtype=np.float32)
            pred = classifier.predict(keypoints)  # tu modelo PyTorch
            logic.actualizar_estado(pred)
            _last_pred = pred
        else:
            _last_pred = "quieto"
            logic.actualizar_estado(_last_pred)

        # 4) Simulador (frame del juego)
        game_frame = sim.update(logic.estado)  # se asume BGR uint8
        cv2.putText(game_frame, f"Gesto: {_last_pred}", (10, 30), FONT, 1, (0, 255, 0), 2)
        cv2.putText(game_frame, f"Estado: {logic.estado}", (10, 70), FONT, 0.8, (0, 0, 255), 2)
        game_frame = cv2.resize(game_frame, (FRAME_W, FRAME_H))

        # 5) Unir vistas y devolver RGB tamaño constante
        out_bgr = np.hstack((bgr, game_frame))
        _last_out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        _last_process_t = now
        return _last_out_rgb

    except Exception as e:
        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.putText(canvas, f"Error: {str(e)}", (10, 40), FONT, 0.9, (0, 0, 255), 2)
        _last_out_rgb = canvas
        return _last_out_rgb


# =========================
# Lanzadores
# =========================
def run_local():
    """Ventana OpenCV (sin web, sin navegador)."""
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # backend estable en Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Cierra otras apps que la usen e intenta de nuevo.")

    # Calentamiento para evitar flash inicial
    for _ in range(5):
        cap.read()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_rgb = procesar_video(frame_rgb)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("CamJumpAI – Modo LOCAL (OpenCV)", out_bgr)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()


def run_gradio():
    """Interfaz Gradio 3.50.2 (no abre navegador) con streaming estable."""
    with gr.Blocks(title="CamJumpAI – Juego por Gestos en Tiempo Real") as demo:
        gr.Markdown("### Cámara (izquierda)  |  Juego (derecha)")
        with gr.Row():
            with gr.Column():
                cam = gr.Image(source="webcam", type="numpy", streaming=True, label="Cámara")
            with gr.Column():
                out = gr.Image(label="Salida (procesada)")

        # En Gradio 3.50.2 se usa 'every' (no 'stream_every') y sin concurrency_limit.
        cam.stream(procesar_video, inputs=cam, outputs=out, every=0.08)  # ~12.5 FPS; sube/baja 0.05–0.12

    # Cola simple (sin concurrency_count deprecado)
    demo.queue(max_size=1)

    # No abrir navegador automáticamente; GUI en 127.0.0.1:7860
    demo.launch(
        inbrowser=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["local", "gradio"],
        default="local",
        help="local = ventana OpenCV (sin web); gradio = interfaz web local (no abre navegador)"
    )
    args = parser.parse_args()

    if args.mode == "local":
        run_local()
    else:
        run_gradio()
