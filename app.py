import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def detectar_pose(image):
    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

# CORREGIDO: Entrada de cámara compatible con Gradio 3.50.2
iface = gr.Interface(
    fn=detectar_pose,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=gr.Image(),
    title="CamJumpAI - Detección de Poses",
    live=True
)

if __name__ == "__main__":
    iface.launch()
