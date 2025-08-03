import cv2
import numpy as np
import mediapipe as mp
from gesture_model import GestureClassifier
from game_logic import GameSimulator

mp_pose = mp.solutions.pose

class VideoProcessor:
    def __init__(self):
        self.classifier = GestureClassifier()  # Modelo que devuelve "salto", etc.
        self.simulator = GameSimulator()
        self.pose = mp_pose.Pose(static_image_mode=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def procesar_frame(self, frame):
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            # Extraer coordenadas normalizadas
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)
            keypoints = np.array(keypoints)

            # Clasificar gesto
            gesture = self.classifier.predict(keypoints)
        else:
            gesture = "quieto"

        # Actualizar juego
        game_frame = self.simulator.update(gesture)

        return game_frame
