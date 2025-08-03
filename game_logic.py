import numpy as np
import cv2

class GameSimulator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.player_x = width // 2
        self.player_y = height - 50
        self.jump_height = 100
        self.jump = False
        self.jump_frame = 0

    def update(self, gesture):
        if gesture == "izquierda":
            self.player_x -= 20
        elif gesture == "derecha":
            self.player_x += 20
        elif gesture == "salto" and not self.jump:
            self.jump = True
            self.jump_frame = 15  # number of frames to jump

        # Simular salto
        if self.jump:
            self.player_y -= self.jump_height // 15
            self.jump_frame -= 1
            if self.jump_frame <= 0:
                self.jump = False
        else:
            self.player_y = self.height - 50

        # Limitar dentro del marco
        self.player_x = max(0, min(self.width, self.player_x))

        return self.render()

    def render(self):
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        cv2.circle(frame, (self.player_x, self.player_y), 30, (0, 0, 255), -1)
        return frame
# game_logic.py

class GameLogic:
    def __init__(self):
        self.estado = "quieto"

    def actualizar_estado(self, gesto):
        if gesto in ["salto", "izquierda", "derecha"]:
            self.estado = gesto
        else:
            self.estado = "quieto"
