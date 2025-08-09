# game_logic.py
# Nombre: Deivi Rodriguez Paulino
# Matrícula: 21-SISN-2-052

import numpy as np
import cv2
import random
import math
from math import sqrt

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _circle_rect_collide(cx, cy, cr, rx, ry, rw, rh):
    """Colisión círculo-rectángulo (jugador vs obstáculo)."""
    closest_x = _clamp(cx, rx, rx + rw)
    closest_y = _clamp(cy, ry, ry + rh)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) <= (cr * cr)

class GameSimulator:
    """
    Mini–plataforma: mueve el círculo rojo con gestos:
      - 'izquierda' -> mover a la izquierda
      - 'derecha'   -> mover a la derecha
      - 'salto'     -> saltar si está en el suelo
    Objetivo: tocar la moneda verde. Evita el obstáculo.
    """
    def __init__(self, width=640, height=480):
        self.width  = width
        self.height = height

        # Mundo
        self.ground_y     = height - 40
        self.bg_color     = (255, 255, 255)

        # Jugador
        self.r_player     = 20
        self.player_x     = width // 2
        self.player_y     = self.ground_y
        self.gravity      = 2.4
        self.jump_vel     = -26.0         # potencia de salto
        self.move_speed   = 14
        self.vy           = 0.0
        self.on_ground    = True

        # Objetivo (moneda)
        self.r_goal       = 14
        self.goal_x       = 0
        self.goal_y       = 0
        self._spawn_goal()                 # respeta altura alcanzable

        # Obstáculo
        self.obs_w        = 70
        self.obs_h        = 22
        self.obs_x        = random.randint(0, self.width - self.obs_w)
        self.obs_y        = self.ground_y - self.obs_h
        self.obs_vx       = 6

        # Marcadores
        self.score        = 0
        self.level        = 1
        self.status_msg   = "¡Listo!"

    def update(self, gesture: str):
        # Entrada por gestos
        if gesture == "izquierda":
            self.player_x -= self.move_speed
        elif gesture == "derecha":
            self.player_x += self.move_speed
        elif gesture == "salto" and self.on_ground:
            self.vy = self.jump_vel
            self.on_ground = False
            self.status_msg = "¡Salto!"

        # Gravedad
        self.vy += self.gravity
        self.player_y += int(self.vy)

        # Suelo / límites
        if self.player_y >= self.ground_y:
            self.player_y = self.ground_y
            self.vy = 0
            self.on_ground = True

        self.player_x = _clamp(self.player_x, self.r_player, self.width - self.r_player)

        # Obstáculo
        self.obs_x += self.obs_vx
        if self.obs_x <= 0 or self.obs_x + self.obs_w >= self.width:
            self.obs_vx *= -1

        # Colisiones
        if self._hit_goal():
            self.score += 1
            self.status_msg = "¡Objetivo!"
            if self.score % 3 == 0:
                self.level += 1
                self.obs_vx += 1 if self.obs_vx >= 0 else -1
            self._spawn_goal()

        if _circle_rect_collide(self.player_x, self.player_y, self.r_player,
                                self.obs_x, self.obs_y, self.obs_w, self.obs_h):
            self.score = max(0, self.score - 1)
            self.status_msg = "¡Ouch!"
            self.player_x -= 15 if self.obs_vx > 0 else -15
            self.player_y = max(self.player_y - 10, self.r_player)

        return self.render()

    def render(self):
        frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        # suelo
        cv2.line(frame, (0, self.ground_y + 1), (self.width, self.ground_y + 1), (180, 180, 180), 2)
        # objetivo
        cv2.circle(frame, (int(self.goal_x), int(self.goal_y)), self.r_goal, (40, 180, 40), -1)
        cv2.circle(frame, (int(self.goal_x), int(self.goal_y)), self.r_goal, (0, 100, 0), 2)
        # obstáculo
        cv2.rectangle(frame, (int(self.obs_x), int(self.obs_y)),
                      (int(self.obs_x + self.obs_w), int(self.obs_y + self.obs_h)), (0, 0, 0), -1)
        # jugador
        cv2.circle(frame, (int(self.player_x), int(self.player_y)), self.r_player, (0, 0, 255), -1)

        cv2.putText(frame, f"Score: {self.score}   Nivel: {self.level}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
        cv2.putText(frame, self.status_msg, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        cv2.putText(frame, "Objetivo: toca la moneda verde. Evita el bloque negro.",
                    (10, self.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        return frame

    def _max_jump_height(self) -> int:
        """Altura máxima alcanzable (para spawnear la moneda en rango realista)."""
        n = max(1, int(math.ceil(-self.jump_vel / self.gravity)))
        dy = n * self.jump_vel + 0.5 * self.gravity * (n - 1) * n
        return int(abs(dy))

    def _spawn_goal(self):
        margin_x = 30
        self.goal_x = random.randint(margin_x, self.width - margin_x)

        max_h = self._max_jump_height()
        max_y = self.ground_y - self.r_goal - 6
        min_y = self.ground_y - (max_h - 10)

        min_y = max(self.height // 2, min_y)
        if min_y >= max_y:
            min_y = max_y - 10

        self.goal_y = random.randint(int(min_y), int(max_y))

    def _hit_goal(self):
        dx = self.player_x - self.goal_x
        dy = self.player_y - self.goal_y
        dist = sqrt(dx * dx + dy * dy)
        return dist <= (self.r_player + self.r_goal)


class GameLogic:
    """Traductor de gesto -> estado para el simulador (con valor por defecto)."""
    def __init__(self):
        self.estado = "quieto"

    def actualizar_estado(self, gesto: str):
        if gesto in ("salto", "izquierda", "derecha"):
            self.estado = gesto
        else:
            self.estado = "quieto"
