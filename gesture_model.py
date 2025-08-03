# gesture_model.py
import numpy as np

class GestureClassifier:
    def __init__(self):
        pass  # Aquí puedes cargar un modelo real con PyTorch si decides entrenarlo

    def predict(self, keypoints: np.ndarray) -> str:
        if len(keypoints) < 34:  # 17 landmarks mínimo
            return "quieto"

        cabeza_y = keypoints[1]
        mano_izq_y = keypoints[15 * 2 + 1]
        mano_der_y = keypoints[16 * 2 + 1]

        if mano_izq_y < cabeza_y and mano_der_y < cabeza_y:
            return "salto"
        elif mano_izq_y < cabeza_y:
            return "izquierda"
        elif mano_der_y < cabeza_y:
            return "derecha"
        else:
            return "quieto"
