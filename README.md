#  CamJumpAI – Juego Interactivo por Cámara usando IA

**Nombre:** Deivi Rodriguez Paulino  
**Matrícula:** 21-SISN-2-052

CamJumpAI es un juego controlado por **gestos** capturados por la cámara. Usa **MediaPipe** para extraer puntos de pose en tiempo real y un modelo de **Deep Learning (PyTorch)** para clasificar gestos que mueven al jugador (izquierda, derecha, salto). Incluye interfaz gráfica con **Gradio** (obligatoria según `Examen.md`) y una versión alternativa en ventana local con **OpenCV**.

---

## 🧠 Arquitectura (pipeline)
Webcam → MediaPipe (Pose) → Keypoints (34: 17×[x,y]) → **PyTorch (MLP)** → Gesto → **Lógica de juego** → Render (OpenCV/Gradio)

---

## ✨ Gestos soportados
- 🙌 **Ambas manos arriba** → `salto`
- ✋ **Mano izquierda arriba** → `izquierda`
- ✋ **Mano derecha arriba** → `derecha`
- 🙅 **Sin manos arriba** → `quieto`

> Nota: El modelo puede entrenarse con tus propias muestras para mayor precisión.

---

## Estructura del proyecto
Proyecto-Final-IA/
├─ app.py # Interfaz gráfica (Gradio)
├─ video_processor.py # Ejecución local en ventana (OpenCV)
├─ gesture_model.py # Carga/uso del modelo PyTorch
├─ game_logic.py # Lógica del juego, física y render
├─ collect_gestures.py # (Nuevo) Recolección de dataset con la webcam
├─ train_gesture.py # (Nuevo) Entrenamiento del modelo (PyTorch)
├─ create_dummy_model.py # Modelo dummy (por si no se entrena)
├─ models/
│ └─ gesture_model.pt # Pesos del modelo entrenado
├─ data/
│ └─ gestures.npz # Dataset recolectado (X: (N,34), y: etiquetas)
├─ requirements.txt # Dependencias
├─ README.md # Este archivo
└─ .gitignore # Archivos a excluir

---

## Requisitos
- **Python 3.10** (recomendado)
- Cámara web habilitada
- Paquetes (versiones sugeridas y probadas en Win+Py3.10):

numpy==1.26.4
opencv-python==4.8.1.78
mediapipe==0.10.9
gradio==3.50.2
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
pillow==10.4.0


---

## Instalación
```bash
git clone https://github.com/<tu-usuario>/<tu-repo-o-fork>.git
cd Proyecto-Final-IA

# Crear entorno
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

## Ejecución

### Modo local (OpenCV)
```bash
(venv) python app.py --mode local

## Modo Gradio (interfaz web local)
```bash
(venv) python app.py --mode gradio

## Demo
- Video: [video_explicativo.mp4](./video_explicativo.mp4) 
