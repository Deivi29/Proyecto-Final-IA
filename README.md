# Proyecto-de-Final-IA

## Nombre Deivi Rodriguez Paulino 

## Matrícula 21-SISN-2-052 

## Proyecto Final 

# 🎮 CamJumpAI - Juego Interactivo por Cámara usando IA

**CamJumpAI** es un juego controlado por gestos corporales capturados por la cámara, utilizando detección de poses en tiempo real con **MediaPipe** y un modelo de *deep learning* implementado en **PyTorch**. El jugador puede mover un punto rojo (el personaje) a la izquierda, derecha o hacerlo saltar mediante gestos con las manos.

---

## 📌 Descripción

Este proyecto fue desarrollado como parte del examen final de la asignatura **Inteligencia Artificial**, y consiste en una aplicación interactiva que:

- Utiliza la cámara para detectar movimientos humanos en tiempo real.
- Emplea un modelo de red neuronal simple en **PyTorch** para clasificar los gestos corporales.
- Simula un entorno de juego minimalista donde los gestos detectados controlan al jugador.
- Presenta dos modos de ejecución: desde interfaz gráfica (con **Gradio**) o directamente en ventana (con **OpenCV**).

---

## 🧠 Tecnologías Usadas

- **Python 3.10+**
- **PyTorch** – para el modelo de predicción de gestos
- **MediaPipe** – para detección de poses en tiempo real
- **OpenCV** – para el procesamiento visual y simulación del juego
- **Gradio** – para la interfaz gráfica interactiva

---

## 🎮 ¿Cómo Funciona el Juego?

El sistema detecta los siguientes gestos a partir de los puntos de referencia del cuerpo:

- 🙌 Ambas manos arriba → **salto**
- ✋ Mano izquierda arriba → **mover a la izquierda**
- ✋ Mano derecha arriba → **mover a la derecha**
- 🙅 Ninguna mano levantada → **quieto**

Estos gestos se interpretan por un modelo de clasificación basado en PyTorch y se aplican en un entorno gráfico donde un punto rojo simula el jugador.

---

## 🧪 Estructura del Proyecto

Proyecto-Final-IA/
├── app.py # Interfaz gráfica con Gradio
├── video_processor.py # Versión con ventana local (sin navegador)
├── gesture_model.py # Modelo y clasificador de gestos con PyTorch
├── game_logic.py # Lógica del juego y simulación
├── create_dummy_model.py # Script para generar el modelo dummy
├── models/
│ └── gesture_model.pt # Modelo de deep learning guardado
├── requirements.txt # Librerías necesarias
├── README.md # Este archivo
└── .gitignore # Exclusión de archivos innecesarios

---

## ⚙️ Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/CamJumpAI.git
cd CamJumpAI
