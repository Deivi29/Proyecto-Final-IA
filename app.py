# app.py
import gradio as gr
import subprocess
import threading

# Función que lanza el script del juego
def lanzar_juego():
    def ejecutar():
        subprocess.run(["python", "video_processor.py"])
    hilo = threading.Thread(target=ejecutar)
    hilo.start()
    return "🎮 Juego lanzado en ventana nativa.\n\nCierra esa ventana para volver a ejecutar."

# Interfaz con botón Gradio
iface = gr.Interface(
    fn=lanzar_juego,
    inputs=[],
    outputs="text",
    title="CamJumpAI - Juego por Gestos en Tiempo Real",
    description="Presiona el botón para iniciar el juego. Se abrirá una ventana del juego controlado por tus gestos detectados por IA.",
    live=False
)

if __name__ == "__main__":
    iface.launch()
