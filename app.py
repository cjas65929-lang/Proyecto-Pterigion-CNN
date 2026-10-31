import os
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Modelo reconstruido y guardado desde tu notebook.
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_pterygium_recuperado.keras")
IMAGE_SIZE = (224, 224)

# El notebook usa 5 salidas softmax y las interpreta como grados 0 a 4.
# Verifica que el orden coincida con las carpetas usadas al entrenar.
CLASS_NAMES = [
    "Grado 0 - Sin pterigión / normal",
    "Grado 1 - Pterigión grado 1",
    "Grado 2 - Pterigión grado 2",
    "Grado 3 - Pterigión grado 3",
    "Grado 4 - Pterigión grado 4",
]

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo: {MODEL_PATH}. "
        "Coloca el archivo .keras en la raíz del repositorio."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

if tuple(model.input_shape[1:]) != (224, 224, 3):
    raise ValueError(f"Entrada inesperada del modelo: {model.input_shape}")
if model.output_shape[-1] != 5:
    raise ValueError(f"El modelo debe tener 5 salidas; tiene: {model.output_shape}")

def predict_eye(image: Image.Image):
    if image is None:
        return {"Sube una imagen": 1.0}

    # Mismo flujo del notebook: RGB, 224x224 y preprocess_input de MobileNetV2.
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.shape != (5,):
        raise ValueError(f"Salida inesperada: {probabilities.shape}")

    # Etiquetas y porcentajes para Gradio.
    results = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(5)
    }
    return results

demo = gr.Interface(
    fn=predict_eye,
    inputs=gr.Image(type="pil", label="Sube una fotografía del ojo"),
    outputs=gr.Label(num_top_classes=5, label="Clasificación estimada"),
    title="Detección y clasificación de pterigión",
    description=(
        "Modelo experimental basado en MobileNetV2. "
        "Clasifica la imagen en grados 0 a 4. "
        "El resultado no constituye un diagnóstico médico ni sustituye "
        "la evaluación de un profesional de oftalmología."
    ),
    theme="soft"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
