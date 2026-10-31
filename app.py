
import os
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================
# CONFIGURACIÓN DEL MODELO
# ============================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelo_pterygium_recuperado.keras"
)

IMAGE_SIZE = (224, 224)

# Clase 0 = Sin pterigión
# Clases 1-4 = Con pterigión

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo: {MODEL_PATH}. "
        "Coloca el archivo .keras en la raíz del repositorio."
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

if tuple(model.input_shape[1:]) != (224, 224, 3):
    raise ValueError(
        f"Entrada inesperada del modelo: {model.input_shape}"
    )

if model.output_shape[-1] != 5:
    raise ValueError(
        f"El modelo debe tener 5 salidas; tiene: {model.output_shape}"
    )


# ============================================
# DETECCIÓN DE PRESENCIA DE PTERIGIÓN
# ============================================

def predict_eye(image: Image.Image):

    if image is None:
        return "Por favor, sube una imagen del ojo."

    try:
        # Procesamiento de la imagen
        image = image.convert("RGB").resize(IMAGE_SIZE)

        array = np.asarray(image, dtype=np.float32)
        array = preprocess_input(array)
        array = np.expand_dims(array, axis=0)

        # Predicción del modelo
        prediccion = model.predict(array, verbose=0)[0]
        prediccion = np.asarray(prediccion, dtype=float)

        if prediccion.shape != (5,):
            raise ValueError(
                f"Salida inesperada: {prediccion.shape}"
            )

        # Clase 0 = Ausencia
        # Clases 1-4 = Presencia
        probabilidad_sin = float(prediccion[0])
        probabilidad_con = float(np.sum(prediccion[1:]))

        # Determinar presencia o ausencia
        if np.argmax(prediccion) == 0:
            resultado = "PTERIGIÓN NO DETECTADO"
        else:
            resultado = "PTERIGIÓN DETECTADO"

        # Mostrar resultado y probabilidades
        return (
            f"{resultado}\n\n"
            f"Probabilidad estimada de presencia: "
            f"{probabilidad_con * 100:.2f}%\n"
            f"Probabilidad estimada de ausencia: "
            f"{probabilidad_sin * 100:.2f}%"
        )

    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"


# ============================================
# INTERFAZ DE GRADIO
# ============================================

demo = gr.Interface(
    fn=predict_eye,

    inputs=gr.Image(
        type="pil",
        label="Sube una fotografía del ojo"
    ),

    outputs=gr.Textbox(
        label="Resultado de detección",
        lines=4
    ),

    title="Detección de Pterigión",

    description=(
        "Carga una fotografía del ojo para obtener una "
        "estimación de presencia o ausencia de pterigión."
    ),

    theme="soft"
)


# ============================================
# INICIAR APLICACIÓN EN RENDER
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
