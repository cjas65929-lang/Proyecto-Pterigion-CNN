# Aplicación de clasificación de pterigión

## Archivos necesarios
- `app.py`
- `requirements.txt`
- `modelo_pterygium_recuperado.keras`

Coloca el modelo `.keras` en la raíz del repositorio, junto a `app.py`.

## Render
- Runtime/Language: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`

La aplicación utiliza el preprocesamiento MobileNetV2 (`preprocess_input`), imágenes RGB de 224x224 y una salida softmax de cinco clases, según el notebook proporcionado.

**Antes de publicar:** confirma que el orden de las cinco carpetas/clases del dataset corresponde exactamente a Grado 0, 1, 2, 3 y 4. El notebook no muestra la salida de `train_generator.class_indices`, por lo que ese orden no se puede verificar solo con el archivo compartido.
