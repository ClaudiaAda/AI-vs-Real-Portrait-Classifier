import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------
# Configuración de la página
# --------------------
st.set_page_config(
    page_title="AI vs Real Portrait Classificator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI vs Real Portrait Classificator")
st.write(
    "### 📸 Sube una imagen de un retrato humano y el modelo analizará si es real o generada artificialmente. 📸"
)
st.write(
    "*Este modelo ha sido entrenado con imágenes generadas por IA (FaceSwap, SFHQ, Stable Diffusion, ThisPersonDoesNotExist, StyleGan). Si se adjuntan fotos de otros modelos, el resultado puede ser menos fiable.*"
)

IMG_SIZE = (224, 224)

# --------------------
# Cargar modelo con cache
# --------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/modelo.h5")
        return model
    except Exception as e:
        st.error("❌ Error cargando el modelo")
        st.exception(e)
        return None

model = load_model()

if model is None:
    st.stop()

# --------------------
# Función de preprocesamiento
# --------------------
def preprocess_uploaded_image(uploaded_file):
    # Abrir imagen con PIL y convertir a RGB
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Batch dimension
    return image

# --------------------
# Cargar imagen del usuario
# --------------------
uploaded_file = st.file_uploader("Elige una imagen... 🖼️", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Columna 1: Imagen subida
    with col1:
        st.image(uploaded_file, caption="Imagen subida")

    # Columna 2: Predicción
    with col2:
        st.write("Procesando... ⏳")
        img_tensor = preprocess_uploaded_image(uploaded_file)
        pred = model.predict(img_tensor, verbose=0)[0][0]  # salida binaria
        label = "Artificial" if pred > 0.5 else "Real"

        # Mostrar resultado
        st.success(f"Predicción: **{label}**")
        st.info(f"Confianza: {pred if pred > 0.5 else 1 - pred:.2f}")











