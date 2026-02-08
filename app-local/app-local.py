import streamlit as st
import tensorflow as tf
import numpy as np

# --------------------
# Cargar modelo
# --------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\Claudia\CursoIA2\PROYECTO\Real-vs-AI\Modelo_DEFINITIVO.h5")
    return model

model = load_model()

IMG_SIZE = (224,224)

# --------------------
# Función de predición
# --------------------
def preprocess_uploaded_image(uploaded_file):
    # Convertir a tensor
    image = tf.image.decode_jpeg(uploaded_file.read(), channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    # Agregar batch dimension
    image = tf.expand_dims(image, axis=0)
    return image

# --------------------
# Configuracion pagina
# --------------------
st.set_page_config(
    page_title = "AI vs Real Portrait Classificator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI vs Real Portrait Classificator")
st.write("### 📸 Sube una imagen de un retrato humano y el modelo analizará si es real o generada artificialmente. 📸")
st.write("*Este modelo ha sido entrenado con imágenes generadas por las herramientas de IA generativa: FaceSwap, SFHQ (StyleGAN Face HQ), Stable Diffusion, ThisPersonDoesNotExist, StyleGan. Si se adjuntan fotos de otros modelos, el resultado puede ser erróneo.*")

# Cargar imagen del usuario
uploaded_file = st.file_uploader("Elige una imagen... 🖼️", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Dividimos en 2 columnas
    col1, col2 = st.columns(2)

    # Col 1: Mostrar imagen subida
    with col1:
        st.image(uploaded_file, caption="Imagen subida",output_format="auto")

    # Col 2: Preprocesar y predecir
    with col2:
        st.write("Procesando... ⏳")
        img_tensor = preprocess_uploaded_image(uploaded_file)
        pred = model.predict(img_tensor)[0][0]  # salida binaria
        label = "Artificial" if pred > 0.5 else "Real"

        # Mostrar resultado
        st.success(f"Predicción: **{label}**")
        st.info(f"Confianza: {pred if pred > 0.5 else 1 - pred:.2f}")
