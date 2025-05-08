import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io

# Carga el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return load_model("images_clasifier.h5")

model = cargar_modelo()

# Define las clases según tu modelo (ajústalas si son diferentes)
class_names = ['alicate', 'cuchillo', 'cúter', 'destornillador', 'martillo', 'tijeras']

# Título de la app
st.title("🔧 Clasificador de Herramientas")

# Sube una imagen
uploaded_file = st.file_uploader("Sube una imagen (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# Procesar y predecir
if uploaded_file is not None:
    try:
        # Muestra la imagen subida
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen subida", use_column_width=True)

        # Preprocesa la imagen
        image = image.resize((150, 150))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        # Resultado
        st.success(f"🔍 Predicción: **{class_names[predicted_index]}**")
        st.info(f"📈 Confianza: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")
else:
    st.warning("Por favor, sube una imagen para clasificar.")
