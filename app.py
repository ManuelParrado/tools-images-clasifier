import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Cargar el modelo
model = load_model('images_clasifier.h5')

# Clases del modelo
classes = ['Alicates', 'Cuchillo', 'Cúter', 'Destornillador', 'Martillo', 'Tijeras']

# Título de la app
st.title("Clasificador de Herramientas con IA")
st.markdown("Sube una imagen y el modelo la clasificará como una de las siguientes clases:")
st.write(", ".join(classes))

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen...", type=[".jpg", ".jpeg", ".png", ".JPG"])

if uploaded_file is not None:
    # Mostrar imagen
    st.image(uploaded_file, caption='Imagen subida', use_column_width=True)

    # Preprocesamiento
    img = load_img(uploaded_file, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = classes[predicted_index]

    # Mostrar resultado
    st.subheader(f"Predicción: {predicted_class}")
    st.bar_chart({clase: float(prob) for clase, prob in zip(classes, predictions[0])})
