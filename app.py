import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io

# Cargar el modelo previamente entrenado
model = load_model('images_clasifier.h5')

# Definir las clases (estos deben coincidir con las clases del modelo)
class_names = ['Alicates', 'Cuchillo', 'Cúter', 'Destornillador', 'Martillo', 'Tijeras']

# Función para realizar la predicción
def predict_image(image):
    # Cargar la imagen
    img = load_img(image, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Agregar la dimensión de batch
    
    # Realizar la predicción
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    
    return class_names[class_index], float(np.max(prediction[0]))

# Interfaz de usuario con Streamlit
st.title("Clasificador de Imágenes")
st.write("Sube una imagen para clasificarla.")

# Subir archivo de imagen
uploaded_file = st.file_uploader("Elige una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
    
    # Realizar la predicción
    with st.spinner('Clasificando...'):
        class_name, confidence = predict_image(uploaded_file)
    
    # Mostrar la clase y la confianza
    st.write(f"**Predicción:** {class_name}")
    st.write(f"**Confianza:** {confidence:.2f}")
