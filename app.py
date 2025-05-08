import streamlit as st
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import io

# Cargar el modelo preentrenado
model = load_model('serialized_model/images_clasifier.h5')

# Nombres de las clases
class_names = ['Alicates', 'Cuchillo', 'Cúter', 'Destornillador', 'Martillo', 'Tijeras']

# Título de la interfaz
st.title('Clasificación de Herramientas')

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Cargar la imagen
    img = load_img(uploaded_file, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch

    # Mostrar la imagen subida
    st.image(img, caption='Imagen cargada', use_column_width=True)

    # Realizar la predicción
    if st.button('Predecir'):
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])
        class_name = class_names[class_index]
        confidence = np.max(prediction[0])

        # Mostrar resultados
        st.write(f"Clase predicha: {class_name}")
        st.write(f"Confianza: {confidence:.2f}")
