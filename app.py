import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io

# Cargar el modelo preentrenado
model = load_model('serialized_model/images_clasifier.h5')

# Títulos y descripción de la app
st.title("Clasificador de Imágenes")
st.write("Esta es una aplicación simple para clasificar imágenes usando un modelo de TensorFlow.")
st.write("Sube una imagen para predecir su clase.")

# Función para procesar la imagen y hacer la predicción
def predict_image(image):
    # Cargar imagen desde el stream de bytes
    img = load_img(io.BytesIO(image.read()), target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch

    # Realizar la predicción
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])  # Clase con mayor probabilidad

    # Nombres de las clases (ajusta esto según las clases de tu modelo)
    class_names = ['Alicates', 'Cuchillo', 'Cúter', 'Destornillador', 'Martillo', 'Tijeras']

    # Obtener la clase y confianza
    predicted_class = class_names[class_index]
    confidence = float(np.max(prediction[0]))

    return predicted_class, confidence

# Subir archivo de imagen
file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if file is not None:
    # Mostrar la imagen subida
    st.image(file, caption="Imagen subida", use_column_width=True)

    # Realizar la predicción
    predicted_class, confidence = predict_image(file)

    # Mostrar resultados
    st.write(f"**Clase Predicha:** {predicted_class}")
    st.write(f"**Confianza:** {confidence:.2f}")

else:
    st.write("Por favor, sube una imagen para clasificarla.")
