import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Cargar el modelo
first_model = load_model('first_images_clasifier.h5')
second_model = load_model('second_images_clasifier.h5')

# Clases del modelo
classes = ['Alicates', 'Cuchillo', 'Cúter', 'Destornillador', 'Martillo', 'Tijeras']

# Título de la app
st.title("Clasificador de Herramientas con IA")
st.markdown("Sube una imagen y el modelo la clasificará como una de las siguientes clases:")
st.write(", ".join(classes))

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen...", type=None)


if uploaded_file is not None:
    # Mostrar imagen
    st.image(uploaded_file, caption='Imagen subida', use_column_width=True)

    # Preprocesamiento
    img = load_img(uploaded_file, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
 # ----------- Primer modelo -----------
    first_predictions = first_model.predict(img_array)
    first_predicted_index = np.argmax(first_predictions)
    first_predicted_class = classes[first_predicted_index]

    st.subheader("Predicción del Primer Modelo")
    st.write(f"Clase predicha: **{first_predicted_class}**")
    st.bar_chart({clase: float(prob) for clase, prob in zip(classes, first_predictions[0])})

    # ----------- Segundo modelo -----------
    second_predictions = second_model.predict(img_array)
    second_predicted_index = np.argmax(second_predictions)
    second_predicted_class = classes[second_predicted_index]

    st.subheader("Predicción del Segundo Modelo")
    st.write(f"Clase predicha: **{second_predicted_class}**")
    st.bar_chart({clase: float(prob) for clase, prob in zip(classes, second_predictions[0])})
