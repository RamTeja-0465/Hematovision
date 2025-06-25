import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/hemato_model.h5")
class_names = ['Platelets', 'RBC', 'WBC']

# Streamlit UI
st.title("🩸 HematoVision - Blood Cell Classifier")
uploaded_file = st.file_uploader("Upload a blood cell image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_index = np.argmax(pred)
    st.success(f"Prediction: **{class_names[class_index]}**")
