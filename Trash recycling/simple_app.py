import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import cv2
import pandas as pd

# Configuration
MODEL_PATH = "material_model.h5"
CLASSES_PATH = "materials.json"
IMG_SIZE = (224, 224)

# Load model and classes
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        st.error("Model or classes file not found. Please run 'python train.py' first.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, "r") as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Failed to load ML artifacts. Error: {e}")
        st.stop()

# Load artifacts
model, class_names = load_artifacts()

st.set_page_config(page_title="Simple Trash Classifier", layout="wide")
st.title("♻️ Simple Trash Material Classifier")
st.markdown("Upload an image or use your camera to classify materials!")

# Image upload option
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

# Camera option
use_camera = st.checkbox("Use Camera Instead")

if use_camera:
    # Simple camera capture
    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None:
        # Convert to numpy array
        bytes_data = camera_input.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Display the captured image
        st.image(camera_input, caption="Captured Image", use_column_width=True)
        
        # Process the image
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        pred_index = np.argmax(predictions)
        pred_class = class_names[pred_index]
        confidence = predictions[pred_index] * 100
        
        # Display results
        st.success(f"**Predicted Material:** {pred_class}")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show all predictions
        st.subheader("All Predictions:")
        df = pd.DataFrame({
            'Material': class_names,
            'Confidence (%)': predictions * 100
        })
        st.dataframe(df, use_container_width=True)

elif uploaded_file is not None:
    # Process uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert to numpy array
    img_array = np.array(image.resize(IMG_SIZE), dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    pred_index = np.argmax(predictions)
    pred_class = class_names[pred_index]
    confidence = predictions[pred_index] * 100
    
    # Display results
    st.success(f"**Predicted Material:** {pred_class}")
    st.metric("Confidence", f"{confidence:.2f}%")
    
    # Show all predictions
    st.subheader("All Predictions:")
    df = pd.DataFrame({
        'Material': class_names,
        'Confidence (%)': predictions * 100
    })
    st.dataframe(df, use_container_width=True)

else:
    st.info("Please upload an image or enable camera to get started!")
