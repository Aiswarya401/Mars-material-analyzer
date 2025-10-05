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

# Mars Mission Recycling information
RECYCLING_INFO = {
    "Aluminium": {
        "description": "🚀 Critical for Mars missions! Aluminum is essential for spacecraft and habitat construction.",
        "recycling_process": [
            "Collection in Mars habitat",
            "Shredding using Mars recycling equipment",
            "Melting using solar-powered furnaces (660°C)",
            "Casting into structural components"
        ],
        "useful_outputs": [
            "Habitat structural beams and panels",
            "Space suit components and helmets",
            "Rover body parts and wheels",
            "Greenhouse framework for food production",
            "Tools and equipment for Mars exploration",
            "Water storage tanks and pipes",
            "Solar panel mounting systems"
        ],
        "environmental_benefit": "🛸 Saves 95% energy vs. Earth mining! Essential for Mars colony sustainability!"
    },
    "Plastic": {
        "description": "🧴 Vital for Mars survival! Plastic recycling is crucial for long-term missions.",
        "recycling_process": [
            "Collection in Mars habitat",
            "Sorting by plastic type using AI",
            "Shredding into pellets",
            "3D printing into new components"
        ],
        "useful_outputs": [
            "3D printed tools and spare parts",
            "Protective covers for equipment",
            "Insulation for Mars habitat walls",
            "Flexible tubing for life support systems",
            "Containers for food and water storage",
            "Protective gear for Mars exploration",
            "Sealing materials for airlocks"
        ],
        "environmental_benefit": "🌍 Reduces waste on Mars! Essential for closed-loop life support systems!"
    },
    "Glass": {
        "description": "🍶 Perfect for Mars! Glass can be infinitely recycled and is essential for habitats.",
        "recycling_process": [
            "Collection and color sorting",
            "Crushing into cullet using Mars equipment",
            "Melting using concentrated solar energy (1500°C)",
            "Forming into new glass products"
        ],
        "useful_outputs": [
            "Habitat windows and domes",
            "Greenhouse panels for food production",
            "Optical components for telescopes",
            "Protective lenses for space suits",
            "Laboratory equipment and containers",
            "Solar concentrator mirrors",
            "Communication dish components"
        ],
        "environmental_benefit": "☀️ Perfect for Mars solar energy systems! Essential for greenhouse food production!"
    },
    "CircuitBoard": {
        "description": "📱 Mission critical! Electronics recycling is vital for Mars technology maintenance.",
        "recycling_process": [
            "Careful dismantling in Mars lab",
            "Removing hazardous components safely",
            "Shredding into small pieces",
            "Extracting precious metals using Mars-safe chemicals"
        ],
        "useful_outputs": [
            "New circuit boards for Mars equipment",
            "Communication system components",
            "Life support system controls",
            "Rover navigation systems",
            "Habitat monitoring sensors",
            "Solar panel control electronics",
            "Emergency backup systems"
        ],
        "environmental_benefit": "🛰️ Critical for Mars mission success! Prevents equipment failures in space!"
    }
}

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

st.set_page_config(page_title="🚀 Mars Recycling Mission", layout="wide")
st.title("🚀 Mars Recycling Mission")
st.markdown("**Mission Control: Upload images to identify materials and discover Mars recycling protocols!**")

# Image upload option
uploaded_file = st.file_uploader("📸 Upload Mars Mission Material Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Process uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="📷 Mars Mission Material", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Mars Analysis Results")
        
        # Convert to numpy array for processing
        img_array = np.array(image.resize(IMG_SIZE), dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        with st.spinner("🔍 Analyzing Mars material..."):
            predictions = model.predict(img_array, verbose=0)[0]
            pred_index = np.argmax(predictions)
            pred_class = class_names[pred_index]
            confidence = predictions[pred_index] * 100
        
        # Display main result
        if confidence > 50:
            st.success(f"🎯 **Mars Mission Material Detected: {pred_class}**")
            st.metric("Scan Confidence", f"{confidence:.1f}%")
        else:
            st.warning(f"⚠️ **Possible Material: {pred_class}** (Low confidence - rescan recommended)")
            st.metric("Scan Confidence", f"{confidence:.1f}%")
        
        # Show all predictions
        st.subheader("📊 All Material Scores")
        df = pd.DataFrame({
            'Material': class_names,
            'Confidence (%)': predictions * 100
        }).sort_values('Confidence (%)', ascending=False)
        st.dataframe(df, use_container_width=True)
    
    # Mars Mission Recycling information section
    st.markdown("---")
    st.subheader("🚀 Mars Mission Recycling Protocol")
    
    if pred_class in RECYCLING_INFO:
        info = RECYCLING_INFO[pred_class]
        
        # Material description
        st.info(f"**{pred_class}**: {info['description']}")
        
        # Mars recycling process
        st.subheader("🔄 Mars Recycling Process")
        for i, step in enumerate(info['recycling_process'], 1):
            st.write(f"**Step {i}:** {step}")
        
        # Mars mission outputs
        st.subheader("🚀 Mars Mission Applications")
        for output in info['useful_outputs']:
            st.write(f"• {output}")
        
        # Mission benefit
        st.subheader("🛸 Mission Impact")
        st.success(info['environmental_benefit'])
        
        # Mission call to action
        st.markdown("---")
        st.success("🎉 **Mission Success!** Material identified for Mars colony sustainability. Proceed to Mars recycling facility!")
        
    else:
        st.warning("Material information not available. Please try with a different material.")

else:
    # Mars mission instructions
    st.info("👆 **Mars Mission Instructions:**")
    st.markdown("""
    1. **Upload Mars Mission Material Image** using the file uploader above
    2. **Scan mission materials** (spacecraft parts, habitat components, equipment)
    3. **Get instant Mars recycling protocol** and mission applications!
    """)
    
    # Show Mars mission materials
    st.subheader("🚀 Mars Mission Materials")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🚀 Aluminum**")
        st.markdown("Spacecraft parts, habitat beams, rover components")
    
    with col2:
        st.markdown("**🧴 Plastic**")
        st.markdown("3D printing materials, protective gear, life support systems")
    
    with col3:
        st.markdown("**🍶 Glass**")
        st.markdown("Habitat windows, greenhouse panels, optical equipment")
    
    with col4:
        st.markdown("**📱 Electronics**")
        st.markdown("Communication systems, life support controls, navigation")
