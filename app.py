import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import sys
import pandas as pd
import requests # Used for API calls to Google
import time

# --- Configuration ---
MODEL_PATH = "material_model.h5"
CLASSES_PATH = "materials.json"
IMG_SIZE = (224, 224)
# 🚨 ACTION PERFORMED: API KEY IS NOW PASTED HERE 
API_KEY = "AIzaSyDB7JydVy57PwOwWAp2E0jVEyyR7liRza8" 

# --- Global State and Artifact Loading ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model and class names."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        st.error("Model or classes file not found. Please run 'python train.py' first.")
        st.stop()

    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, "r") as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Failed to load ML artifacts. Error: {e}")
        sys.exit(1)

# --- LLM Integration for Detailed Analysis (Mars Edition) ---

def generate_recycling_info(predicted_material):
    """Generates detailed, structured recycling information using the Gemini API."""
    
    if not API_KEY:
        st.error("🚨 **API Key Missing!** Please obtain a Google Gemini API key and paste it into the `API_KEY` variable in the `app.py` code.")
        return None

    # 1. Define the LLM's goal (System Instruction) - Focused on MARS
    system_prompt = (
        "You are an expert in Mars In-Situ Resource Utilization (ISRU) and Martian Recycling Strategy. "
        "Your task is to provide a comprehensive, structured guide for recycling a single material using realistic Martian resources and technologies. "
        "The output MUST be a valid JSON object matching the provided schema, with clear, concise, and accurate information. "
        "Do not include any introductory or concluding text outside the JSON block."
    )

    # 2. Define the specific task (User Query) - Focused on MARS
    user_query = (
        f"Generate a detailed analysis for the material: '{predicted_material}', considering a permanent human base on Mars. "
        "Describe its components, feasible recycling methods on Mars, and the resulting Martian infrastructure or product outcomes."
    )

    # 3. Define the desired structured output (Response Schema)
    schema = {
        "type": "OBJECT",
        "properties": {
            "MaterialName": {"type": "STRING", "description": "The common name of the material."},
            "Components": {
                "type": "ARRAY",
                "description": "A list of 3-5 common items/components made from the material and their primary use in a Martian colony (e.g., Water storage, Habitat beams).",
                "items": {"type": "STRING"}
            },
            "RecyclingMethods": {
                "type": "ARRAY",
                "description": "A list of 3 primary recycling steps or methods suitable for a Mars colony.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "Method": {"type": "STRING", "description": "Name of the recycling technique (e.g., Shredding, Solar Melting)."},
                        "Outcome": {"type": "STRING", "description": "The direct product or processed material resulting from this method."}
                    }
                }
            },
            "MissionApplications": {
                "type": "ARRAY",
                "description": "A list of 5 key infrastructure or mission applications that the recycled material can be used for on Mars.",
                "items": {"type": "STRING"}
            },
            "FacilityWorkflow": {
                "type": "ARRAY",
                "description": "A 6-step procedural workflow for processing this material at a Mars facility.",
                "items": {"type": "STRING"}
            },
            "MissionImpactSummary": {"type": "STRING", "description": "A brief, powerful statement about the mission impact and resource efficiency (e.g., Saves X% energy vs. mining)."}
        },
        "propertyOrdering": ["MaterialName", "Components", "RecyclingMethods", "MissionApplications", "FacilityWorkflow", "MissionImpactSummary"]
    }
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }

    # 5. Make the API Call with Retry Logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() 
            result = response.json()
            
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            info = json.loads(json_text)
            return info
        
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            time.sleep(2 ** attempt) 
            if attempt == max_retries - 1:
                st.error(f"Failed to generate analysis after {max_retries} attempts. The service is unreachable or your API key is invalid.")
                return None
    return None

# --- Prediction Function ---

def predict_and_analyze(image_bytes, model, class_names):
    """Processes image bytes, makes a prediction, and returns analysis data."""
    try:
        # 1. Load and preprocess the image
        img = Image.open(image_bytes).convert("RGB")
        
        # 2. Preprocess for model
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Make Prediction
        predictions = model.predict(img_array, verbose=0)[0]
        pred_index = np.argmax(predictions)
        pred_class = class_names[pred_index]
        confidence = predictions[pred_index] * 100
        
        return img, pred_class, predictions, confidence
    except Exception as e:
        st.error(f"Error during image processing or prediction: {e}")
        return None, None, None, None

# --- Streamlit App UI ---

# 1. Load the ML artifacts
model, class_names = load_artifacts()

if model:
    st.set_page_config(page_title="Martian Recycling Classifier", layout="wide")
    st.title("🚀 Martian Recycling Classifier")
    st.header("Mars In-Situ Resource Utilization (ISRU) Assistant")
    st.markdown("Use your camera to capture an item for classification and generate a recycling plan specific to a permanent human base on Mars.")
    st.markdown("---")

    # The Camera Input Component
    captured_image_bytes = st.camera_input("Take a Photo of the Material to Analyze")
    
    # Check if a photo was captured
    if captured_image_bytes is not None:
        
        # 2. Prediction and Analysis
        image, pred_class, predictions, confidence = predict_and_analyze(captured_image_bytes, model, class_names)
        
        if pred_class:
            st.markdown("## Analysis Running...")
            
            # Use columns for visual layout
            col_image, col_results = st.columns([1, 1.5])
            
            with col_image:
                st.image(image, caption="Captured Image", use_column_width=True)
                st.markdown("---")
                st.metric(label="Predicted Material", value=pred_class, delta=f"{confidence:.2f}% Confidence")
            
            with col_results:
                # Display Prediction Scores
                st.markdown("### All Prediction Scores")
                df = pd.DataFrame({
                    'Material': class_names,
                    'Confidence (%)': predictions * 100
                })
                df = df.sort_values(by='Confidence (%)', ascending=False)
                st.bar_chart(df.set_index('Material'))
                
                # Generate detailed info using LLM
                st.markdown("---")
                st.subheader("💡 Martian Recycling Guide")
                
                with st.spinner(f"Generating comprehensive Martian ISRU guide for **{pred_class}**..."):
                    info = generate_recycling_info(pred_class)
                
                if info:
                    # -----------------------------------------------------
                    # OUTPUT START: Highly Structured Martian Report
                    # -----------------------------------------------------

                    st.markdown("## 🏭 Mars Recycling Facility")
                    
                    # 1. Mission Impact
                    st.markdown("### 📈 Mission Impact")
                    impact_summary = info.get('MissionImpactSummary', 'Mission success status is unavailable.')
                    st.success(f"✅ {impact_summary} | Essential for Mars colony sustainability!")
                    
                    # 2. Components Section
                    st.markdown("### 🧬 Material Components and Uses")
                    components = info.get('Components', [])
                    if components:
                        st.markdown("- " + "\n- ".join(components))

                    # 3. Mission Applications
                    st.markdown("### 🚀 Mars Mission Applications")
                    applications = info.get('MissionApplications', [])
                    if applications:
                        st.markdown("- " + "\n- ".join(applications))

                    # 4. Recycling Methods
                    st.markdown("## ⚙️ Mars Recycling Process")
                    methods = info.get('RecyclingMethods', [])
                    if methods:
                        for i, item in enumerate(methods):
                            method = item.get('Method', 'N/A')
                            outcome = item.get('Outcome', 'N/A')
                            st.markdown(f"**Step {i+1}: {method}**")
                            st.info(f"**Outcome:** {outcome}")
                    
                    # 5. Facility Workflow
                    st.markdown("## 🛠️ Mars Recycling Facility Workflow")
                    workflow = info.get('FacilityWorkflow', [])
                    if workflow:
                        for i, step in enumerate(workflow):
                            # Using Streamlit Markdown for bold step numbers
                            st.markdown(f"**{i+1}.** {step}")
                    
                    # -----------------------------------------------------
                    # OUTPUT END
                    # -----------------------------------------------------
