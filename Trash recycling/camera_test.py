import streamlit as st
import cv2
import numpy as np

st.title("🔍 Camera Test")

# Simple camera test without WebRTC
if st.button("Test Camera"):
    try:
        # Try to access camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            st.success("✅ Camera is accessible!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                st.success("✅ Camera can capture frames!")
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Camera Test Image", use_column_width=True)
            else:
                st.error("❌ Camera cannot capture frames")
        else:
            st.error("❌ Camera is not accessible")
            
        cap.release()
        
    except Exception as e:
        st.error(f"❌ Camera test failed: {e}")

st.info("This test helps diagnose camera issues. If this works, the main app should work too.")
