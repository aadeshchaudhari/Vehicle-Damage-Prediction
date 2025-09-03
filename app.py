import streamlit as st
from model_helper import predict
import os

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Save the file to temp path
        image_path = "temp_file.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(image_path, caption="Uploaded File", use_container_width=True)

        # Call prediction
        prediction = predict(image_path)
        st.success(f"Predicted Class: **{prediction}**")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

