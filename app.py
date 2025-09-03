import streamlit as st
from model_helper import predict
import tempfile
from PIL import Image

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Display uploaded image
    image = Image.open(tmp_path)
    st.image(image, caption="Uploaded File", use_container_width=True)

    # Make prediction
    prediction = predict(tmp_path)
    st.info(f"Predicted Class: **{prediction}**")
