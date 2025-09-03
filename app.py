import streamlit as st
from model_helper import predict
import tempfile

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Create a temporary file for the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)

    # Run prediction
    prediction = predict(tmp_path)
    st.info(f"Predicted Class: **{prediction}**")


