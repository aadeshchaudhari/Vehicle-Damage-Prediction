import streamlit as st
from model_helper import predict
import tempfile

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    try:
        prediction = predict(tmp_path)
        st.info(f"Predicted Class: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
