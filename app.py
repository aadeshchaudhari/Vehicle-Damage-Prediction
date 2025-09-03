import streamlit as st
from model_helper import predict

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)

    try:
        prediction = predict(image_path)
        st.success(f"Predicted Class: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
