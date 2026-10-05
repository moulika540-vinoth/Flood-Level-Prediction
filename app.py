# FloodLens - simple Streamlit app
# Run with:  streamlit run app.py
import os
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

MODEL_PATH = "floodlens_model.keras"
classes = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"]

st.title("🌊 FloodLens")
st.write("Flood Level Prediction using Deep Learning")


@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


# Check that the model file exists
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please place 'floodlens_model.keras' in the same folder as app.py.")
    st.stop()

model = load_model()

uploaded = st.file_uploader("Upload a flood/car image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    try:
        img = Image.open(uploaded)
        img.load()
    except (UnidentifiedImageError, OSError):
        st.error("This file is not a valid image.")
        st.stop()

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Prepare image: RGB, 128x128, values 0-1, add batch dimension
    x = img.convert("RGB").resize((128, 128))
    x = np.array(x, dtype="float32") / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    best = int(np.argmax(probs))

    st.subheader("Predicted Flood Level: " + classes[best])
    st.write("Confidence: {:.2f}%".format(float(probs[best]) * 100))
