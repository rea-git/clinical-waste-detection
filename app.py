import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model("clinicalwaste.keras")

# Define preprocessing function
def preprocess_image(image):
    """
    Preprocess the uploaded image for the model.
    - Resize to (250, 250, 3)
    - Normalize pixel values
    """
    image = image.resize((250, 250))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define prediction function
def predict_image(image):
    """
    Predict whether the image is clinical waste or not.
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_labels = ["Not Clinical Waste", "Clinical Waste"]
    return class_labels[class_idx], prediction[0][class_idx]

# Streamlit App
st.title("Clinical Waste Detection")
st.write("Upload an image to predict whether it is clinical waste.")

# Upload image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        label, confidence = predict_image(image)
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")
