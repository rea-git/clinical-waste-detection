import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("medicalwaste.keras")

# Label mapping
label_map = {'Body_tissue_or_organ_BT': 0,
 'gauze': 1,
 'glass_equipment_packaging_551_GE': 2,
 'gloves': 3,
 'mask': 4,
 'metal_equipment_packaging_ME': 5,
 'organic_wastes_OW': 6,
 'paper_equipment_packaging__PE': 7,
 'plastic_equipment_packaging_PP': 8,
 'syringe': 9,
 'syringe_needles_SN': 10,
 'tweezers': 11}

# Reverse mapping from index to label
index_to_label = {v: k for k, v in label_map.items()}

# Define the Streamlit app
st.title("Medical Waste Classification")
st.write("Upload an image to predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    target_size = (150, 150)  # Replace with your model's input size
    image = image.resize(target_size)  # Resize to match the model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions, axis=-1)[0]  # Extract class index

    # Debugging: Check predicted index
   # st.write(f"Predicted Index: {predicted_index}")

    # Check if the predicted index exists in the mapping
    if predicted_index in index_to_label:
        predicted_label = index_to_label[predicted_index]
        st.write(f"Predicted Class: **{predicted_label}**")
    else:
        st.error("Predicted index is not in the label mapping. Check the model and label mapping consistency.")
