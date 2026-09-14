import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Bone Maturity Portal", layout="centered")

st.title("Pediatric Bone Maturity Diagnostic Portal")
st.write("Upload a child's hand X-ray to predict their skeletal growth category phase.")
st.warning(" Coursework Prototype Notice: This system is an educational coursework prototype and is NOT a medical device.")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("bone_age_model.h5")

try:
    model = load_my_model()
    st.success("AI Model loaded successfully!")
except Exception as e:
    st.error("Could not find 'bone_age_model.h5'. Make sure you ran Step 9 first.")

uploaded_file = st.file_uploader("Upload an X-ray scan...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("L")
    st.image(pil_image, caption="Uploaded Scan Preview", width=300)
    
    opencv_img = np.array(pil_image)
    resized_img = cv2.resize(opencv_img, (128, 128))
    normalized_img = resized_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized_img, axis=(0, -1))
    
    prediction_scores = model.predict(input_tensor)
    predicted_class_index = int(np.argmax(prediction_scores))
    
    # FIXED: Explicitly cast the array index score to a standard float 
    confidence_value = float(prediction_scores[0][predicted_class_index] * 100)
    
    categories = ["Phase 0: Early Growth Phase", "Phase 1: Mid-Stage Development", "Phase 2: Mature Bone Structure"]
    
    st.subheader("Diagnostic Assessment Result:")
    st.write(f"**Classification:** {categories[predicted_class_index]}")
    st.write(f"**Confidence Index:** {confidence_value:.2f}%")
    st.info(f"**Plain Language Summary:** The model is {confidence_value:.1f}% sure that this hand scan shows features matching {categories[predicted_class_index]}.")