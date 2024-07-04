import streamlit as st
from keras_preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model
import numpy as np

# Load the trained Keras model
try:
    model = load_model('model.h5')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

def preprocess_image(image):
    img = Image.open(image).convert('L').resize((48, 48))  # Convert to grayscale and resize
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, 48, 48, 1))  # Reshape for model input
    return img_array / 255.0  # Normalize pixel values

st.set_page_config(page_title="Emotion Analyzer", layout="centered")

st.title("Emotion Analyzer")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    submit = st.button('Generate')

    if submit:
        try:
            img_array = preprocess_image(uploaded_file)
            prediction = model.predict(img_array)
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            predicted_emotion = emotions[prediction.argmax()]
            st.write(f"The predicted emotion is: {predicted_emotion}")
        except Exception as e:
            st.error(f"Error predicting emotion: {str(e)}")

