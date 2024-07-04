import numpy as np
import pickle
import streamlit as st
from keras_preprocessing.image import load_img, img_to_array
from PIL import Image
import sys

# Load the trained model
model = pickle.load(open('artifacts/model.pkl', 'rb'))

def ef(image):
    img = Image.open(image).convert('L')  # Convert image to grayscale
    img = img.resize((48, 48))            # Resize image to 48x48
    feature = img_to_array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def getresponse(image):
    label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    img = ef(image)
    pred = model.predict(img)
    pred_label = label[pred.argmax()]
    return pred_label

st.set_page_config(page_title="Predict the Emotion",
                   page_icon="",
                   layout="centered",
                   initial_sidebar_state="collapsed")
st.header("Emotion Analyzer")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

submit = st.button("Generate")

# Final response
if submit and uploaded_file is not None:
    emotion = getresponse(uploaded_file)
    st.write(f"The predicted emotion is: {emotion}")
else:
    if submit:
        st.write("Please upload an image file.")

sys.stdout.flush()