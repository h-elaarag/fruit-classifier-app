'''Absolutely! Here's a full app.py Streamlit demo script that:

Loads your .tflite model

Lets you upload an image

Runs inference and shows the prediction with confidence

Has a "Stop App" button to exit'''

# to run: streamlit run app.py


import numpy as np
import streamlit as st
from PIL import Image
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import sys

# Load TFLite model and allocate tensors
#interpreter = tflite.Interpreter(model_path="model_v3_legacy.tflite")
interpreter = tf.lite.Interpreter(model_path="model_v3_legacy.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels must match your model output order
class_labels = ['apple', 'banana', 'orange']

def preprocess_image(image: Image.Image):
    image = image.resize((100, 100))
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # normalize if needed
    image = np.expand_dims(image, axis=0)
    return image

def predict(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_index = np.argmax(output_data)
    confidence = np.max(output_data)
    return pred_index, confidence

st.title("🍎 Fruit Classifier Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    pred_index, confidence = predict(image)
    predicted_class = class_labels[pred_index]

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}")

if st.button("Stop App"):
    st.write("Stopping the app... please close this tab or terminal.")
    sys.exit()
