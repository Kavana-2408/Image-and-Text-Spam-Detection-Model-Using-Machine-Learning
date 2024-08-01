import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

# Load the model configuration and weights
with open('spam_image_filter_model.pkl', 'rb') as f:
    model_json = pickle.load(f)
model = tf.keras.models.model_from_json(model_json)
model.load_weights('spam_image_filter_weights.h5')


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_spam(img_array):
    prediction = model.predict(img_array)
    return prediction[0][0]


st.text("All you need is")
st.title("\n #! /bin/bash")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = preprocess_image(uploaded_file)

    prediction_result = predict_spam(img_array)

    spam_threshold = 0.5

    if prediction_result >= spam_threshold:
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("The uploaded image is predicted as SPAM.")
    else:
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("The uploaded image is predicted as NOT SPAM.")
