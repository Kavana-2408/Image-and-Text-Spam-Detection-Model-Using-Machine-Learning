import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the text spam detection model and vectorizer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model_text = pickle.load(open('model.pkl', 'rb'))

# Load the image spam detection model
with open('spam_image_filter_model.pkl', 'rb') as f:
    model_json = pickle.load(f)
model_image = tf.keras.models.model_from_json(model_json)
model_image.load_weights('spam_image_filter_weights.h5')

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Image preprocessing function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict functions
def predict_text(text):
    transformed_text = transform_text(text)
    vector_input = tfidf.transform([transformed_text])
    return model_text.predict(vector_input)[0]

def predict_image(img_array):
    prediction = model_image.predict(img_array)
    return prediction[0][0]

# Streamlit app
st.title("Spam Detection")

# Choose detection type
option = st.selectbox("Choose detection type:", ["Text", "Image"])

if option == "Text":
    st.subheader("Text Spam Detection")
    input_sms = st.text_area("Enter the message")

    if st.button('Predict Text'):
        result = predict_text(input_sms)
        if result == 1:
            st.header("Spam Detected")
        else:
            st.header("Not Spam")

elif option == "Image":
    st.subheader("Image Spam Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_array = preprocess_image(uploaded_file)
        prediction_result = predict_image(img_array)
        spam_threshold = 0.5

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        if prediction_result >= spam_threshold:
            st.write("The uploaded image is predicted as SPAM.")
        else:
            st.write("The uploaded image is predicted as NOT SPAM.")
