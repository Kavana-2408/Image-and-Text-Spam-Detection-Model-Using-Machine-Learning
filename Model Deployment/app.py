import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Combined Spam Detection", page_icon=":guardsman:", layout="wide")

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the text spam detection model and vectorizer
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
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
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

# Streamlit app content


st.title("\n #! /bin /bash")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("Text Input")
    input_sms = st.text_area("Enter the message", height=200)

with col2:
    st.header("Image Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
if st.button('Predict'):
    if input_sms and uploaded_file:
        transformed_text = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_text])
        image_array = preprocess_image(uploaded_file)

        text_result = predict_text(input_sms)
        image_result = predict_image(image_array)

        text_prediction = 1 if text_result == 1 else 0
        image_prediction = 1 if image_result >= 0.5 else 0

        final_prediction = text_prediction or image_prediction

        st.subheader("Text Prediction")
        st.write("Spam Detected" if text_prediction == 1 else "Not Spam")

        st.subheader("Image Prediction")
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=False
                 )
        st.write("Spam Detected" if image_prediction == 1 else "Not Spam")

        if final_prediction == 1:
            st.header("The input is Predicted as SPAM.")
        else:
            st.header("The input is Predicted as NOT SPAM.")
    else:
        st.error("Please provide both text and image for prediction.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        
    .img{
    height: 40px;
    width:40px;
    }
    </style>
    <div class="footer">
        <p>Made with CSE(AIML) 6th Sem</p>
    </div>
    """, unsafe_allow_html=True)
