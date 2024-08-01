
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'spam_image_filter_model (1).h5'
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def model_predict(file_path, model):
    processed_img = preprocess_image(file_path)
    preds = model.predict(processed_img)

    # Threshold for considering if it's spam or not (adjust as needed)
    spam_threshold = 0.5

    if preds >= spam_threshold:
        return "The uploaded image is predicted as SPAM."
    else:
        return "The uploaded image is predicted as NOT SPAM."

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = model_predict(file_path, model)
        return result
    return "No file uploaded."

if __name__ == '__main__':
    app.run(port=5001, debug=True)
