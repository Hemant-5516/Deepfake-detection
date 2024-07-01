from flask import Flask, request, render_template
import numpy as np
from keras.preprocessing import image
from model1 import *
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    file = request.files['file']
    image_path = 'uploaded_image.jpg'
    file.save(image_path)
    result = validate_image(image_path)
    return result

def validate_image(image_path):
    IMAGE_SIZE = (256, 256)
    loaded_model = load_model('model.h5')
    loaded_model.load_weights('model_weights/checkpoint.weights.h5')
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image
    prediction = loaded_model.predict(img_array)
    if prediction[0] > 0.5:
        result = "Input image is Real"
    else:
        result = "Input image is a deepfake"
    print(result)
    print(prediction[0])
    return result

if __name__ == '__main__':
    app.run(debug=True)