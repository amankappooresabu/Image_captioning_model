import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow import keras

# Set environment variables to limit TensorFlow's resource usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (errors only)
os.environ["OMP_NUM_THREADS"] = "2"  # Limit number of threads used by TensorFlow
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"  # Limit TensorFlow intra-op parallelism
os.environ["TF_NUM_INTEROP_THREADS"] = "2"  # Limit TensorFlow inter-op parallelism

# Custom layer (if required in your model)
class NotEqual(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x, y = inputs
        return tf.not_equal(x, y)

    def get_config(self):
        config = super(NotEqual, self).get_config()
        return config

# Function to recreate your model architecture
def create_model():
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)
    return keras.Model(inputs, outputs)

# Initialize the Flask app
app = Flask(__name__)

# Load the model once when the Flask app starts
model = create_model()
model.load_weights('my_image_captioning_model.h5')

# Define the desired image size
desired_width = 224
desired_height = 224

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read and process the image using PIL
    img = Image.open(BytesIO(file.read()))
    img = img.resize((desired_width, desired_height))  # Resize image
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img)

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    # Run the app, set debug to False for production
    app.run(debug=False)
