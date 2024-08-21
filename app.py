import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the model
def load_model():
    try:
        json_file_path = "emotiondetector2.json"
        with open(json_file_path, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={"Sequential": tf.keras.models.Sequential})
        model.load_weights("emotiondetector2.h5")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

# Ensure uploads are allowed
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Extract features from the uploaded image
def extract_features(image):
    feature = np.array(image, dtype='float32')
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Labels for the emotion predictions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # URL for the uploaded image
            image_url = url_for('uploaded_file', filename=filename)

            # Return the result
            return render_template('result.html', expression=prediction_label, image_url=image_url)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
