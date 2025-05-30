from flask import Flask, request, render_template, redirect, url_for, session
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uuid
from flask_session import Session

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load model
model = load_model('mobilenetv2.h5')
classes = ['Bitu Agia', 'Junum Ese']

@app.route('/')
def home():
    image_url = session.get('image_url', None)
    prediction = session.get('prediction', None)
    confidence = session.get('confidence', None)
    return render_template('index.html', image_url=image_url, prediction=prediction, confidence=confidence)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        session['image_url'] = url_for('static', filename='uploads/' + filename)
        session['prediction'] = None
        session['confidence'] = None
        return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    # Simpan file
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Simpan URL file untuk ditampilkan
    image_url = url_for('static', filename='uploads/' + filename)
    session['image_url'] = image_url

    # Load dan klasifikasi gambar
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = classes[1] if prediction > 0.5 else classes[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    session['prediction'] = label
    session['confidence'] = f"{confidence:.2%}"

    return redirect(url_for('home'))



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)