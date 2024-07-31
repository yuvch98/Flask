from flask import Flask, jsonify, request, session, render_template, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import os
from preprocessing import preprocess
import numpy as np
import atexit
import shutil
import cv2
from flasgger import Swagger

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'static/frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER
swagger = Swagger(app)

# Load the model once at startup
model = tf.keras.models.load_model('anomaly_detection_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

@app.route('/')
def home():
    """
    Home page that allows video upload.
    ---
    responses:
      200:
        description: Home page with video upload form
    """
    return render_template('upload.html')

@app.route('/hello', methods=['POST'])
def hello():
    """
    Hello endpoint that returns a greeting message.
    ---
    parameters:
      - in: body
        name: name
        description: Name of the person
        schema:
          type: object
          required:
            - name
          properties:
            name:
              type: string
    responses:
      200:
        description: Greeting message
        schema:
          type: object
          properties:
            message:
              type: string
      400:
        description: Error message
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        # Get JSON data from the request body
        data = request.get_json()

        # Extract the name from the JSON data
        name = data.get('name')

        # Return the name from the JSON body
        return jsonify({
            'message': f'Hello, {name}!'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Upload and process a video.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The video file to upload
    responses:
      200:
        description: File uploaded and processed successfully
        schema:
          type: object
          properties:
            message:
              type: string
            prediction:
              type: string
            frames:
              type: array
              items:
                type: string
      400:
        description: Error message
        schema:
          type: object
          properties:
            error:
              type: string
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Ensure the upload directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        if not os.path.exists(app.config['FRAMES_FOLDER']):
            os.makedirs(app.config['FRAMES_FOLDER'])

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded video
        prediction, frame_paths = process_video(filepath)
        predicted_label = 'Normal Video' if prediction >= 0.5 else 'Anomaly Video'

        # Save the frame paths to the session
        session['frame_paths'] = frame_paths
        session['prediction'] = predicted_label

        return redirect(url_for('display_upload'))

@app.route('/upload', methods=['GET'])
def display_upload():
    """
    Display the uploaded video frames and prediction.
    ---
    responses:
      200:
        description: Display the frames and prediction result
    """
    if 'frame_paths' in session:
        frame_paths = session['frame_paths']
    else:
        frame_paths = []

    prediction = session.get('prediction', 'No Prediction')

    return render_template('display.html', frame_paths=frame_paths, prediction=prediction)

def process_video(filepath):
    # Placeholder for video processing code
    print(f"Processing video: {filepath}")

    # Example: Read video frames and make predictions
    frames_for_prediction, frames_for_display = read_video(filepath)

    # Debug: Print shape of frames
    print(f"Shape of frames for prediction: {frames_for_prediction.shape}")
    print(f"Shape of frames for display: {frames_for_display.shape}")

    # Debug: Print model summary
    print(model.summary())

    prediction = model.predict(frames_for_prediction)[0][0]
    print(f"Prediction: {prediction}")

    # Save frames to the filesystem and get their paths
    frame_paths = save_frames_to_filesystem(frames_for_display)

    return prediction, frame_paths

def read_video(filepath):
    frames_for_prediction, frames_for_display = preprocess(filepath)
    print(frames_for_prediction.shape)
    print(frames_for_display.shape)
    return frames_for_prediction, frames_for_display

def save_frames_to_filesystem(frames):
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_uint8 = frame.astype(np.uint8)  # Ensure proper scaling
        frame_filename = f'frame_{i}.png'
        frame_path = os.path.join(app.config['FRAMES_FOLDER'], frame_filename)
        cv2.imwrite(frame_path, frame_uint8)
        frame_paths.append(f'frames/{frame_filename}')  # Use forward slashes
    return frame_paths

def cleanup_uploads_folder():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    if os.path.exists(app.config['FRAMES_FOLDER']):
        shutil.rmtree(app.config['FRAMES_FOLDER'])
    print("Uploads and frames folders cleared")


atexit.register(cleanup_uploads_folder)


def main():
    app.run(debug=False)