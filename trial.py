from fastapi import FastAPI
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
from typing import List
from tensorflow.keras.models import model_from_json
from fastapi import FastAPI, UploadFile, Form,File
from fastapi.responses import JSONResponse,HTMLResponse
import io
app = FastAPI()
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
DATA_FOLDER='dataset'
TRAIN_SAMPLE_FOLDER='train_sample_videos'
TEST_FOLDER='test_videos'
# with open("model.json", "r") as json_file:
#     model_json = json_file.read()
# model = model_from_json(model_json)
model_url = 'https://raw.githubusercontent.com/n0v33n/SIH/main/my_modeldeepfake.keras'
model_path = 'my_modeldeepfake.keras'

# Function to download the model from the GitHub raw URL
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, 'wb') as model_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        model_file.write(chunk)
            print(f"Model downloaded and saved at {model_path}")
        else:
            print(f"Failed to download the model. HTTP Status code: {response.status_code}")
    else:
        print("Model already exists locally")

# Call the function to download the model if it's not present locally
download_model(model_url, model_path)

# Load the model using TensorFlow
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")
feature_extractor=build_feature_extractor()
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask
def sequence_prediction(path):
    frames = load_video(os.path.join(DATA_FOLDER, TEST_FOLDER, path))
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Fake/Real Prediction</title>
    <style>
        body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f5;
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        }

        .container {
        background-color: #fff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        max-width: 500px;
        width: 100%;
        }

        h1 {
        color: #333;
        margin-bottom: 20px;
        font-size: 24px;
        }

        p {
        color: #666;
        margin-bottom: 20px;
        font-size: 14px;
        }

        input[type="file"] {
        display: none;
        }

        .custom-file-upload {
        background-color: #007BFF;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        display: inline-block;
        font-size: 16px;
        transition: background-color 0.3s;
        margin-bottom: 20px;
        }

        .custom-file-upload:hover {
        background-color: #0056b3;
        }

        button {
        background-color: #28a745;
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        margin-top: 20px;
        transition: background-color 0.3s;
        }

        button:hover {
        background-color: #218838;
        }

        .result, .error {
        margin-top: 30px;
        font-size: 18px;
        color: #333;
        }

        .error {
        color: red;
        }

        .loader {
        border: 4px solid #f3f3f3;
        border-radius: 50%;
        border-top: 4px solid #007BFF;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
        display: none;
        }

        @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
        }

        video {
        width: 100%;
        height: auto;
        margin-top: 20px;
        display: none;
        border-radius: 8px;
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.1);
        }
    </style>
    </head>
    <body>

    <div class="container">
        <h1>Fake or Real Video Prediction</h1>
        <p>Upload a video to preview and predict whether it is fake or real.</p>

        <label for="videoInput" class="custom-file-upload">Choose Video</label>
        <input type="file" id="videoInput" accept="video/*">

        <video id="videoPreview" controls></video>

        <button id="predictButton" disabled>Predict</button>

        <div class="loader" id="loader"></div>

        <div class="result" id="result"></div>
        <div class="error" id="error"></div>
    </div>

    <script>
        const videoInput = document.getElementById('videoInput');
        const videoPreview = document.getElementById('videoPreview');
        const predictButton = document.getElementById('predictButton');
        const loader = document.getElementById('loader');
        const resultDiv = document.getElementById('result');
        const errorDiv = document.getElementById('error');

        // Event listener for video file upload
        videoInput.addEventListener('change', function() {
        if (videoInput.files.length > 0) {
            const file = videoInput.files[0];
            const fileURL = URL.createObjectURL(file);
            
            // Set the video preview source and display it
            videoPreview.src = fileURL;
            videoPreview.style.display = 'block';
            predictButton.disabled = false; // Enable predict button
        }
        });

        // Event listener for Predict button
        predictButton.addEventListener('click', async function() {
        resultDiv.innerHTML = '';
        errorDiv.innerHTML = '';
        loader.style.display = 'block';
        predictButton.disabled = true;

        const videoFile = videoInput.files[0];
        const formData = new FormData();
        formData.append('video', videoFile);

        try {
            const response = await fetch('/predict/', {
            method: 'POST',
            body: formData
            });

            const data = await response.json();
            loader.style.display = 'none';

            if (response.ok) {
            resultDiv.innerHTML = `Prediction: <strong>${data.prediction}</strong>`;
            } else {
            errorDiv.textContent = `Error: ${data.message}`;
            }
        } catch (error) {
            loader.style.display = 'none';
            errorDiv.textContent = `An error occurred: ${error.message}`;
        } finally {
            predictButton.disabled = false;
        }
        });
    </script>

</body>
</html>



    """
    return HTMLResponse(content=content)
# POST method to receive a video file
@app.post("/predict/")
async def predict_post(video: UploadFile = File(...)):
    try:
        def predict_video(frames):
            """
            Predict if a video is FAKE or REAL.
            :param frames: A numpy array of shape (1, MAX_SEQ_LENGTH, NUM_FEATURES) containing the video frames' features.
            :return: 'FAKE' or 'REAL'
            """

            # Assuming the input frames are already in the correct format for the model (features extracted)
            frame_mask = np.ones((1, MAX_SEQ_LENGTH), dtype="bool")  # All frames present, so mask is fully '1'

            # Predict using the loaded model
            prediction = model.predict([frames, frame_mask])[0]

            # Return the class label based on the prediction score
            if prediction >= 0.5:
                return "FAKE"
            else:
                return "REAL"

        # Example usage (in web app you would take video input and extract features before this step)
        # Here we assume `video_features` is a preprocessed numpy array of shape (1, MAX_SEQ_LENGTH, NUM_FEATURES)
        video_features = np.random.rand(1, MAX_SEQ_LENGTH, NUM_FEATURES)  # Replace this with actual video feature extraction

        # Predict the class of the video
        result = predict_video(video_features)
        return JSONResponse(content={"filename": video.filename, "prediction": result})
    
    except Exception as e:
        return JSONResponse(content={"message": f"An error occurred: {str(e)}"}, status_code=500)
# GET method for API route instructions
@app.get("/predict/")
async def predict_get():
    return JSONResponse(content={"message": "Please use POST method to upload a file"})

