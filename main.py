from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import tensorflow as tf
import requests

app = FastAPI()

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
DATA_FOLDER = 'dataset'
TEST_FOLDER = 'test_videos'

# Load the pre-trained model

# Replace with your GitHub raw URL
model_url = 'https://github.com/n0v33n/SIH/blob/main/my_modeldeepfake.keras'
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


# Utility function to crop video frames to a center square
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Function to load video and resize its frames
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
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Function to build a feature extractor using InceptionV3
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

feature_extractor = build_feature_extractor()

# Function to prepare video frames for model input
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

# Function to predict if a video is FAKE or REAL
def predict_video(frame_features):
    frame_mask = np.ones((1, MAX_SEQ_LENGTH), dtype="bool")  # All frames present, so mask is fully '1'
    prediction = model.predict([frame_features, frame_mask])[0]
    return "FAKE" if prediction >= 0.5 else "REAL"

# POST method to receive a video file and return prediction
@app.post("/predict/")
async def predict_post(video: UploadFile = File(...)):
    try:
        # Save the uploaded video to a temporary file
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())

        # Load video frames and extract features
        frames = load_video(temp_video_path)
        frame_features, _ = prepare_single_video(frames)

        # Predict the class of the video
        result = predict_video(frame_features)

        # Clean up the temporary video file
        os.remove(temp_video_path)

        return JSONResponse(content={"filename": video.filename, "prediction": result})

    except Exception as e:
        return JSONResponse(content={"message": f"An error occurred: {str(e)}"}, status_code=500)

# GET method for API route instructions
@app.get("/predict/")
async def predict_get():
    return JSONResponse(content={"message": "Please use POST method to upload a file"})
