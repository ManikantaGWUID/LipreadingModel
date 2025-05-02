import tensorflow as tf
from dataProcessing import load_video   # ← reusing from your script
from model import build_model                   # ← reusing your model builder
import string
from mtcnn import MTCNN
import cv2
from typing import List
import os
import numpy as np
# --- Vocabulary mappings
vocab = list(string.ascii_lowercase + " ")
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)
detector = MTCNN()

# ------------------------------
# Configuration
# ------------------------------

EXPECTED_FRAMES = 75
OUTPUT_SIZE = (150, 50)
FPS = 40

# ------------------------------
# Helper Functions
# ------------------------------

def get_invalid_video_paths(root_dir: str, expected_frames: int = EXPECTED_FRAMES) -> List[str]:
    invalid_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mpg'):
                video_path = os.path.join(subdir, file)
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count != expected_frames:
                        invalid_files.append(video_path)
                cap.release()
    return invalid_files

def extract_mouth(points, frame_shape, margin: int = 10):
    left = points[0]['keypoints']['mouth_left']
    right = points[0]['keypoints']['mouth_right']
    top = points[0]['keypoints']['nose']
    x1 = max(left[0] - margin, 0)
    y1 = max(top[1] + 10, 0)
    x2 = min(right[0] + margin, frame_shape[1])
    y2 = min(right[1] + margin * 2, frame_shape[0])
    return y1, y2, x1, x2

def standardize(img: np.ndarray) -> np.ndarray:
    x = img.astype("float32")
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    return (x - mean) / (std + 1e-6)

def load_video(path: str) -> tf.Tensor:
    frames = []
    cap = cv2.VideoCapture(path)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Unable to read frames from {path}")

    results = detector.detect_faces(frame)
    if not results:
        cap.release()
        raise ValueError(f"No face detected in {path}")

    y1, y2, x1, x2 = extract_mouth(results, frame.shape)
    cap.release()
    cap = cv2.VideoCapture(path)

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        mouth = frame[y1:y2, x1:x2, :]
        resized = cv2.resize(mouth, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        smooth = cv2.bilateralFilter(resized, d=5, sigmaColor=1000, sigmaSpace=100)
        frames.append(standardize(smooth))
    cap.release()

    return tf.stack(frames)

# --- Inference Function
def predict_video(path_to_video: str, model_weights: str) -> str:
    model = build_model()
    model.load_weights(model_weights)

    frames = load_video(path_to_video)          # ← reusing preprocessing
    frames = tf.expand_dims(frames, axis=0)            # (1, 75, 50, 150, 3)

    yhat = model.predict(frames)
    decoded, _ = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)
    decoded_text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')
    return decoded_text

if __name__ == "__main__":
    video_path = "../data/s1_processed/bbaf2n.mpg"
    model_weights_path = "..//models//checkpoint.weights.h5"

    result = predict_video(video_path, model_weights_path)
    print(f"Prediction: {result}")