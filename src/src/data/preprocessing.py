import os
import cv2
import numpy as np
import tensorflow as tf
import imageio
from mtcnn import MTCNN
from tqdm import tqdm
from typing import List

# ------------------------------
# Configuration
# ------------------------------
detector = MTCNN()
EXPECTED_FRAMES = 75
OUTPUT_SIZE = (150, 50)
FPS = 25

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

def process_and_save_all_videos(input_root: str, output_root: str, invalid_videos: List[str]):
    os.makedirs(output_root, exist_ok=True)
    for speaker in sorted(os.listdir(input_root)):
        speaker_path = os.path.join(input_root, speaker)
        if not os.path.isdir(speaker_path):
            continue

        output_speaker_dir = os.path.join(output_root, speaker)
        os.makedirs(output_speaker_dir, exist_ok=True)

        for video_file in tqdm(os.listdir(speaker_path), desc=f"Processing {speaker}"):
            if not video_file.endswith(".mpg"):
                continue

            video_path = os.path.join(speaker_path, video_file)
            if video_path not in invalid_videos:
                try:
                    frames_tensor = load_video(video_path)
                    frames_np = frames_tensor.numpy()
                    video_id = os.path.splitext(video_file)[0]
                    np.save(os.path.join(output_speaker_dir, f"{video_id}.npy"), frames_np)
                except Exception as e:
                    print(f"Failed on {video_path}: {e}")

def tensor_to_gif(tensor: tf.Tensor, out_path: str = "lip_seq.gif", fps: int = FPS) -> None:
    x = tensor.numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype("uint8")
    imageio.mimsave(out_path, list(x), duration=1 / fps, loop=0)

# ------------------------------
# Main Entry
# ------------------------------

if __name__ == "__main__":
    SOURCE_DATA_DIR = "..\\data"
    OUTPUT_DATA_DIR = "..\\processed_data"

    detector = MTCNN()

    invalid_video_paths = get_invalid_video_paths(SOURCE_DATA_DIR)
    process_and_save_all_videos(SOURCE_DATA_DIR, OUTPUT_DATA_DIR, invalid_video_paths)

    print("✅ All videos processed and saved as .npy files.")
