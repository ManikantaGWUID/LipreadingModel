import streamlit as st
import tensorflow as tf
from model import build_model
from PIL import Image
import os
import tempfile
from dataProcessing import tensor_to_gif  # Import your GIF maker
from dataProcessing import load_video  # Reuse your logic
import string
import ffmpeg
from PIL import Image
import imageio
def convert_to_mp4(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)


vocab = list(string.ascii_lowercase + " ")
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# --- Styling ---
st.set_page_config(
  page_title="LIP Reading",
  page_icon="../static/GWU.jpeg",
  layout="wide",              # ← add this
  initial_sidebar_state="auto"
)
import base64

# Utility to convert logo image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Encode the GW logo (you can use any .png or .jpg)
encoded_logo = get_base64_image("../static/GWU.jpeg")  # Or wherever your logo is stored

# --- Inject Header ---
st.markdown(f"""
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .custom-header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 60px;
            background-color: #002c51;
            color: white;
            display: flex;
            align-items: center;
            padding: 0 20px;
            z-index: 10000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .custom-header img {{
            height: 40px;
            margin-right: 15px;
        }}

        .custom-header span {{
            font-size: 20px;
            font-weight: 600;
        }}
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: #EFF6FD  !important;  /* sky blue */
        }}
    </style>

    <div class="custom-header">
        <img src="data:image/png;base64,{encoded_logo}" >
        <span>Lip Reading App</span>
    </div>
""", unsafe_allow_html=True)

# ✅ Push content below fixed header
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)


# --- Upload Section ---
st.markdown(f'<h3 style="color: black;">Upload a video (.mpg format)</h3>', unsafe_allow_html=True)
import cv2

def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 700, 400  # fallback values
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

video_file = st.file_uploader("Upload", type=["mpg"])
def display_resized_gif(gif_path, width, height):
    with open(gif_path, "rb") as f:
        gif_data = f.read()
        b64_encoded = base64.b64encode(gif_data).decode("utf-8")

    gif_html = f"""
    <img src="data:image/gif;base64,{b64_encoded}" width="{width}" height="{height}" style="display: block; margin: auto;" />
    """
    st.markdown(gif_html, unsafe_allow_html=True)
# Check if user uploaded, else load default
if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mpg") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name
else:
    # Use a default video path (ensure this file exists in your project)
    tmp_path = "../defaultVideo/bbae1s.mpg"  # Change this to match your structure


# Convert to .mp4 for display
convert_to_mp4(tmp_path, "converted.mp4")
frames = load_video(tmp_path)
tensor_to_gif(frames, "output.gif")
frames = tf.expand_dims(frames, axis=0)
col1, col2 = st.columns(2)
video_width, video_height = get_video_dimensions("converted.mp4")
with col1:
    st.markdown(f"<h4 style='color: black;'>{'Uploaded Video' if video_file else 'Default Video'}</h4>", unsafe_allow_html=True)

    st.video("converted.mp4")


with col2:
    st.markdown(f"<h4 style='color: black;'>Extracted Lips</h4>", unsafe_allow_html=True)
    display_resized_gif("output.gif", 400, 150)
spinner = st.empty()


# Prediction trigger
if st.button("Predict"):
    spinner = st.empty()
    with spinner.container():
        st.markdown("""
        <div style="position: fixed; top: 50%; left: 50%;
                    transform: translate(-50%, -50%);
                    background-color: #EFF6FD; padding: 20px 40px;
                    border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    z-index: 9999; text-align: center;">
            <div style="border: 6px solid #f3f3f3; border-top: 6px solid #333;
                        border-radius: 50%; width: 40px; height: 40px;
                        animation: spin 1s linear infinite; margin: auto;"></div>
            <p style="margin-top: 12px; font-weight: 600; color: black;">
                Processing...
            </p>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    """, unsafe_allow_html=True)

        model = build_model()
        model.load_weights("../BiggerModel/checkpoint.weights.h5")
        yhat = model.predict(frames)
        decoded, _ = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)
        pred_text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode("utf-8")

        spinner.empty()
    


    st.markdown(f"<h4 style='color: black;'>📝 Predicted Text</h4>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background-color: #d4edda; padding: 1rem; border-radius: 8px;
                border-left: 6px solid #28a745; color: black;'>
        <h4 style='margin: 0;'>{pred_text}</h4>
    </div>
""", unsafe_allow_html=True)
