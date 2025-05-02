# 🧠 Lip Reading App – From Visual Speech to Text

> **"Can we accurately predict spoken words from silent video?"**  
> This project aims to bridge communication gaps where audio fails — through visual speech recognition.

---

## 📌 Introduction

In environments where audio is unreliable or unavailable — such as noisy surroundings, surveillance, or for the hearing-impaired — traditional speech recognition breaks down.

This project presents an end-to-end **deep learning lip reading system** that converts silent videos into text by analyzing only the **lip movements** of the speaker. It integrates cutting-edge techniques in **computer vision**, **sequence modeling**, and **web deployment**.

---

## 🚀 Key Features

- 🔍 **Mouth Region Detection** using MTCNN  
- 🎞 **Spatio-Temporal Feature Extraction** via 3D Convolutional Neural Networks (3D CNNs)  
- 🔁 **Sequence Modeling** with Bidirectional LSTMs  
- 🔡 **Text Alignment** using Connectionist Temporal Classification (CTC Loss)  
- 💡 **Real-Time Web Interface** built with Streamlit  
- 🎯 Evaluated on speaker-independent test sets using GRID corpus  

---

## 🗂 Dataset – GRID Corpus

A widely-used benchmark for visual speech recognition with a structured grammar:

```
<command> <color> <preposition> <letter> <digit> <adverb>
e.g., "bin blue at f two now"
```

- 🎥 1,000 video samples per speaker  
- 👄 75 frames per video (~3 seconds)  
- 👤 34 unique speakers with diverse styles  

---

## 🧱 Model Architecture

```text
MTCNN → 3D CNN → BiLSTM → Dense → CTC Loss
```

- **MTCNN** – Detects facial landmarks to crop the mouth region  
- **3D CNN** – Extracts spatio-temporal features across video frames  
- **BiLSTM** – Models temporal dynamics from both directions  
- **CTC Loss** – Aligns video sequences to character outputs without frame-level labels  

---

## 🔄 Workflow

1. **Extract Lip Region** from video using MTCNN  
2. **Preprocess Data** into standardized, normalized frames  
3. **Train Model** with frame-aligned transcripts  
4. **Predict** character sequences from silent videos  
5. **Serve Results** via a Streamlit web app

---

## 📊 Evaluation Results

| Split               | Speakers         | Purpose                        | WER (%) |
|--------------------|------------------|--------------------------------|---------|
| Train              | 6 speakers (80%) | Model learning                 | —       |
| Validation (Seen)  | 10% from train   | Tune hyperparameters           | 19.6    |
| Test (Seen)        | Remaining from train | Evaluate generalization   | 18.9    |
| Test (Unseen)      | 2 new speakers   | Evaluate speaker-independence  | 34.7    |

---

## 🌐 Live Demo (Streamlit UI)

- Upload a `.mpg` video
- Watch the system extract lips and animate as a GIF
- View the predicted transcription in real-time

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** – Deep learning modeling  
- **OpenCV** – Frame extraction  
- **MTCNN** – Mouth detection  
- **Streamlit** – Frontend application  
- **NumPy, TQDM, ImageIO, FFmpeg** – Utilities  

---

## 🧭 Future Work

- 🔁 Use larger datasets like LRS2/LRS3  
- 🎯 Add attention mechanisms for better focus  
- 🧠 Expand to continuous sentence-level lip reading  
- 🎥 Create interview sync-checking tools  
- 📱 Mobile and edge deployment  

---

## 📁 Project Structure

```
lipreading_project/
│
├── data/              # Raw videos, aligns, processed .npy
├── src/               # Source code
│   ├── data/          # Preprocessing logic
│   ├── model/         # Model architecture and training
│   ├── inference/     # Inference pipeline
│   └── utils/         # GIF and helper functions
├── app/               # Streamlit app interface
├── models/            # Saved checkpoints
├── scripts/           # Entry scripts for training/inference
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 👥 Authors

**Team 5 – Lip Reading App**  
Guided by Prof. Max  
George Washington University

---

## 📜 License

This project is for educational and research purposes only.  
