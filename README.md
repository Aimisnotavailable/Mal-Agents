
---

# 🤟 Sign to Sentence – Real‑Time ASL Recognition & Speech

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-orange)](https://developers.google.com/mediapipe)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-green)](https://ollama.com/)
[![Piper TTS](https://img.shields.io/badge/TTS-Piper-9cf)](https://github.com/rhasspy/piper)

A real‑time American Sign Language (ASL) recognition pipeline that collects individual signs, generates a coherent natural‑language sentence using a local LLM, and speaks it aloud – **all running offline**.

<p align="center">
  <img src="demo.gif" alt="Demo GIF" width="600"/>
</p>

## ✨ Features

- **Hand landmark detection** using MediaPipe Tasks (`hand_landmarker.task`)
- **Custom ASL classifier** trained on 21‑landmark data (two hands supported)
- **Hold‑to‑add** interaction – hold a sign steady for 1.5s to record it
- **Sentence generation** via Ollama + Llama 3.2 (fully local)
- **Text‑to‑speech** with Piper TTS (offline, high‑quality voices)
- **Typing animation** while audio plays
- **No cloud dependencies** – privacy‑first design

## 🛠️ Tech Stack

| Component            | Technology                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Hand tracking        | [MediaPipe Hand Landmarker Task](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |
| ASL classification   | Scikit‑learn model (`model.pkl` + `label_encoder.pkl`)                      |
| LLM inference        | [Ollama](https://ollama.com/) with `llama3.2`                               |
| Prompt chaining      | [LangChain](https://www.langchain.com/)                                     |
| TTS                  | [Piper](https://github.com/rhasspy/piper) (`en_US-amy-medium`)              |
| GUI / video capture  | OpenCV                                                                      |
| Audio playback       | PyAudio + Python `wave`                                                      |

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sign-to-sentence.git
cd sign-to-sentence
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary><b>📋 requirements.txt</b></summary>

```
opencv-python
mediapipe
joblib
numpy
pyaudio
langchain
langchain-ollama
```
</details>

### 3. Download required models

#### ASL Classifier
Place your trained `model.pkl` and `label_encoder.pkl` in the project root.  
*(If you don't have a trained model, see [Training Your Own Model](#training-your-own-model).)*

#### MediaPipe Hand Landmarker
The script automatically downloads `hand_landmarker.task` on first run.  
You can also download it manually from [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

#### Piper TTS Model
1. Install Piper:
   ```bash
   pip install piper-tts
   ```
2. Download the `en_US-amy-medium` model:
   ```bash
   # Create models directory
   mkdir -p models/piper_tts
   cd models/piper_tts
   wget https://github.com/rhasspy/piper/releases/download/v0.1.0/voice-en_US-amy-medium.onnx
   wget https://github.com/rhasspy/piper/releases/download/v0.1.0/voice-en_US-amy-medium.onnx.json
   ```
   *(Or use `curl` on Windows/macOS.)*

### 4. Install and run Ollama
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2   # Download the model
ollama serve           # Start the Ollama server (runs in background)
```

## 🚀 Usage

Run the main script:
```bash
python asl.py
```

### Controls
- **Hold a sign** for 1.5 seconds – the word is added to the list.
- **Remove both hands** from the frame – sentence generation starts automatically.
- **Press `r`** – reset the collected words.
- **Press `q`** – quit.

### Configuration
You can tweak the following variables at the top of `asl.py`:

| Variable          | Description                               | Default |
|-------------------|-------------------------------------------|---------|
| `HOLD_TIME`       | Seconds to hold sign before adding        | 1.5     |
| `HAND_LOST_TIME`  | Seconds without hands before generating   | 1.5     |
| `MAX_WORDS`       | Maximum words allowed in list             | 10      |
| `TYPING_SPEED`    | Characters per second in typing animation | 0.03    |

## 🧪 Training Your Own ASL Model

If you don't have `model.pkl` and `label_encoder.pkl`, you can train a classifier using your own dataset.  
A sample training script (not included) would:

1. Capture hand landmarks for each ASL sign using MediaPipe.
2. Flatten the 21‑landmark × 4 values = 84 features per hand.
3. Pad/truncate to two hands (168 features).
4. Train a classifier (e.g., Random Forest) and save the model + label encoder.

Example structure:
```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# X.shape = (n_samples, 168)
# y = list of labels

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
```

## 📁 Project Structure

```
.
├── asl.py                  # Main application
├── model.pkl               # Trained ASL classifier
├── label_encoder.pkl       # Label encoder for classifier
├── hand_landmarker.task    # MediaPipe model (auto‑downloaded)
├── models/
│   └── piper_tts/
│       ├── en_US-amy-medium.onnx
│       └── en_US-amy-medium.onnx.json
└── requirements.txt
```

## 🙏 Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) for the hand landmarking model.
- [Ollama](https://ollama.com/) and Meta for Llama 3.2.
- [Piper TTS](https://github.com/rhasspy/piper) for high‑quality offline speech synthesis.
- [LangChain](https://www.langchain.com/) for simplifying LLM prompt chaining.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for the accessibility and open‑source communities.*

---