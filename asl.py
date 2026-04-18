import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import time
import sys
import shutil
import tempfile
import subprocess
import wave
import pyaudio
from collections import Counter
import threading

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ================= CONFIGURATION =================
# ASL classifier files
ASL_MODEL_PATH = "model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Piper TTS models
PIPER_MODEL_PATH = os.path.join("models", "piper_tts", "en_US-amy-medium.onnx")
PIPER_CONFIG_PATH = os.path.join("models", "piper_tts", "en_US-amy-medium.onnx.json")

# Emotion mapping (all neutral for now)
EMOTION_PARAMS = {
    "Neutral": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
}

# Recognition settings
HOLD_TIME = 1.5          # seconds to hold sign before adding
HAND_LOST_TIME = 1.5     # seconds without any hand before generating
MAX_WORDS = 10

# Window size
WIN_WIDTH, WIN_HEIGHT = 1200, 700 
WEBCAM_WIDTH, WEBCAM_HEIGHT = 640, 480

# Colors (BGR)
COLOR_BG = (30, 30, 30)
COLOR_PANEL = (50, 50, 50)
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (0, 255, 0)
COLOR_WARNING = (0, 165, 255)

# ========== TTS SETUP (Piper) ==========
def get_piper_command():
    piper_exe = shutil.which("piper")
    if piper_exe:
        return [piper_exe]
    try:
        subprocess.run([sys.executable, "-m", "piper", "--help"],
                       capture_output=True, check=True)
        return [sys.executable, "-m", "piper"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("Piper not found. Install piper-tts or add to PATH.")

def safe_delete(file_path, retries=5, delay=0.2):
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)

def speak_text_piper(text, dominant_emotion="Neutral"):
    if not text:
        return
    params = EMOTION_PARAMS.get(dominant_emotion, EMOTION_PARAMS["Neutral"])
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_file = tmp.name
    print(f"\n[Piper] Generating speech: {text[:100]}...")
    cmd = get_piper_command() + [
        "-m", PIPER_MODEL_PATH,
        "--output_file", wav_file,
        "--length-scale", str(params["length_scale"]),
        "--noise-scale", str(params["noise_scale"]),
        "--noise-w", str(params["noise_w"]),
    ]
    result = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Piper error: {result.stderr}")
        safe_delete(wav_file)
        return
    if os.path.getsize(wav_file) == 0:
        print("Zero-byte audio. Check Piper/model.")
        safe_delete(wav_file)
        return
    try:
        with wave.open(wav_file, 'rb') as wf:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            p.terminate()
    except Exception as e:
        print(f"Playback error: {e}")
    finally:
        safe_delete(wav_file)

# ========== LangChain LLM ==========
llm = OllamaLLM(model="llama3.2")
template = """
Be a helpful sentence generator.
Only generate one sentence given a series of words.
Format the grammar properly.
Keep the meaning behind it don't change it.
Use the emotion markers to properly structure the output
Make the sentence tone to be persuasive and convincing yet simple and concise
Only generate the output text, nothing else

The list of words are {list_of_words} with their emotion in order are {emotions}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

def generate_sentence(words):
    if not words:
        return "No words to generate a sentence."
    try:
        list_of_words = ["Use", "Technology", "Good", "Help", "People"]
        emotions = ["Neutral", "Neutral","Neutral","Neutral","Neutral",]
        result = chain.invoke({"list_of_words": list_of_words, "emotions" : emotions})
        return result.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I couldn't generate a sentence."

# ========== ASL Recognizer Class (MediaPipe Tasks API) ==========
class ASLRecognizer:
    def __init__(self, model_path, encoder_path):
        self.model = joblib.load(model_path)
        self.le = joblib.load(encoder_path)

        # Download hand landmarker model if not present
        MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        MODEL_PATH_LOCAL = "hand_landmarker.task"
        if not os.path.exists(MODEL_PATH_LOCAL):
            print("Downloading MediaPipe hand landmarker model...")
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH_LOCAL)
            print("Download complete.")

        # Initialize MediaPipe HandLandmarker (Tasks API)
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH_LOCAL),
            running_mode=VisionRunningMode.VIDEO,   # for video stream
            num_hands=2,                            # detect up to 2 hands
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = HandLandmarker.create_from_options(options)

        # Hand landmark connections (same as mp.solutions.hands.HAND_CONNECTIONS)
        self.HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),        # thumb
            (0,5),(5,6),(6,7),(7,8),        # index
            (0,9),(9,10),(10,11),(11,12),   # middle
            (0,13),(13,14),(14,15),(15,16), # ring
            (0,17),(17,18),(18,19),(19,20), # pinky
            (5,9),(9,13),(13,17)            # palm
        ]

        # Frame counter for timestamp (milliseconds)
        self.frame_counter = 0

    def predict_from_landmarks(self, hand_landmarks_list):
        """
        hand_landmarks_list: list of NormalizedLandmark lists from detection_result.hand_landmarks
        Returns predicted sign string.
        """
        hand_data = []
        if hand_landmarks_list:
            # Sort hands left to right by wrist x-coordinate (landmark 0)
            sorted_hands = sorted(hand_landmarks_list, key=lambda h: h[0].x)
            for landmarks in sorted_hands[:2]:
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                hand_data.append(row)

        # Pad to exactly two hands (84 zeros per missing hand)
        while len(hand_data) < 2:
            hand_data.append([0.0] * 84)   # 21 landmarks * 4 = 84

        X = np.array([hand_data[0] + hand_data[1]])  # shape (1, 168)
        pred_encoded = self.model.predict(X)[0]
        return self.le.inverse_transform([pred_encoded])[0]

    def process_frame(self, rgb_image):
        """
        Run detection on an RGB image (numpy array).
        Returns detection_result object from HandLandmarker.
        """
        # Convert numpy image to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Timestamp increases by ~33ms per frame (30 fps)
        timestamp_ms = self.frame_counter * 33
        self.frame_counter += 1
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return detection_result

    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draw landmarks and connections on frame.
        hand_landmarks: NormalizedLandmark list for one hand.
        """
        h, w, _ = frame.shape
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            x1, y1 = int(start_point.x * w), int(start_point.y * h)
            x2, y2 = int(end_point.x * w), int(end_point.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw landmarks
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def close(self):
        self.detector.close()

# ========== UI Drawing Helpers ==========
def draw_text_panel(canvas, words_list, status_msg, generated_text=""):
    h, w = canvas.shape[:2]
    panel_x = WEBCAM_WIDTH + 10
    
    # Fill entire right side with panel color
    cv2.rectangle(canvas, (WEBCAM_WIDTH, 0), (w, h), COLOR_PANEL, -1)
    
    # ----- Collected Words Section -----
    cv2.putText(canvas, "Collected Words", (panel_x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2)
    
    y = 90
    for i, word in enumerate(words_list):
        text = f"{i+1}. {word}"
        cv2.putText(canvas, text, (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        y += 40
        if y > h - 200:   # Leave space for generated text
            break
    
    # ----- Generated Sentence Section -----
    if generated_text:
        # Section title
        gen_y = max(y + 20, h - 180)
        cv2.putText(canvas, "Generated Sentence:", (panel_x, gen_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
        gen_y += 35
        
        # Wrap text to fit panel width (approx 500px, ~50 chars per line)
        max_chars_per_line = 50
        words = generated_text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line += (" " if current_line else "") + word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw each line
        for line in lines:
            cv2.putText(canvas, line, (panel_x, gen_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            gen_y += 30
            if gen_y > h - 30:
                break   # Stop if we run out of space
    
    # ----- Status Message (bottom) -----
    cv2.putText(canvas, status_msg, (panel_x, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARNING, 2)

def draw_loading_overlay(canvas, message="Generating sentence..."):
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (WIN_WIDTH, WIN_HEIGHT), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = (WIN_WIDTH - text_size[0]) // 2
    text_y = (WIN_HEIGHT + text_size[1]) // 2
    cv2.putText(canvas, message, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

# Global storage for generated sentence (used across threads)
_generated_sentence_full = ""
_generation_error = None

def generate_and_speak_thread(words):
    """Background task: generate sentence and speak it."""
    global _generated_sentence_full, _generation_error
    try:
        sentence = generate_sentence(words)
        _generated_sentence_full = sentence
        speak_text_piper(sentence, "Neutral")
    except Exception as e:
        _generation_error = str(e)
        _generated_sentence_full = "[Error generating sentence]"

# ========== State Machine ==========
STATE_COLLECTING = "collecting"
STATE_GENERATING = "generating"
STATE_DONE = "done"

def main():
    print("Loading ASL model...")
    global _generated_sentence_full, _generation_error
    recognizer = ASLRecognizer(ASL_MODEL_PATH, LABEL_ENCODER_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    cv2.namedWindow("Sign to Sentence", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign to Sentence", WIN_WIDTH, WIN_HEIGHT)

    # ---------- Application state ----------
    words_list = []
    current_sign = None
    sign_start_time = 0
    hand_lost_time = None

    STATE_COLLECTING = 0
    STATE_GENERATING = 1
    STATE_DONE = 2
    state = STATE_COLLECTING

    generated_sentence_display = ""       # Animated portion shown on screen
    typing_index = 0
    last_typing_update = 0
    TYPING_SPEED = 0.03                   # seconds per character

    status_msg = "Show your hand to begin"
    background_thread = None

    print("\n=== Sign to Sentence Demo ===")
    print("Hold a sign steady for 1.5 seconds to add it.")
    print("Remove hand(s) to generate sentence.")
    print("Press 'q' to quit, 'r' to reset.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe detection
        results = recognizer.process_frame(rgb_frame)

        # Create canvas and place resized webcam feed
        canvas = np.zeros((WIN_HEIGHT, WIN_WIDTH, 3), dtype=np.uint8)
        canvas[:, :] = COLOR_PANEL
        frame_resized = cv2.resize(frame, (WEBCAM_WIDTH, WEBCAM_HEIGHT))
        canvas[0:WEBCAM_HEIGHT, 0:WEBCAM_WIDTH] = frame_resized
        cv2.rectangle(canvas, (0, 0), (WEBCAM_WIDTH, WEBCAM_HEIGHT), COLOR_ACCENT, 2)

        hand_detected = results.hand_landmarks is not None and len(results.hand_landmarks) > 0

        # ---------- State Handling ----------
        if state == STATE_COLLECTING:
            # Reset if hand appears after DONE? Not needed here; we'll handle DONE separately.
            if hand_detected:
                hand_lost_time = None
                # Draw landmarks
                for hand_landmarks in results.hand_landmarks:
                    recognizer.draw_landmarks(
                        canvas[0:WEBCAM_HEIGHT, 0:WEBCAM_WIDTH], hand_landmarks)

                predicted = recognizer.predict_from_landmarks(results.hand_landmarks)

                # Hold logic
                if predicted == current_sign:
                    if time.time() - sign_start_time >= HOLD_TIME:
                        if not words_list or words_list[-1] != predicted:
                            if len(words_list) < MAX_WORDS:
                                words_list.append(predicted)
                                status_msg = f"Added: {predicted}"
                            else:
                                status_msg = "Max words reached"
                        sign_start_time = time.time()
                else:
                    current_sign = predicted
                    sign_start_time = time.time()
                    status_msg = f"Holding: {predicted}"

                # Draw hold progress bar
                if current_sign:
                    hold_progress = min((time.time() - sign_start_time) / HOLD_TIME, 1.0)
                    bar_x = 10
                    bar_y = WEBCAM_HEIGHT - 30
                    bar_w = int(200 * hold_progress)
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 200, bar_y + 15), (100, 100, 100), 2)
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), COLOR_ACCENT, -1)
            else:
                current_sign = None
                if hand_lost_time is None:
                    hand_lost_time = time.time()
                elif time.time() - hand_lost_time >= HAND_LOST_TIME:
                    if words_list:
                        state = STATE_GENERATING
                        status_msg = "Generating sentence..."
                        typing_index = 0
                        generated_sentence_display = ""
                        # Launch background thread
                        background_thread = threading.Thread(
                            target=generate_and_speak_thread,
                            args=(words_list.copy(),),
                            daemon=True
                        )
                        background_thread.start()
                if hand_lost_time:
                    elapsed = time.time() - hand_lost_time
                    status_msg = f"No hand ({elapsed:.1f}s / {HAND_LOST_TIME}s)"
                else:
                    status_msg = "No hand detected"

        elif state == STATE_GENERATING:
            # Typing animation: reveal characters gradually from the global full sentence
            now = time.time()
            if _generated_sentence_full and now - last_typing_update >= TYPING_SPEED:
                if typing_index < len(_generated_sentence_full):
                    typing_index += 1
                    generated_sentence_display = _generated_sentence_full[:typing_index]
                    last_typing_update = now
                else:
                    # Full sentence displayed; wait for thread to finish audio
                    pass

            # Check if background thread has completed
            if background_thread and not background_thread.is_alive():
                # Ensure the full sentence is displayed (in case typing was slower)
                generated_sentence_display = _generated_sentence_full
                state = STATE_DONE
                status_msg = "Done. Show your hand to start over."

        elif state == STATE_DONE:
            # Show full sentence and wait for hand to reset
            if hand_detected:
                # Reset everything for a new session
                words_list = []
                current_sign = None
                sign_start_time = 0
                hand_lost_time = None
                _generated_sentence_full = ""
                generated_sentence_display = ""
                typing_index = 0
                state = STATE_COLLECTING
                status_msg = "Reset. Show your hand."
            else:
                status_msg = "Done. Show your hand to start over."

        # ---------- Draw right panel ----------
        # For GENERATING and DONE states, show the animated/complete sentence
        display_text = generated_sentence_display if state in (STATE_GENERATING, STATE_DONE) else ""
        draw_text_panel(canvas, words_list, status_msg, display_text)

        cv2.imshow("Sign to Sentence", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Manual reset
            words_list = []
            current_sign = None
            sign_start_time = 0
            hand_lost_time = None
            state = STATE_COLLECTING
            _generated_sentence_full = ""
            generated_sentence_display = ""
            typing_index = 0
            status_msg = "Reset. Show your hand."

    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()