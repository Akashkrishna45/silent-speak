"""
SilentSpeak AI - Configuration v3.2 (Reverted)
480 features (4 aggregations) to match existing trained models.
"""

import numpy as np

# VOCABULARY
WORDS = [
    "help", "water", "food", "medicine", "pain", "bathroom",
    "yes", "no", "hello", "thanks", "please", "stop",
    "call", "home", "family", "hungry", "thirsty", "emergency",
    "good", "bad"
]

SPEECH_MAP = {
    "help": "I need help.",
    "water": "I need water.",
    "food": "I need food.",
    "medicine": "I need medicine.",
    "pain": "I am in pain.",
    "bathroom": "I need to use the bathroom.",
    "yes": "Yes.",
    "no": "No.",
    "hello": "Hello.",
    "thanks": "Thank you.",
    "please": "Please help me.",
    "stop": "Stop.",
    "call": "Please call someone.",
    "home": "I want to go home.",
    "family": "I need my family.",
    "hungry": "I am hungry.",
    "thirsty": "I am thirsty.",
    "emergency": "This is an emergency.",
    "good": "I am good.",
    "bad": "I do not feel well."
}

# FACIAL LANDMARKS (Lips + Jaw) - 59 points
LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
    310, 311, 312, 13, 82, 81, 80, 191,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
    103, 67, 109, 10, 338
]

# BURST CONFIGURATION
BURST_SIZE = 30           # Frames per burst (~1 second)
RAW_FEATURES_PER_FRAME = len(LANDMARKS) * 2 + 2   # 120 (59*2 + 2)
AGGREGATION_METHODS = ["mean", "std", "max", "min"]
TOTAL_FEATURES = RAW_FEATURES_PER_FRAME * len(AGGREGATION_METHODS)  # 480

# TRAINING
TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# REAL-TIME
SLIDING_WINDOW_SIZE = 30
CONFIDENCE_THRESHOLD = 0.50    # Lowered for better live detection
SPEECH_COOLDOWN = 1.5          # Faster response for demo
PREDICTION_BUFFER_SIZE = 5

# PATHS
DATA_DIR = "data/raw"
MODEL_FILE = "models/silent_speak_model.pkl"
SCALER_FILE = "models/scaler.pkl"
METADATA_FILE = "models/model_metadata.json"

# SHARED FEATURE EXTRACTION
def extract_frame_features(face_landmarks):
    """Extract 120 features from a single frame: 59 landmarks (x,y) + mouth width + mouth height."""
    try:
        row = []
        for idx in LANDMARKS:
            lm = face_landmarks.landmark[idx]
            row.extend([lm.x, lm.y])

        left_corner = face_landmarks.landmark[61]
        right_corner = face_landmarks.landmark[291]
        mouth_width = abs(right_corner.x - left_corner.x)

        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_height = abs(lower_lip.y - upper_lip.y)

        row.append(mouth_width)
        row.append(mouth_height)
        return row
    except Exception:
        return None

def aggregate_burst(features_list):
    """
    Aggregate a burst of frame features into a single feature vector.
    Returns 480 features: mean, std, max, min for each of 120 raw features.
    """
    if len(features_list) == 0:
        return None

    arr = np.array(features_list, dtype=np.float32)

    mean_feats = np.mean(arr, axis=0)
    std_feats = np.std(arr, axis=0)
    max_feats = np.max(arr, axis=0)
    min_feats = np.min(arr, axis=0)

    return np.concatenate([mean_feats, std_feats, max_feats, min_feats])