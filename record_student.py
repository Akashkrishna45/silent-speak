"""
SilentSpeak AI - Student Data Recorder
Fast-track: 15 bursts per word, 20 frames per burst
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import sys

# Add PARENT directory (project root) to path since this file is in training/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import extract_frame_features, LANDMARKS

WORDS = [
    "bad", "bathroom", "call", "emergency", "family", "food", "good", "hello", 
    "help", "home", "hungry", "medicine", "no", "pain", "please", "stop", 
    "thanks", "thirsty", "water", "yes"
]

BURSTS_PER_WORD = 15
FRAMES_PER_BURST = 20
COOLDOWN_FRAMES = 10

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def record_student(student_name):
    os.makedirs(f"data/{student_name}", exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not found!")
        return

    print(f"\n{'='*60}")
    print(f"RECORDING FOR: {student_name.upper()}")
    print(f"Words: {len(WORDS)} | Bursts per word: {BURSTS_PER_WORD}")
    print(f"Estimated time: ~12-15 minutes")
    print(f"{'='*60}")
    print("\nINSTRUCTIONS:")
    print("1. Face the camera directly")
    print("2. When you see 'SPEAK NOW', silently mouth the word")
    print("3. Hold the lip position for ~0.7 seconds")
    print("4. Rest when you see 'REST'")
    print("5. Press SPACE to start each word")
    print("6. Press Q anytime to quit\n")

    for word in WORDS:
        print(f"\n--- WORD {WORDS.index(word)+1}/{len(WORDS)}: '{word.upper()}' ---")
        print("Press SPACE when ready...")

        # Wait for space
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"WORD: {word.upper()}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to start", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("SilentSpeak Recorder", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        bursts = []
        burst_count = 0
        frame_buffer = []
        state = "COUNTDOWN"
        countdown = 30

        while burst_count < BURSTS_PER_WORD:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            progress = f"{burst_count}/{BURSTS_PER_WORD}"
            cv2.putText(frame, f"WORD: {word.upper()} | {progress}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            if state == "COUNTDOWN":
                cv2.putText(frame, f"GET READY... {countdown//30 + 1}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                countdown -= 1
                if countdown <= 0:
                    state = "RECORDING"
                    frame_buffer = []

            elif state == "RECORDING":
                cv2.putText(frame, "SPEAK NOW!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                if results.multi_face_landmarks:
                    features = extract_frame_features(results.multi_face_landmarks[0])
                    if features is not None:
                        frame_buffer.append(features)
                else:
                    frame_buffer = []

                if len(frame_buffer) >= FRAMES_PER_BURST:
                    burst_arr = np.array(frame_buffer)
                    bursts.append(burst_arr)
                    burst_count += 1
                    state = "COOLDOWN"
                    cooldown_counter = COOLDOWN_FRAMES
                    frame_buffer = []

            elif state == "COOLDOWN":
                cv2.putText(frame, "REST...", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    state = "COUNTDOWN"
                    countdown = 20

            cv2.imshow("SilentSpeak Recorder", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording aborted!")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Save bursts
        if bursts:
            all_bursts = np.array(bursts)
            aggregated = []
            for burst in all_bursts:
                mean_feat = np.mean(burst, axis=0)
                std_feat = np.std(burst, axis=0)
                min_feat = np.min(burst, axis=0)
                max_feat = np.max(burst, axis=0)
                combined = np.concatenate([mean_feat, std_feat, min_feat, max_feat])
                aggregated.append(combined)

            csv_path = f"data/{student_name}/{word}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in aggregated:
                    writer.writerow(row)

            print(f"  Saved {len(bursts)} bursts -> {csv_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"DONE! Data saved in: data/{student_name}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    name = input("Enter student name (lowercase, no spaces, e.g., 'rahul'): ").strip().lower()
    if not name:
        print("Name required!")
    else:
        record_student(name)