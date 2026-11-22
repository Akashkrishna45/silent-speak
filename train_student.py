"""
Train model on student data
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

WORDS = [
    "bad", "bathroom", "call", "emergency", "family", "food", "good", "hello", 
    "help", "home", "hungry", "medicine", "no", "pain", "please", "stop", 
    "thanks", "thirsty", "water", "yes"
]

def train_student(student_name, epochs=150, batch_size=16):
    data_dir = f"data/{student_name}"
    if not os.path.exists(data_dir):
        print(f"ERROR: No data found for {student_name}")
        return

    print(f"\n{'='*60}")
    print(f"TRAINING MODEL FOR: {student_name.upper()}")
    print(f"{'='*60}")

    X = []
    y = []

    for idx, word in enumerate(WORDS):
        csv_path = f"{data_dir}/{word}.csv"
        if not os.path.exists(csv_path):
            print(f"  Missing: {word}.csv (will skip)")
            continue

        df = pd.read_csv(csv_path, header=None)
        for _, row in df.iterrows():
            X.append(row.values)
            y.append(idx)
        print(f"  {word:12s}: {len(df):3d} samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\nTotal samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    if len(X) < 50:
        print("ERROR: Too few samples! Need at least 50.")
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Model
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(len(WORDS), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ]

    print(f"\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc*100:.1f}%")
    print(f"{'='*60}")

    # Per-class accuracy
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    print("\nPer-word accuracy:")
    for idx, word in enumerate(WORDS):
        mask = y_test == idx
        if mask.sum() > 0:
            acc = (pred_classes[mask] == idx).mean() * 100
            bar = "█" * int(acc / 5)
            print(f"  {word:12s}: {acc:5.1f}% {bar}")

    # Save
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{student_name}_model.keras")
    joblib.dump(scaler, f"models/{student_name}_scaler.pkl")
    joblib.dump({w: i for i, w in enumerate(WORDS)}, f"models/{student_name}_class_mapping.pkl")

    print(f"\n{'='*60}")
    print(f"SAVED:")
    print(f"  models/{student_name}_model.keras")
    print(f"  models/{student_name}_scaler.pkl")
    print(f"  models/{student_name}_class_mapping.pkl")
    print(f"{'='*60}")

if __name__ == "__main__":
    name = input("Student name to train: ").strip().lower()
    if not name:
        print("Name required!")
    else:
        train_student(name)