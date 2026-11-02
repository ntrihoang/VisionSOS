
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.config import FEATURES_CSV, FEATURE_COLS, LABEL_COL
from src.ml.predict import predict
from src.recognition.recognizer import GestureRecognizer


CONFIDENCE_THRESHOLD = 0.70

GESTURE_SEQUENCE = [
    "OPEN_PALM",
    "THUMB_TUCKED",
    "THUMB_ENCLOSED"
]


def load_dataset():
    """Đọc dataset landmarks từ CSV."""
    df = pd.read_csv(FEATURES_CSV)

    required_columns = FEATURE_COLS + [LABEL_COL]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def select_valid_samples(df, target_label, num_samples=3):
    """
    Chọn các mẫu có:
    - Actual label đúng với target_label.
    - Model dự đoán đúng target_label.
    - Confidence >= threshold.
    """

    candidates = df[df[LABEL_COL] == target_label]

    selected = []

    for _, row in candidates.iterrows():

        features = row[FEATURE_COLS].to_numpy(dtype=np.float32)

        predicted_label, confidence = predict(features)

        if (
            predicted_label == target_label
            and confidence >= CONFIDENCE_THRESHOLD
        ):
            selected.append(features)

        if len(selected) == num_samples:
            break

    if len(selected) < num_samples:
        raise ValueError(
            f"Not enough valid samples for {target_label}. "
            f"Found {len(selected)}/{num_samples}"
        )

    return selected


def test_sos_sequence():
    print("\n========== SOS INTEGRATION TEST ==========")

    df = load_dataset()

    recognizer = GestureRecognizer(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stable_frames=3,
        sequence_timeout=5.0,
        cooldown=2.0
    )

    timestamp = 0.0
    final_result = None

    # Lần lượt đưa 3 cử chỉ vào Recognizer
    for gesture in GESTURE_SEQUENCE:

        print(f"\n--- Testing {gesture} ---")

        samples = select_valid_samples(
            df,
            target_label=gesture,
            num_samples=3
        )

        for i, features in enumerate(samples, start=1):

            timestamp += 0.1

            result = recognizer.process(
                features,
                timestamp=timestamp
            )

            print(
                f"Frame {i}: "
                f"label={result['label']}, "
                f"confidence={result['confidence']:.4f}, "
                f"state={result['state']}, "
                f"stage={result['stage']}"
            )

            final_result = result

    print("\n========== FINAL RESULT ==========")
    print(final_result)

    assert final_result["confirmed"] is True
    assert final_result["state"] == "SOS_CONFIRMED"
    assert final_result["stage"] == 3

    print("\nTEST PASSED: SOS CONFIRMED!")


if __name__ == "__main__":
    test_sos_sequence()
