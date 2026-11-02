import sys
from pathlib import Path

import pandas as pd

# Thêm thư mục gốc vào Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.ml.predict import predict
from src.ml.config import FEATURES_CSV, FEATURE_COLS


def main():

    # Đọc dataset
    df = pd.read_csv(FEATURES_CSV)

    # Lấy một mẫu dữ liệu thật
    sample = df[FEATURE_COLS].iloc[0].to_numpy()

    # Dự đoán
    label, confidence = predict(sample)

    print("Prediction test")
    print("--------------------")
    print(f"Features shape: {sample.shape}")
    print(f"Predicted label: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Actual label: {df['label'].iloc[0]}")


if __name__ == "__main__":
    main()