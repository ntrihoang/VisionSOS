"""
SilentSOS - TV2
Hàm predict() bàn giao cho TV3 / TV4.

Interface:
    predict(features: np.ndarray) -> tuple[str, float]
    - features: shape (63,) hoặc (1, 63), dtype float
    - return: (label, confidence)
      label ∈ LABEL, confidence ∈ [0, 1]
      Không tự chuyển thành "OTHER" theo ngưỡng (TV3 quyết định).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURE_DIM, LABEL, MODEL_PATH

_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model chưa được train: {MODEL_PATH}\n"
                "Chạy: python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features: np.ndarray) -> tuple[str, float]:
    """
    Dự đoán nhãn + confidence từ vector 63 chiều.

    Parameters
    ----------
    features : np.ndarray
        Shape (63,) hoặc (1, 63).

    Returns
    -------
    label : str
        Một trong LABEL.
    confidence : float
        Xác suất của lớp được chọn (max predict_proba).
    """
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    if feats.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} features, got {feats.shape[1]}"
        )

    clf = _load_model()
    proba = clf.predict_proba(feats)[0]
    idx = int(np.argmax(proba))
    label = str(clf.classes_[idx])
    confidence = float(proba[idx])
    return label, confidence


def predict_batch(features: np.ndarray) -> list[tuple[str, float]]:
    """Dự đoán nhiều mẫu cùng lúc. features shape (N, 63)."""
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected (N, {FEATURE_DIM}), got {feats.shape}")

    clf = _load_model()
    proba = clf.predict_proba(feats)
    results = []
    for row in proba:
        idx = int(np.argmax(row))
        results.append((str(clf.classes_[idx]), float(row[idx])))
    return results


if __name__ == "__main__":
    # Smoke test
    print("Smoke test predict()")
    dummy = np.zeros(FEATURE_DIM, dtype=np.float32)
    try:
        label, conf = predict(dummy)
        print(f"  label={label}, confidence={conf:.4f}")
    except FileNotFoundError as e:
        print(f"  (chưa có model) {e}")
