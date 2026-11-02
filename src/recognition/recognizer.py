
"""
recognizer.py

Kết nối ML Predictor với TemporalVerifier.

Input:
    features: 63 đặc trưng landmarks từ MediaPipe.

Output:
    label: Nhãn cử chỉ do model dự đoán.
    confidence: Độ tin cậy của model.
    status: Trạng thái xác minh chuỗi cử chỉ.
"""

import time
import numpy as np

from src.ml.predict import predict
from src.recognition.temporal_verifier import TemporalVerifier


class GestureRecognizer:
    def __init__(
        self,
        confidence_threshold=0.80,
        stable_frames=3,
        sequence_timeout=5.0,
        cooldown=2.0
    ):
        """
        Khởi tạo bộ nhận diện cử chỉ.

        Các tham số được chuyển cho TemporalVerifier.
        """

        self.verifier = TemporalVerifier(
            confidence_threshold=confidence_threshold,
            stable_frames=stable_frames,
            sequence_timeout=sequence_timeout,
            cooldown=cooldown
        )

    def process(self, features, timestamp=None):
        """
        Xử lý một frame.

        Args:
            features: Mảng 63 features từ MediaPipe.
            timestamp: Thời điểm frame được xử lý.

        Returns:
            Dictionary chứa kết quả dự đoán và xác minh.
        """

        # 1. Kiểm tra input
        features = np.asarray(features, dtype=np.float32)

        if features.shape not in [(63,), (1, 63)]:
            raise ValueError(
                f"Expected 63 features, got shape {features.shape}"
            )

        # 2. Lấy thời gian hiện tại nếu không được truyền vào
        if timestamp is None:
            timestamp = time.monotonic()

        # 3. Dự đoán cử chỉ bằng model ML
        label, confidence = predict(features)

        # 4. Xác minh cử chỉ theo thời gian
        verification = self.verifier.update(
            label=label,
            confidence=confidence,
            timestamp=timestamp
        )

        # 5. Kết hợp kết quả
        return {
            "label": label,
            "confidence": confidence,
            **verification
        }

    def reset(self):
        """
        Reset trạng thái xác minh chuỗi SOS.
        """
        self.verifier.reset()
