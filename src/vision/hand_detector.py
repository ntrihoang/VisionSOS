
import cv2
import mediapipe as mp

from src.vision.features import extract_features


class HandDetector:
    def __init__(
        self,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(self, frame, draw=True):
        """
        Phát hiện bàn tay và trích xuất 63 features.

        Returns:
            features: numpy array shape (63,) hoặc None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        if draw:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

        features = extract_features(hand_landmarks)

        return features

    def close(self):
        self.hands.close()
